"""Bit semantics of the stock kL>1024 decode SDPA (kernel_sdpav_2pass_*).

The stock CUDA route for decode attention switches at kL > 1024 to a
two-kernel scheme:

  pass 1  32 slabs per head, each a (BN=8 warps x 32 lanes) block; keys are
          interleaved per warp with stride 256 (slab*8 + warp), base-2 online
          softmax per warp, warp merge through shared memory with the -1e9
          lane mask and a LINEAR j=1..7 fold, fp32 partials scaled to the
          slab max written to scratch (partials, sums, maxs per slab).

  pass 2  one 32x32 block per head; lane l owns slab l's max/sum, the global
          max/sum come from xor reductions, each warp w holds slab w's
          partial, a transposed shared merge folds 32 slabs per component,
          __frcp_rn of the global sum, one cast to bf16 at the write.

Both are ported here as standalone kernels sharing an fp32 scratch and
compared bitwise against mx.fast.scaled_dot_product_attention on decode
shapes (cudnn SDPA disabled).

    python benchmarks/maple_attention_2pass_semantics.py
"""

import argparse
import json

import mlx.core as mx

PASS1_SRC = r"""
    // One virtual block per (head, slab): gridDim.x == H_ * 32.
    constexpr int BN = 8;
    constexpr int VPT = D_ / 32;  // values per thread
    const int lane = threadIdx.x & 31;
    const int wrp = threadIdx.x >> 5;
    if (wrp >= BN) return;
    const int head = blockIdx.x >> 5;
    const int slab = blockIdx.x & 31;
    const int kvh = head / GQA_;

    __shared__ float outs[BN][33];
    __shared__ float maxs_s[BN];
    __shared__ float sums_s[BN];

    const float scale_log2 = SCALE_ * 1.44269504089f;
    float q[VPT], k[VPT], o[VPT];
    #pragma unroll
    for (int i = 0; i < VPT; ++i) {
        q[i] = scale_log2 * __bfloat162float(
            Q[(long long)head * D_ + VPT * lane + i]);
        o[i] = 0.0f;
    }
    float max_score = -3.402823466e38f;
    float sum_exp = 0.0f;

    const long long kh = (long long)kvh * KL_ * D_;
    for (int i = slab * BN + wrp; i < KL_; i += 32 * BN) {
        #pragma unroll
        for (int j = 0; j < VPT; ++j)
            k[j] = __bfloat162float(Kc[kh + (long long)i * D_ + VPT * lane + j]);
        float score = 0.0f;
        #pragma unroll
        for (int j = 0; j < VPT; ++j) score += q[j] * k[j];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            score += __shfl_xor_sync(0xffffffffu, score, off);
        const float new_max = fmaxf(max_score, score);
        const float factor = exp2f(max_score - new_max);
        const float exp_score = exp2f(score - new_max);
        max_score = new_max;
        sum_exp = sum_exp * factor + exp_score;
        #pragma unroll
        for (int j = 0; j < VPT; ++j)
            o[j] = o[j] * factor + exp_score * __bfloat162float(
                Vc[kh + (long long)i * D_ + VPT * lane + j]);
    }

    if (lane == 0) { maxs_s[wrp] = max_score; sums_s[wrp] = sum_exp; }
    __syncthreads();
    float wmax = (lane < BN) ? maxs_s[lane] : -1e9f;
    float new_max = wmax;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        new_max = fmaxf(new_max, __shfl_xor_sync(0xffffffffu, new_max, off));
    const float factor = exp2f(wmax - new_max);
    float se = (lane < BN) ? sums_s[lane] : 0.0f;
    se *= factor;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        se += __shfl_xor_sync(0xffffffffu, se, off);

    const long long p = (long long)head * 32 + slab;
    if (wrp == 0 && lane == 0) { sums[p] = se; maxs[p] = new_max; }

    const float ff = exp2f(maxs_s[wrp] - new_max);
    #pragma unroll
    for (int i = 0; i < VPT; ++i) {
        outs[wrp][lane] = o[i] * ff;
        __syncthreads();
        if (wrp == 0) {
            float ot = outs[0][lane];
            #pragma unroll
            for (int j = 1; j < BN; ++j) ot += outs[j][lane];
            o[i] = ot;
        }
        __syncthreads();
    }
    if (wrp == 0) {
        #pragma unroll
        for (int i = 0; i < VPT; ++i)
            partials[p * D_ + VPT * lane + i] = o[i];
    }
"""

PASS2_SRC = r"""
    // One block per head: gridDim.x == H_, 32x32 threads.
    constexpr int VPT = D_ / 32;
    const int lane = threadIdx.x & 31;
    const int wrp = threadIdx.x >> 5;
    const int head = blockIdx.x;

    __shared__ float outs[32][33];

    const long long p0 = (long long)head * 32;
    float bmax = maxs[p0 + lane];
    float new_max = bmax;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        new_max = fmaxf(new_max, __shfl_xor_sync(0xffffffffu, new_max, off));
    const float factor = exp2f(bmax - new_max);
    float se = sums[p0 + lane] * factor;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        se += __shfl_xor_sync(0xffffffffu, se, off);
    se = (se == 0.0f) ? 0.0f : __frcp_rn(se);

    float o[VPT];
    #pragma unroll
    for (int i = 0; i < VPT; ++i)
        o[i] = partials[(p0 + wrp) * D_ + VPT * lane + i];

    #pragma unroll
    for (int i = 0; i < VPT; ++i) {
        outs[lane][wrp] = o[i];
        __syncthreads();
        float ot = outs[wrp][lane] * factor;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            ot += __shfl_xor_sync(0xffffffffu, ot, off);
        o[i] = ot * se;
        __syncthreads();
    }
    if (lane == 0) {
        #pragma unroll
        for (int i = 0; i < VPT; ++i)
            O[(long long)head * D_ + VPT * wrp + i] = __nv_bfloat16(o[i]);
    }
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--dim", type=int, default=128)
    args = ap.parse_args()

    report = {}
    H, KVH, D = args.heads, args.kv_heads, args.dim
    scale = D ** -0.5

    p1 = mx.fast.cuda_kernel(
        name="sdpa_2pass_1_port",
        input_names=["Q", "Kc", "Vc"],
        output_names=["partials", "sums", "maxs"],
        source=PASS1_SRC.replace("SCALE_", f"{scale:.17e}f"))
    p2 = mx.fast.cuda_kernel(
        name="sdpa_2pass_2_port",
        input_names=["partials", "sums", "maxs"],
        output_names=["O"],
        source=PASS2_SRC)

    for kl in (1025, 1360, 2048, 3333, 4096, 8192):
        hits = 0
        for t in range(12):
            mx.random.seed(21000 + 31 * kl + t)
            q = mx.random.normal((1, H, 1, D)).astype(mx.bfloat16)
            k = mx.random.normal((1, KVH, kl, D)).astype(mx.bfloat16)
            v = mx.random.normal((1, KVH, kl, D)).astype(mx.bfloat16)
            mx.eval(q, k, v)
            ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
            partials, sums, maxs = p1(
                inputs=[q.reshape(-1), k.reshape(-1), v.reshape(-1)],
                template=[("D_", D), ("KL_", kl), ("GQA_", H // KVH)],
                grid=(H * 32 * 256, 1, 1), threadgroup=(256, 1, 1),
                output_shapes=[(H * 32 * D,), (H * 32,), (H * 32,)],
                output_dtypes=[mx.float32, mx.float32, mx.float32])
            (got,) = p2(
                inputs=[partials, sums, maxs],
                template=[("D_", D)],
                grid=(H * 1024, 1, 1), threadgroup=(1024, 1, 1),
                output_shapes=[(H * D,)], output_dtypes=[mx.bfloat16])
            mx.eval(got, ref)
            hits += int(mx.array_equal(got, ref.reshape(-1)).item())
        report[f"sdpa_2pass_kL{kl}"] = f"{hits}/12"

    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
