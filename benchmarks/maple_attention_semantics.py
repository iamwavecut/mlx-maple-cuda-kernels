"""Bit semantics of the decode attention ops, toward an attention megakernel.

Two ports, each compared bitwise against the stock op on decode shapes:

  sdpa   kernel_sdpav_1pass (the kL <= 1024 route with cudnn SDPA disabled):
         BN=32 warps x 32 lanes, D/32 values per lane, queries pre-scaled by
         scale*log2e, keys interleaved per warp with stride 32, base-2 online
         softmax (exp2f), xor-butterfly reductions, cross-warp merge through
         shared memory with __frcp_rn, one cast to bf16 at the write.

  gemv   the dense bf16 qkv/o_proj path: gemv_single with a float
         accumulator, four consecutive columns per lane then stride 128,
         fma-contracted, descending shuffle tree, one bf16 rounding.

The kL > 1024 route (kernel_sdpav_2pass_*) is not pinned here yet.

    python benchmarks/maple_attention_semantics.py
"""

import argparse
import json

import mlx.core as mx

SDPA_SRC = r"""
    // One block per (batch=0, q_seq=0, head): gridDim.x == H_.
    constexpr int BN = 32;
    constexpr int VPT = D_ / 32;  // values per thread
    const int lane = threadIdx.x & 31;
    const int wrp = threadIdx.x >> 5;
    if (wrp >= BN) return;
    const int head = blockIdx.x;
    const int kvh = head / GQA_;

    __shared__ float outs[BN][33];
    __shared__ float maxs[BN];
    __shared__ float sums[BN];

    const float scale_log2 = SCALE_ * 1.44269504088896340736f;
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
    for (int i = wrp; i < KL_; i += BN) {
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

    if (lane == 0) { maxs[wrp] = max_score; sums[wrp] = sum_exp; }
    __syncthreads();
    max_score = maxs[lane];
    float new_max = max_score;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        new_max = fmaxf(new_max, __shfl_xor_sync(0xffffffffu, new_max, off));
    const float factor = exp2f(max_score - new_max);
    float se = sums[lane] * factor;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        se += __shfl_xor_sync(0xffffffffu, se, off);
    se = (se == 0.0f) ? 0.0f : __frcp_rn(se);

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

GEMV_SRC = r"""
    constexpr int NPT = 4;
    const int lane = threadIdx.x & 31;
    const int wrp = threadIdx.x >> 5;
    const int row = blockIdx.x * (THREADS_ / 32) + wrp;
    if (row >= NROWS_) return;
    float sum = 0.0f;
    const __nv_bfloat16* wrow = mat + (long long)row * K_;
    for (int col = NPT * lane; col < K_; col += 32 * NPT) {
        sum = fmaf(__bfloat162float(wrow[col]),
                   __bfloat162float(vec[col]), sum);
        sum = fmaf(__bfloat162float(wrow[col + 1]),
                   __bfloat162float(vec[col + 1]), sum);
        sum = fmaf(__bfloat162float(wrow[col + 2]),
                   __bfloat162float(vec[col + 2]), sum);
        sum = fmaf(__bfloat162float(wrow[col + 3]),
                   __bfloat162float(vec[col + 3]), sum);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, o);
    if (lane == 0) out[row] = __nv_bfloat16(sum);
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

    # floats cannot ride the template list; bake the scale into the source
    sdpa_kern = mx.fast.cuda_kernel(
        name="sdpa_port", input_names=["Q", "Kc", "Vc"], output_names=["O"],
        source=SDPA_SRC.replace("SCALE_", f"{scale:.17e}f"))
    for kl in (17, 64, 333, 640, 1024):
        hits = 0
        for t in range(12):
            mx.random.seed(12000 + 31 * kl + t)
            q = mx.random.normal((1, H, 1, D)).astype(mx.bfloat16)
            k = mx.random.normal((1, KVH, kl, D)).astype(mx.bfloat16)
            v = mx.random.normal((1, KVH, kl, D)).astype(mx.bfloat16)
            mx.eval(q, k, v)
            ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
            (got,) = sdpa_kern(
                inputs=[q.reshape(-1), k.reshape(-1), v.reshape(-1)],
                template=[("D_", D), ("KL_", kl), ("GQA_", H // KVH),
                          ("H_", H)],
                grid=(H * 32 * 32, 1, 1), threadgroup=(32 * 32, 1, 1),
                output_shapes=[(H * D,)], output_dtypes=[mx.bfloat16])
            mx.eval(got, ref)
            hits += int(mx.array_equal(got, ref.reshape(-1)).item())
        report[f"sdpa_1pass_kL{kl}"] = f"{hits}/12"

    gemv_kern = mx.fast.cuda_kernel(
        name="bf16_gemv_port", input_names=["mat", "vec"],
        output_names=["out"], source=GEMV_SRC)
    for n, k in ((3072, 2048), (2048, 2048)):
        hits = 0
        for t in range(12):
            mx.random.seed(15000 + n + t)
            w = (mx.random.normal((n, k)) * 0.05).astype(mx.bfloat16)
            x = mx.random.normal((k,)).astype(mx.bfloat16)
            mx.eval(w, x)
            ref = (x.reshape(1, k) @ w.T).reshape(-1)
            (got,) = gemv_kern(
                inputs=[w, x],
                template=[("K_", k), ("NROWS_", n), ("THREADS_", 256)],
                grid=((n // 8) * 256, 1, 1), threadgroup=(256, 1, 1),
                output_shapes=[(n,)], output_dtypes=[mx.bfloat16])
            mx.eval(got, ref)
            hits += int(mx.array_equal(got, ref).item())
        report[f"gemv_bf16_{n}x{k}"] = f"{hits}/12"

    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
