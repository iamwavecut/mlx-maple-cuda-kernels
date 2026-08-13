"""Pack-causal SDPA: the last speculative-verify phase.

Draft token i attends to the cache plus drafts 0..i-1 (and itself):
kL_i = base + i + 1. One block per (head, pack-row) runs the proven
kernel_sdpav_1pass recipe with that row's kL over [cache | pack] --
compared bitwise against L sequential stock SDPA calls with the cache
grown one draft at a time.

    python benchmarks/maple_pack_causal_sdpa.py
"""
import argparse
import json

import mlx.core as mx

SRC = r"""
    // gridDim.x == H_ * L_ blocks of 32x32; block b serves head b/L_,
    // pack row b%L_, with kL = BASE_ + row + 1 over [cache | pack].
    constexpr int BN = 32;
    constexpr int VPT = D_ / 32;
    const int lane = threadIdx.x & 31;
    const int wrp = threadIdx.x >> 5;
    const int head = blockIdx.x / L_;
    const int row = blockIdx.x % L_;
    const int kvh = head / GQA_;
    const int kL = BASE_ + row + 1;

    __shared__ float outs[32][33];
    __shared__ float maxs[32];
    __shared__ float sums[32];

    const float scale_log2 = SCALE_ * 1.44269504088896340736f;
    float q[VPT], k[VPT], o[VPT];
    #pragma unroll
    for (int i = 0; i < VPT; ++i) {
        q[i] = scale_log2 * __bfloat162float(
            Q[((long long)row * H_ + head) * D_ + VPT * lane + i]);
        o[i] = 0.0f;
    }
    float max_score = -3.402823466e38f;
    float sum_exp = 0.0f;
    const long long ch = (long long)kvh * BASE_ * D_;
    const long long ph = (long long)kvh * L_ * D_;
    for (int i = wrp; i < kL; i += BN) {
        const __nv_bfloat16* kro = (i < BASE_)
            ? (Kc + ch + (long long)i * D_)
            : (Kp + ph + (long long)(i - BASE_) * D_);
        #pragma unroll
        for (int j = 0; j < VPT; ++j)
            k[j] = __bfloat162float(kro[VPT * lane + j]);
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
        const __nv_bfloat16* vro = (i < BASE_)
            ? (Vc + ch + (long long)i * D_)
            : (Vp + ph + (long long)(i - BASE_) * D_);
        #pragma unroll
        for (int j = 0; j < VPT; ++j)
            o[j] = o[j] * factor
                 + exp_score * __bfloat162float(vro[VPT * lane + j]);
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
            O[((long long)row * H_ + head) * D_ + VPT * wrp + i] =
                __nv_bfloat16(o[i]);
    }
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--dim", type=int, default=128)
    args = ap.parse_args()

    H, KVH, D = args.heads, args.kv_heads, args.dim
    scale = D ** -0.5
    report = {}
    kern = mx.fast.cuda_kernel(
        name="pack_causal_sdpa", input_names=["Q", "Kc", "Vc", "Kp", "Vp"],
        output_names=["O"],
        source=SRC.replace("SCALE_", f"{scale:.17e}f"))

    for base, L in ((200, 4), (200, 8), (700, 16), (1000, 8)):
        hits = 0
        for t in range(10):
            mx.random.seed(71000 + base + L + t)
            kc = mx.random.normal((1, KVH, base, D)).astype(mx.bfloat16)
            vc = mx.random.normal((1, KVH, base, D)).astype(mx.bfloat16)
            kp = mx.random.normal((1, KVH, L, D)).astype(mx.bfloat16)
            vp = mx.random.normal((1, KVH, L, D)).astype(mx.bfloat16)
            qp = mx.random.normal((L, H, D)).astype(mx.bfloat16)
            mx.eval(kc, vc, kp, vp, qp)
            refs = []
            for i in range(L):
                kfull = mx.concatenate([kc, kp[:, :, :i + 1]], axis=2)
                vfull = mx.concatenate([vc, vp[:, :, :i + 1]], axis=2)
                r = mx.fast.scaled_dot_product_attention(
                    qp[i][None, :, None, :], kfull, vfull, scale=scale)
                refs.append(r.reshape(1, H, D))
            ref = mx.concatenate(refs, axis=0)
            (got,) = kern(
                inputs=[qp.reshape(-1), kc.reshape(-1), vc.reshape(-1),
                        kp.reshape(-1), vp.reshape(-1)],
                template=[("D_", D), ("H_", H), ("GQA_", H // KVH),
                          ("L_", L), ("BASE_", base)],
                grid=(H * L * 1024, 1, 1), threadgroup=(1024, 1, 1),
                output_shapes=[(L, H, D)], output_dtypes=[mx.bfloat16])
            mx.eval(got, ref)
            hits += int(mx.array_equal(got, ref).item())
        report[f"pack_base{base}_L{L}"] = f"{hits}/10"

    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
