"""Bit semantics of the L-row phases for speculative verification.

A verify pass runs L draft tokens through every per-token phase at once.
The recipes are per-token independent, so an L-row port SHOULD equal L
sequential single-token runs bit for bit — this pins it on hardware for
the two phases that matter first:

  router   the fp32 logits gemv (4 consecutive cols/lane, stride 128,
           fma, descending tree): L rows as L independent warp-sets in
           one dispatch vs L single-row dispatches.

  qmv      the 2-bit affine gs=128 projection recipe (bf16 HFMA
           accumulators, elems_per_thread 16, one warp per output row):
           L activation rows as L independent warp-sets vs L dispatches.

    python benchmarks/maple_lrow_semantics.py
"""

import argparse
import json

import mlx.core as mx

QMV_LROW_SRC = r"""
    // L_ activation rows through the exact 2-bit qmv recipe; one warp per
    // (row, out_row). rows: L_ x K_ fp32 (exact float(bf16) activations);
    // wq packed [N_][K_/16]; sc/bi bf16 [N_][K_/128]; out L_ x N_ bf16.
    const int lane = threadIdx.x & 31;
    const int wrp = threadIdx.x >> 5;
    const int gw = blockIdx.x * (THREADS_ / 32) + wrp;
    const int row = gw / N_;
    const int orow = gw % N_;
    if (row >= L_) return;
    const float* xf = x + (long long)row * K_;
    const unsigned int* wrow =
        reinterpret_cast<const unsigned int*>(wq) + (long long)orow * (K_ >> 4);
    const __nv_bfloat16* srow =
        reinterpret_cast<const __nv_bfloat16*>(sc) + (long long)orow * (K_ >> 7);
    const __nv_bfloat16* brow =
        reinterpret_cast<const __nv_bfloat16*>(bi) + (long long)orow * (K_ >> 7);
    __nv_bfloat16 sums[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) sums[i] = __nv_bfloat16(0.0f);
    for (int base = lane * 16; base < K_; base += 512) {
        const unsigned int word = wrow[base >> 4];
        const __nv_bfloat16 scale = srow[base >> 7];
        const __nv_bfloat16 bias = brow[base >> 7];
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int q = (word >> (2 * i)) & 3;
            const __nv_bfloat16 wdq = __hadd(
                __hmul(__nv_bfloat16(float(q)), scale), bias);
            sums[i] = __hfma(__nv_bfloat16(xf[base + i]), wdq, sums[i]);
        }
    }
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) sum += __bfloat162float(sums[i]);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if (lane == 0)
        out[(long long)row * N_ + orow] = __nv_bfloat16(sum);
"""


ROUTER_LROW_SRC = r"""
    // rows: L_ activations of size K_; out: L_ x N_ fp32 logits.
    // One warp per (row, 8-col tile) — the single-row recipe verbatim,
    // just indexed by which row this warp serves.
    constexpr int NPT = 4;
    const int lane = threadIdx.x & 31;
    const int wrp = threadIdx.x >> 5;
    const int gw = blockIdx.x * (THREADS_ / 32) + wrp;
    const int row = gw / (N_ / 8);
    const int tile = gw % (N_ / 8);
    if (row >= L_) return;
    const int col = tile * 8 + (lane >> 2);
    const int k0 = (lane & 3) * NPT;
    float sum = 0.0f;
    const float* xr = x + (long long)row * K_;
    const float* wr = w + (long long)col * K_;
    for (int k = k0; k < K_; k += 32) {
        sum = fmaf(wr[k], xr[k], sum);
        sum = fmaf(wr[k + 1], xr[k + 1], sum);
        sum = fmaf(wr[k + 2], xr[k + 2], sum);
        sum = fmaf(wr[k + 3], xr[k + 3], sum);
    }
    #pragma unroll
    for (int o = 2; o > 0; o >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, o);
    if ((lane & 3) == 0) out[(long long)row * N_ + col] = sum;
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-experts", type=int, default=256)
    ap.add_argument("--dim", type=int, default=2048)
    args = ap.parse_args()

    N, K = args.n_experts, args.dim
    report = {}

    kern = mx.fast.cuda_kernel(
        name="router_lrow_probe", input_names=["x", "w"],
        output_names=["out"], source=ROUTER_LROW_SRC)

    for L in (2, 4, 8, 16):
        hits = 0
        for t in range(12):
            mx.random.seed(52000 + 31 * L + t)
            x = mx.random.normal((L, K)).astype(mx.float32)
            w = (mx.random.normal((N, K)) * 0.05).astype(mx.float32)
            mx.eval(x, w)
            # reference: L single-row dispatches of the same kernel
            refs = []
            for r in range(L):
                (o,) = kern(
                    inputs=[x[r:r + 1].reshape(-1), w.reshape(-1)],
                    template=[("K_", K), ("N_", N), ("L_", 1),
                              ("THREADS_", 256)],
                    grid=((((N // 8) * 32 + 255) // 256) * 256, 1, 1),
                    threadgroup=(256, 1, 1),
                    output_shapes=[(1, N)], output_dtypes=[mx.float32])
                refs.append(o)
            ref = mx.concatenate(refs, axis=0)
            (got,) = kern(
                inputs=[x.reshape(-1), w.reshape(-1)],
                template=[("K_", K), ("N_", N), ("L_", L),
                          ("THREADS_", 256)],
                grid=(((L * (N // 8) * 32 + 255) // 256) * 256, 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[(L, N)], output_dtypes=[mx.float32])
            mx.eval(got, ref)
            hits += int(mx.array_equal(got, ref).item())
        report[f"router_L{L}"] = f"{hits}/12"

    qkern = mx.fast.cuda_kernel(
        name="qmv_lrow_probe", input_names=["x", "wq", "sc", "bi"],
        output_names=["out"], source=QMV_LROW_SRC)
    QN, QK = 3072, 2048
    for L in (2, 4, 8, 16):
        hits = 0
        for t in range(12):
            mx.random.seed(63000 + 31 * L + t)
            wq = mx.random.randint(0, 2**31 - 1,
                                   (QN, QK // 16)).astype(mx.uint32)
            sc = (mx.random.normal((QN, QK // 128)) * 0.05
                  ).astype(mx.bfloat16)
            bi = (mx.random.normal((QN, QK // 128)) * 0.01
                  ).astype(mx.bfloat16)
            xb = mx.random.normal((L, QK)).astype(mx.bfloat16)
            x = xb.astype(mx.float32)
            mx.eval(wq, sc, bi, x)
            refs = []
            for r in range(L):
                (o,) = qkern(
                    inputs=[x[r:r + 1].reshape(-1), wq, sc, bi],
                    template=[("K_", QK), ("N_", QN), ("L_", 1),
                              ("THREADS_", 256)],
                    grid=(((QN * 32 + 255) // 256) * 256, 1, 1),
                    threadgroup=(256, 1, 1),
                    output_shapes=[(1, QN)], output_dtypes=[mx.bfloat16])
                refs.append(o)
            ref = mx.concatenate(refs, axis=0)
            (got,) = qkern(
                inputs=[x.reshape(-1), wq, sc, bi],
                template=[("K_", QK), ("N_", QN), ("L_", L),
                          ("THREADS_", 256)],
                grid=(((L * QN * 32 + 255) // 256) * 256, 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[(L, QN)], output_dtypes=[mx.bfloat16])
            mx.eval(got, ref)
            hits += int(mx.array_equal(got, ref).item())
        report[f"qmv_L{L}"] = f"{hits}/12"

    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
