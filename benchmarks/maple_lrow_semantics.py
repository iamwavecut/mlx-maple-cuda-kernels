"""Bit semantics of the L-row phases for speculative verification.

A verify pass runs L draft tokens through every per-token phase at once.
The recipes are per-token independent, so an L-row port SHOULD equal L
sequential single-token runs bit for bit — this pins it on hardware for
the two phases that matter first:

  router   the fp32 logits gemv (4 consecutive cols/lane, stride 128,
           fma, descending tree): L rows as L independent warp-sets in
           one dispatch vs L single-row dispatches.

(The 2-bit qmv L-row probe follows the same pattern and lands with the
port itself.)

    python benchmarks/maple_lrow_semantics.py
"""

import argparse
import json

import mlx.core as mx

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

    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
