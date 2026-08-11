"""Bit-exact reproduction of the stock MoE matmul, outside the model.

The megakernel's fast lane is within ~1 ULP instead of array-exact because a
software fp32 loop cannot reproduce `qmm_naive`, the kernel the stock path
dispatches to for a decode step's gathered experts (M*B >= 8). This probe
demonstrates the reproduction that closes that gap, column for column:

  - dequant exactly as `cute_dequant`: bf16(bf16(q * s) + z)
  - the same tensor-core atom, `mma.sync.aligned.m16n8k16.row.col.f32.bf16.
    bf16.f32`, with only row 0 of the A fragment populated
  - k-tiles of max(64, group_size) accumulated in order into fp32, eight
    16-wide atoms per tile, one bf16 rounding in the epilogue

Each output column's value depends only on the k-order, so any grid layout
over columns preserves bits — which is what makes an array-exact expert
phase inside the megakernel possible.

Prints one JSON line: whether every column of a real expert's up_gate and
down projections matches `mx.gather_qmm` bit for bit (dispatched with B=8 so
the reference really is qmm_naive), plus the gather_qmv comparison for
contrast.

    python benchmarks/maple_qmm_naive_repro.py --model model-cuda
"""

import argparse
import json

import mlx.core as mx

SRC = r"""
    // One warp per n8 column octet; each warp walks the full K reduction
    // itself, because bits live in the k-order and nowhere else.
    constexpr int K = K_;
    constexpr int N = N_;
    constexpr int GS = 128;
    constexpr int WARPS = THREADS_ / 32;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    for (int oct = blockIdx.x * WARPS + warp; oct * 8 < N;
         oct += GRID_ * WARPS) {
        const int col = oct * 8 + (lane >> 2);
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

        for (int kt = 0; kt < K / 128; ++kt) {
            #pragma unroll
            for (int ka = 0; ka < 128 / 16; ++ka) {
                const int kbase = kt * 128 + ka * 16;
                const int akol = (lane & 3) * 2;
                const bool arow0 = (lane >> 2) == 0;
                __nv_bfloat16 zero = __nv_bfloat16(0.0f);
                __nv_bfloat16 a0 = arow0 ? x[kbase + akol] : zero;
                __nv_bfloat16 a1 = arow0 ? x[kbase + akol + 1] : zero;
                __nv_bfloat16 a4 = arow0 ? x[kbase + 8 + akol] : zero;
                __nv_bfloat16 a5 = arow0 ? x[kbase + 8 + akol + 1] : zero;

                __nv_bfloat16 bfrag[4];
                #pragma unroll
                for (int half = 0; half < 2; ++half) {
                    #pragma unroll
                    for (int piece = 0; piece < 2; ++piece) {
                        const int k = kbase + half * 8 + (lane & 3) * 2 + piece;
                        const unsigned int word =
                            wq[(long long)col * (K / 16) + (k / 16)];
                        const int q = (word >> (2 * (k % 16))) & 3;
                        const int g = k / GS;
                        const __nv_bfloat16 s =
                            sc[(long long)col * (K / GS) + g];
                        const __nv_bfloat16 z =
                            bi[(long long)col * (K / GS) + g];
                        bfrag[half * 2 + piece] =
                            __hadd(__hmul(__nv_bfloat16(float(q)), s), z);
                    }
                }

                unsigned a01 = (unsigned(__bfloat16_as_ushort(a1)) << 16)
                             | unsigned(__bfloat16_as_ushort(a0));
                unsigned a45 = (unsigned(__bfloat16_as_ushort(a5)) << 16)
                             | unsigned(__bfloat16_as_ushort(a4));
                unsigned azz = 0u;
                unsigned b01 = (unsigned(__bfloat16_as_ushort(bfrag[1])) << 16)
                             | unsigned(__bfloat16_as_ushort(bfrag[0]));
                unsigned b23 = (unsigned(__bfloat16_as_ushort(bfrag[3])) << 16)
                             | unsigned(__bfloat16_as_ushort(bfrag[2]));
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc0), "+f"(acc1), "+f"(acc2), "+f"(acc3)
                    : "r"(a01), "r"(azz), "r"(a45), "r"(azz),
                      "r"(b01), "r"(b23));
            }
        }

        if ((lane >> 2) == 0) {
            out[oct * 8 + 2 * (lane & 3)] = __nv_bfloat16(acc0);
            out[oct * 8 + 2 * (lane & 3) + 1] = __nv_bfloat16(acc1);
        }
    }
"""

_kernel_cache = {}


def _kernel():
    k = _kernel_cache.get("k")
    if k is None:
        k = _kernel_cache["k"] = mx.fast.cuda_kernel(
            name="maple_qmm_naive_repro",
            input_names=["x", "wq", "sc", "bi"],
            output_names=["out"],
            source=SRC,
        )
    return k


def reproduce(x, proj, expert):
    n, k = proj.output_dims, proj.input_dims
    (got,) = _kernel()(
        inputs=[x, proj.weight[expert].view(mx.uint32).reshape(-1),
                proj.scales[expert].reshape(-1),
                proj.biases[expert].reshape(-1)],
        template=[("K_", k), ("N_", n), ("THREADS_", 256), ("GRID_", 32)],
        grid=(32 * 256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(n,)], output_dtypes=[mx.bfloat16],
    )
    mx.eval(got)
    return got


def reference(x, proj, expert, repeats):
    k = proj.input_dims
    idx = mx.full((1, repeats), expert, dtype=mx.uint32)
    ref = mx.gather_qmm(
        x.reshape(1, 1, 1, k), proj.weight, proj.scales, proj.biases,
        rhs_indices=idx, transpose=True, group_size=128, bits=2,
    ).reshape(repeats, -1)
    mx.eval(ref)
    return ref[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--experts", type=int, nargs="+", default=[0, 5, 17, 42])
    args = ap.parse_args()

    from mlx_lm import load
    model, _, _ = load(
        args.model, return_config=True,
        model_config={"model_file": None, "use_flash_head": False},
        tokenizer_config={"trust_remote_code": False},
        trust_remote_code=False)
    mlp = model.model.layers[args.layer].mlp.switch_mlp

    mx.random.seed(3)
    report = {"layer": args.layer, "projections": {}}
    for name, proj in (("up_gate", mlp.up_gate_proj), ("down", mlp.down_proj)):
        x = mx.random.normal((proj.input_dims,)).astype(mx.bfloat16)
        mx.eval(x)
        exact = 0
        qmv_exact = 0
        for e in args.experts:
            got = reproduce(x, proj, e)
            naive = reference(x, proj, e, 8)   # B=8 -> qmm_naive
            qmv = reference(x, proj, e, 1)     # B=1 -> gather_qmv
            exact += int(mx.array_equal(got, naive).item())
            qmv_exact += int(mx.array_equal(got, qmv).item())
        report["projections"][name] = {
            "columns": proj.output_dims,
            "experts_tested": len(args.experts),
            "bit_exact_vs_qmm_naive": exact,
            "bit_exact_vs_gather_qmv": qmv_exact,
        }
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
