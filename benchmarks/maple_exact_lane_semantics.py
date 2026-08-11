"""The bit-level semantics of every stock op the exact fast lane must match.

Companion to `maple_qmm_naive_repro.py` (the expert matmuls). Each section
establishes, by bitwise comparison on hardware, the reduction order of one
stock building block of the router / aggregation chain:

  logits      x32 @ W32.T -> MLX gemv_single: fp32, four consecutive columns
              per lane then a 128 stride, fma-contracted, descending
              shfl tree (equivalently cg::reduce's butterfly at lane 0)
  softmax     online single-pass max/normalizer (softmax.cu), BLOCK_DIM=64,
              N_READS=4, xor-butterfly all-reduces, identity-padded second
              level -- ported line for line below
  top-8       argpartition(kth=-8) tail order == argsort tail (ascending by
              value), including ties
  renorm      scores.sum(-1) on (1,1,8): row_reduce_simple, two four-term
              sequential partials then one add (the flat-(8,) all_reduce
              route is linear -- shape picks the kernel and the bits)
  aggregate   (y32 * s).sum(axis=-2): col_reduce_small's linear loop with the
              multiply rounded separately -- __fmul_rn then __fadd_rn, NO fma
              contraction (contraction is exactly what broke the obvious
              candidates)

Prints one JSON report. Every "x/y" is bitwise array_equal counts.

    python benchmarks/maple_exact_lane_semantics.py --model model-cuda
"""

import argparse
import json

import mlx.core as mx

GEMV_SRC = r"""
    constexpr int NPT = 4;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * (THREADS_ / 32) + warp;
    if (row >= NROWS_) return;

    float sum = 0.0f;
    for (int col = NPT * lane; col < K_; col += 32 * NPT) {
        const float4 m = *reinterpret_cast<const float4*>(mat + (long long)row * K_ + col);
        const float4 v = *reinterpret_cast<const float4*>(vec + col);
        if (FMA_ == 1) {
            sum = fmaf(m.x, v.x, sum);
            sum = fmaf(m.y, v.y, sum);
            sum = fmaf(m.z, v.z, sum);
            sum = fmaf(m.w, v.w, sum);
        } else {
            sum = sum + m.x * v.x;
            sum = sum + m.y * v.y;
            sum = sum + m.z * v.z;
            sum = sum + m.w * v.w;
        }
    }
    if (RED_ == 0) {
        // descending shfl_down tree
        for (int o = 16; o > 0; o >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, o);
    } else if (RED_ == 1) {
        // xor butterfly
        for (int o = 16; o > 0; o >>= 1)
            sum += __shfl_xor_sync(0xffffffffu, sum, o);
    } else {
        // ascending xor butterfly
        for (int o = 1; o < 32; o <<= 1)
            sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    if (lane == 0) out[row] = sum;
"""

SOFTMAX_SRC = r"""
    constexpr int BLOCK_DIM = 64;
    constexpr int N_READS = 4;
    constexpr int AX = 256;
    const int tid = threadIdx.x;
    if (tid >= BLOCK_DIM) return;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    constexpr int WARPS = BLOCK_DIM / 32;

    float vals[N_READS];
    #pragma unroll
    for (int i = 0; i < N_READS; ++i)
        vals[i] = x[tid * N_READS + i];

    float maxval = -INFINITY;
    #pragma unroll
    for (int i = 0; i < N_READS; ++i)
        maxval = fmaxf(maxval, vals[i]);
    float normalizer = 0.0f;
    #pragma unroll
    for (int i = 0; i < N_READS; ++i)
        normalizer = normalizer + __expf(vals[i] - maxval);

    // warp all-reduce, xor butterfly, exactly cg::reduce's shape
    float prevmax = maxval;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffffu, maxval, o));
    normalizer = normalizer * __expf(prevmax - maxval);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        normalizer = normalizer + __shfl_xor_sync(0xffffffffu, normalizer, o);

    __shared__ float local_max[32];
    __shared__ float local_norm[32];
    prevmax = maxval;
    if (lane == 0) local_max[warp] = maxval;
    __syncthreads();
    maxval = (lane < WARPS) ? local_max[lane] : -INFINITY;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffffu, maxval, o));
    normalizer = normalizer * __expf(prevmax - maxval);
    if (lane == 0) local_norm[warp] = normalizer;
    __syncthreads();
    normalizer = (lane < WARPS) ? local_norm[lane] : 0.0f;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        normalizer = normalizer + __shfl_xor_sync(0xffffffffu, normalizer, o);
    normalizer = 1.0f / normalizer;

    #pragma unroll
    for (int i = 0; i < N_READS; ++i)
        out[tid * N_READS + i] = __expf(vals[i] - maxval) * normalizer;
"""

SUM8_SRC = r"""
    // sum(axis=-1) over (1,1,8) dispatches to row_reduce_simple with
    // N_READS=4: two sequential four-term partials, then one add.  The
    // shape matters: a flat (8,) array takes all_reduce instead, whose
    // order is plain linear -- pinning the wrong route cost a day.
    if (threadIdx.x != 0) return;
    const float lo = __fadd_rn(__fadd_rn(__fadd_rn(x[0], x[1]), x[2]), x[3]);
    const float hi = __fadd_rn(__fadd_rn(__fadd_rn(x[4], x[5]), x[6]), x[7]);
    out[0] = __fadd_rn(lo, hi);
"""

AGG_SRC = r"""
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= NC_) return;
    float s = 0.0f;
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        const float p = __fmul_rn(__bfloat162float(y[e * NC_ + c]), sc[e]);
        s = __fadd_rn(s, p);
    }
    out[c] = __nv_bfloat16(s);
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=3)
    args = ap.parse_args()

    from mlx_lm import load
    model, _, _ = load(
        args.model, return_config=True,
        model_config={"model_file": None, "use_flash_head": False},
        tokenizer_config={"trust_remote_code": False},
        trust_remote_code=False)
    gate = model.model.layers[args.layer].mlp.gate
    w32 = gate.weight.astype(mx.float32)
    nrows, k = w32.shape
    mx.eval(w32)
    report = {}

    # --- logits ---
    kern = mx.fast.cuda_kernel(
        name="exact_router_gemv", input_names=["mat", "vec"],
        output_names=["out"], source=GEMV_SRC)
    hits = 0
    for i in range(16):
        mx.random.seed(11 + i)
        x = mx.random.normal((k,)).astype(mx.bfloat16).astype(mx.float32)
        mx.eval(x)
        ref = (x.reshape(1, k) @ w32.T).reshape(-1)
        (got,) = kern(
            inputs=[w32, x],
            template=[("K_", k), ("NROWS_", nrows), ("THREADS_", 256),
                      ("FMA_", 1), ("RED_", 0)],
            grid=((nrows // 8) * 256, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(nrows,)], output_dtypes=[mx.float32])
        mx.eval(got, ref)
        hits += int(mx.array_equal(got, ref).item())
    report["logits_fp32_gemv"] = f"{hits}/16"

    # --- softmax ---
    kern = mx.fast.cuda_kernel(
        name="exact_router_softmax", input_names=["x"],
        output_names=["out"], source=SOFTMAX_SRC)
    hits = 0
    for i in range(100):
        mx.random.seed(5000 + i)
        v = mx.random.normal((256,)).astype(mx.float32) * 4
        mx.eval(v)
        ref = mx.softmax(v)
        (got,) = kern(
            inputs=[v], template=[("D_", 0)],
            grid=(64, 1, 1), threadgroup=(64, 1, 1),
            output_shapes=[(256,)], output_dtypes=[mx.float32])
        mx.eval(got, ref)
        hits += int(mx.array_equal(got, ref).item())
    report["softmax_256_online_port"] = f"{hits}/100"

    # --- top-8 order ---
    agree = ties = 0
    for i in range(200):
        mx.random.seed(1000 + i)
        v = mx.random.normal((256,)).astype(mx.float32)
        sm = mx.softmax(v)
        a = mx.argpartition(sm, kth=-8, axis=-1)[..., -8:]
        b = mx.argsort(sm, axis=-1)[..., -8:]
        mx.eval(a, b)
        agree += int(mx.array_equal(a, b).item())
        v2 = mx.concatenate([v[:255], v[254:255]])
        sm2 = mx.softmax(v2)
        a2 = mx.argpartition(sm2, kth=-8, axis=-1)[..., -8:]
        b2 = mx.argsort(sm2, axis=-1)[..., -8:]
        mx.eval(a2, b2)
        ties += int(mx.array_equal(a2, b2).item())
    report["argpartition_tail_is_argsort_tail"] = f"{agree}/200"
    report["argpartition_tail_is_argsort_tail_ties"] = f"{ties}/200"

    # --- renorm sum ---
    kern = mx.fast.cuda_kernel(
        name="exact_renorm_sum", input_names=["x"], output_names=["out"],
        source=SUM8_SRC)
    hits = 0
    for i in range(64):
        mx.random.seed(3000 + i)
        v = mx.random.normal((8,)).astype(mx.float32)
        mx.eval(v)
        ref = v.reshape(1, 1, 8).sum(axis=-1)  # the router's actual shape
        (got,) = kern(
            inputs=[v], template=[("D_", 0)],
            grid=(32, 1, 1), threadgroup=(32, 1, 1),
            output_shapes=[(1,)], output_dtypes=[mx.float32])
        mx.eval(got, ref)
        hits += int(mx.array_equal(got, ref).item())
    report["renorm_sum8_linear"] = f"{hits}/64"

    # --- aggregation ---
    kern = mx.fast.cuda_kernel(
        name="exact_aggregate", input_names=["y", "sc"],
        output_names=["out"], source=AGG_SRC)
    nc = 2048
    hits = 0
    for i in range(64):
        mx.random.seed(4000 + i)
        y = mx.random.normal((8, nc)).astype(mx.bfloat16)
        sc = mx.softmax(mx.random.normal((8,)).astype(mx.float32))
        mx.eval(y, sc)
        ref = ((y.astype(mx.float32) * sc[:, None])
               .sum(axis=0).astype(mx.bfloat16))
        (got,) = kern(
            inputs=[y.reshape(-1), sc], template=[("NC_", nc)],
            grid=(nc, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(nc,)], output_dtypes=[mx.bfloat16])
        mx.eval(got, ref)
        hits += int(mx.array_equal(got, ref).item())
    report["aggregate_linear_uncontracted"] = f"{hits}/64"

    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
