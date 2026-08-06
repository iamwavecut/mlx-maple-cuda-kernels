#!/usr/bin/env python3
"""Prototype Maple row-alpha ternary top-8 GEMV and compare with GatherQMM."""

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load


SOURCE = r"""
    constexpr int WARP = 32;
    constexpr int WARPS = THREADS / WARP;
    constexpr int ROWS_PER_BLOCK = WARPS * ROWS_PER_WARP;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int slot = blockIdx.y;
    int row0 = blockIdx.x * ROWS_PER_BLOCK + warp * ROWS_PER_WARP;
    int expert = static_cast<int>(rhs_indices[slot]);
    int x_base = X_BATCHED ? slot * K : 0;

    const T_* xv_ptr = x + x_base;
    if constexpr (STAGE_X) {
        extern __shared__ unsigned char raw_smem[];
        T_* staged_x = reinterpret_cast<T_*>(raw_smem);
        for (int col = tid; col < K; col += THREADS) {
            staged_x[col] = x[x_base + col];
        }
        __syncthreads();
        xv_ptr = staged_x;
    }
    if (row0 >= N) return;

    float sums[ROWS_PER_WARP] = {0.0f};
    float alpha[ROWS_PER_WARP] = {0.0f};
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        int row = row0 + r;
        if (row < N) {
            long long scale_off =
                (static_cast<long long>(expert) * N + row) * GROUPS;
            alpha[r] = static_cast<float>(scales[scale_off]);
        }
    }

    for (int col = lane; col < K; col += WARP) {
        float xv = static_cast<float>(xv_ptr[col]);
        int word = col >> 4;
        int shift = (col & 15) << 1;
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            int row = row0 + r;
            if (row < N) {
                long long weight_off =
                    (static_cast<long long>(expert) * N + row) * WORDS + word;
                unsigned int code = (weights[weight_off] >> shift) & 3u;
                float weight = (static_cast<int>(code) - 1) * alpha[r];
                sums[r] = fmaf(xv, weight, sums[r]);
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sums[r] += __shfl_down_sync(0xffffffff, sums[r], offset);
        }
        if (lane == 0 && row0 + r < N) {
            out[slot * N + row0 + r] = static_cast<T_>(sums[r]);
        }
    }
"""


def make_kernel(name, k, n, rows_per_warp, threads, stage_x):
    return mx.fast.cuda_kernel(
        name=name,
        input_names=["x", "weights", "scales", "rhs_indices"],
        output_names=["out"],
        source=SOURCE,
        shared_memory=(k * 2 if stage_x else 0),
    )


def call_kernel(kernel, x, layer, indices, rows_per_warp, threads, stage_x, batched):
    k = layer.input_dims
    n = layer.output_dims
    warps = threads // 32
    blocks_x = (n + warps * rows_per_warp - 1) // (warps * rows_per_warp)
    return kernel(
        inputs=[x.reshape(-1), layer.weight, layer.scales, indices.reshape(-1)],
        template=[
            ("T_", x.dtype),
            ("K", k),
            ("N", n),
            ("WORDS", k // 16),
            ("GROUPS", k // layer.group_size),
            ("ROWS_PER_WARP", rows_per_warp),
            ("THREADS", threads),
            ("STAGE_X", stage_x),
            ("X_BATCHED", batched),
        ],
        grid=(blocks_x * threads, indices.size, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(indices.size, n)],
        output_dtypes=[x.dtype],
    )[0]


def reference(x, layer, indices, batched):
    rhs = indices.reshape(1, 1, -1)
    if batched:
        xin = x.reshape(1, 1, indices.size, 1, layer.input_dims)
        lhs = mx.arange(indices.size, dtype=mx.uint32)
    else:
        xin = x.reshape(1, 1, 1, 1, layer.input_dims)
        lhs = mx.zeros((indices.size,), dtype=mx.uint32)
    return mx.gather_qmm(
        xin,
        layer.weight,
        layer.scales,
        layer.biases,
        lhs_indices=lhs,
        rhs_indices=rhs,
        transpose=True,
        group_size=layer.group_size,
        bits=layer.bits,
        mode=layer.mode,
    ).reshape(indices.size, layer.output_dims)


def measure(fn, warmup, trials):
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    vals = []
    for _ in range(trials):
        tic = time.perf_counter_ns()
        mx.eval(fn())
        mx.synchronize()
        vals.append((time.perf_counter_ns() - tic) / 1000)
    return {
        "median_us": statistics.median(vals),
        "mean_us": statistics.fmean(vals),
        "p10_us": sorted(vals)[max(0, int(0.1 * len(vals)) - 1)],
        "p90_us": sorted(vals)[min(len(vals) - 1, int(0.9 * len(vals)))],
    }


def compare(got, ref):
    mx.eval(got, ref)
    diff = mx.abs(got.astype(mx.float32) - ref.astype(mx.float32))
    return {
        "array_equal": bool(mx.array_equal(got, ref)),
        "allclose_2e2": bool(mx.allclose(got, ref, rtol=2e-2, atol=2e-2)),
        "max_abs": float(mx.max(diff).item()),
        "mean_abs": float(mx.mean(diff).item()),
        "different": int(mx.sum(got != ref).item()),
        "elements": got.size,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--trials", type=int, default=100)
    args = p.parse_args()

    mx.random.seed(20260806)
    model = load(str(args.model), tokenizer_config={"trust_remote_code": True}, trust_remote_code=True)[0]
    moe = model.model.layers[0].mlp.switch_mlp
    indices = mx.array([3, 17, 42, 71, 109, 155, 201, 250], dtype=mx.uint32)
    x_up = mx.random.normal((moe.up_gate_proj.input_dims,)).astype(mx.bfloat16)
    x_down = mx.random.normal((8, moe.down_proj.input_dims)).astype(mx.bfloat16)
    mx.eval(indices, x_up, x_down)

    records = [{"type": "environment", "device": dict(mx.device_info(mx.gpu))}]
    for label, layer, x, batched in [
        ("up_gate", moe.up_gate_proj, x_up, False),
        ("down", moe.down_proj, x_down, True),
    ]:
        ref_fn = lambda x=x, layer=layer, batched=batched: reference(x, layer, indices, batched)
        ref = ref_fn()
        mx.eval(ref)
        ref_t = measure(ref_fn, args.warmup, args.trials)
        records.append({"type": "reference", "case": label, **ref_t})
        for threads in (128, 256, 512):
            for rows in (2, 4, 8, 16):
                for stage in (False, True):
                    name = f"maple_ternary_{label}_t{threads}_r{rows}_s{int(stage)}"
                    kernel = make_kernel(name, layer.input_dims, layer.output_dims, rows, threads, stage)
                    fn = lambda kernel=kernel, x=x, layer=layer, rows=rows, threads=threads, stage=stage, batched=batched: call_kernel(
                        kernel, x, layer, indices, rows, threads, stage, batched
                    )
                    try:
                        got = fn()
                        quality = compare(got, ref)
                        timing = measure(fn, args.warmup, args.trials)
                        rec = {
                            "type": "candidate",
                            "case": label,
                            "threads": threads,
                            "rows_per_warp": rows,
                            "stage_x": stage,
                            **quality,
                            **timing,
                            "speedup": ref_t["median_us"] / timing["median_us"],
                        }
                    except Exception as exc:
                        rec = {
                            "type": "candidate", "case": label,
                            "threads": threads, "rows_per_warp": rows,
                            "stage_x": stage, "error": f"{type(exc).__name__}: {exc}"
                        }
                    print(json.dumps(rec, sort_keys=True), flush=True)
                    records.append(rec)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))


if __name__ == "__main__":
    main()
