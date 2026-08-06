#!/usr/bin/env python3
"""Validate Maple's row-alpha ternary specialization across every MoE layer."""

import argparse
import hashlib
import json
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


def kernel(name, k, stage_x):
    return mx.fast.cuda_kernel(
        name=name,
        input_names=["x", "weights", "scales", "rhs_indices"],
        output_names=["out"],
        source=SOURCE,
        shared_memory=k * 2 if stage_x else 0,
    )


def custom(kern, x, layer, indices, threads, rows, stage_x, batched):
    k = layer.input_dims
    n = layer.output_dims
    blocks_x = (n + (threads // 32) * rows - 1) // ((threads // 32) * rows)
    return kern(
        inputs=[x.reshape(-1), layer.weight, layer.scales, indices.reshape(-1)],
        template=[
            ("T_", x.dtype), ("K", k), ("N", n), ("WORDS", k // 16),
            ("GROUPS", k // layer.group_size), ("ROWS_PER_WARP", rows),
            ("THREADS", threads), ("STAGE_X", stage_x),
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
        xin, layer.weight, layer.scales, layer.biases,
        lhs_indices=lhs, rhs_indices=rhs, transpose=True,
        group_size=layer.group_size, bits=layer.bits, mode=layer.mode,
    ).reshape(indices.size, layer.output_dims)


def schema(layer):
    row_alpha = mx.all(layer.scales == layer.scales[..., :1])
    affine_bias = mx.all(layer.biases == -layer.scales)
    code3_bits = (layer.weight & (layer.weight >> 1)) & mx.array(
        0x55555555, dtype=mx.uint32
    )
    no_code3 = mx.all(code3_bits == 0)
    mx.eval(row_alpha, affine_bias, no_code3)
    return {
        "row_alpha": bool(row_alpha),
        "bias_is_negative_scale": bool(affine_bias),
        "no_code_3": bool(no_code3),
    }


def sample_x(k, slots, sample, batched):
    shape = (slots, k) if batched else (k,)
    if sample == 0:
        return mx.random.normal(shape).astype(mx.bfloat16)
    if sample == 1:
        return (mx.random.normal(shape) * 0.01).astype(mx.bfloat16)
    if sample == 2:
        return (mx.random.normal(shape) * 10.0).astype(mx.bfloat16)
    if sample == 3:
        return mx.random.uniform(low=-7.0, high=7.0, shape=shape).astype(mx.bfloat16)
    if sample == 4:
        return mx.zeros(shape, dtype=mx.bfloat16)
    base = mx.arange(k, dtype=mx.float32)
    base = mx.where((base.astype(mx.uint32) & 1) == 0, 3.5, -3.5)
    return (mx.broadcast_to(base, shape) if batched else base).astype(mx.bfloat16)


def digest(a):
    return hashlib.sha256(bytes(a.astype(mx.bfloat16))).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=6)
    args = p.parse_args()

    mx.random.seed(20260806)
    model = load(
        str(args.model), tokenizer_config={"trust_remote_code": True},
        trust_remote_code=True,
    )[0]
    up_kernel = kernel("maple_validate_ternary_up", 2048, True)
    down_kernel = kernel("maple_validate_ternary_down", 512, True)
    records = [{"type": "environment", "device": dict(mx.device_info(mx.gpu))}]
    total_different = 0
    schema_failures = 0

    for layer_idx, decoder in enumerate(model.model.layers):
        moe = decoder.mlp.switch_mlp
        for label, layer, batched, kern, threads, rows, stage in [
            ("up_gate", moe.up_gate_proj, False, up_kernel, 512, 4, True),
            ("down", moe.down_proj, True, down_kernel, 512, 8, True),
        ]:
            inv = schema(layer)
            schema_failures += 0 if all(inv.values()) else 1
            records.append({"type": "schema", "layer": layer_idx, "case": label, **inv})
            for sample in range(args.samples):
                values = [(sample * 37 + layer_idx * 11 + j * 29) % 256 for j in range(8)]
                indices = mx.array(values, dtype=mx.uint32)
                x = sample_x(layer.input_dims, 8, sample, batched)
                ref = reference(x, layer, indices, batched)
                got = custom(kern, x, layer, indices, threads, rows, stage, batched)
                mx.eval(ref, got)
                diff = mx.abs(ref.astype(mx.float32) - got.astype(mx.float32))
                different = int(mx.sum(ref != got).item())
                total_different += different
                rec = {
                    "type": "comparison", "layer": layer_idx, "case": label,
                    "sample": sample, "indices": values,
                    "array_equal": different == 0, "different": different,
                    "elements": ref.size, "max_abs": float(mx.max(diff).item()),
                    "mean_abs": float(mx.mean(diff).item()),
                    "reference_sha256": digest(ref), "candidate_sha256": digest(got),
                }
                print(json.dumps(rec, sort_keys=True), flush=True)
                records.append(rec)

    summary = {
        "type": "summary", "schema_failures": schema_failures,
        "total_different": total_different,
        "all_array_equal": schema_failures == 0 and total_different == 0,
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    records.append(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    if not summary["all_array_equal"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
