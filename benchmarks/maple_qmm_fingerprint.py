#!/usr/bin/env python3
"""Cross-process array-exact gate for Maple's gathered affine-W2 projections."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import maple


def digest(x):
    mx.eval(x)
    return hashlib.sha256(bytes(x.view(mx.uint16))).hexdigest()


def backend_hashes():
    distribution = importlib.metadata.distribution("mlx-cuda-12")
    hashes = {}
    for entry in distribution.files or []:
        path = Path(distribution.locate_file(entry))
        if path.is_file() and (path.suffix == ".so" or ".so." in path.name):
            hashes[str(entry)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def fixed_input(k, slots, layer_idx, sample, batched):
    size = slots * k if batched else k
    base = mx.arange(size, dtype=mx.int32)
    values = ((base * 29 + layer_idx * 17 + sample * 43) % 257 - 128).astype(
        mx.float32
    )
    values = values / (1.0 if sample == 2 else (8.0 if sample == 1 else 64.0))
    shape = (slots, k) if batched else (k,)
    return values.astype(mx.bfloat16).reshape(shape)


def project(x, layer, indices, batched):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()

    maple._use_cached_decode_lhs = False
    maple._cuda_router_indices_uint32 = False
    maple._use_cuda_ternary_up_gate = False
    maple._use_approximate_router = False
    maple._use_approximate_add_rms = False
    model, _tokenizer = load(
        str(args.model),
        model_config={"model_file": None, "use_flash_head": False},
        tokenizer_config={"trust_remote_code": False},
        trust_remote_code=False,
    )
    if Path(inspect.getfile(type(model))).resolve() != Path(maple.__file__).resolve():
        raise RuntimeError("checkpoint-local model source was loaded")

    reference = mx.load(str(args.reference)) if args.reference else None
    tensors = {}
    records = [
        {
            "type": "environment",
            "device": dict(mx.device_info(mx.gpu)),
            "maple_sha256": hashlib.sha256(Path(maple.__file__).read_bytes()).hexdigest(),
            "mlx": mx.__version__,
            "mlx_cuda_12": importlib.metadata.version("mlx-cuda-12"),
            "backend_library_sha256": backend_hashes(),
            "tile": {
                key: os.environ.get(key)
                for key in [
                    "MLX_QMM_NAIVE_TILE_M",
                    "MLX_QMM_NAIVE_TILE_N",
                    "MLX_QMM_NAIVE_TILE_K",
                ]
            },
            "reference_sha256": (
                hashlib.sha256(args.reference.read_bytes()).hexdigest()
                if args.reference
                else None
            ),
        }
    ]
    total_different = 0
    missing = []
    schema_failures = 0
    all_equal = True
    for layer_idx, decoder in enumerate(model.model.layers):
        moe = decoder.mlp.switch_mlp
        for label, layer, batched in [
            ("up_gate", moe.up_gate_proj, False),
            ("down", moe.down_proj, True),
        ]:
            for sample in range(args.samples):
                ids = [
                    (layer_idx * 37 + sample * 53 + slot * 29) % layer.num_experts
                    for slot in range(8)
                ]
                indices = mx.array(ids, dtype=mx.int32)
                x = fixed_input(
                    layer.input_dims, indices.size, layer_idx, sample, batched
                )
                got = project(x, layer, indices, batched)
                mx.eval(got)
                key = f"layer_{layer_idx:02d}_{label}_sample_{sample}"
                tensors[key] = got
                rec = {
                    "type": "projection",
                    "key": key,
                    "layer": layer_idx,
                    "projection": label,
                    "sample": sample,
                    "shape": list(got.shape),
                    "dtype": str(got.dtype),
                    "sha256": digest(got),
                }
                if reference is not None:
                    if key not in reference:
                        missing.append(key)
                        all_equal = False
                        schema_failures += 1
                        rec.update(array_equal=False, missing_reference=True)
                    else:
                        want = reference[key]
                        shape_equal = got.shape == want.shape
                        dtype_equal = got.dtype == want.dtype
                        same = shape_equal and dtype_equal and bool(mx.array_equal(got, want))
                        all_equal = all_equal and same
                        if not shape_equal or not dtype_equal:
                            schema_failures += 1
                        if shape_equal:
                            diff = mx.abs(
                                got.astype(mx.float32) - want.astype(mx.float32)
                            )
                            different = int(mx.sum(got != want).item())
                            max_abs = float(mx.max(diff).item())
                            total_different += different
                        else:
                            different = None
                            max_abs = None
                        rec.update(
                            array_equal=same,
                            shape_equal=shape_equal,
                            dtype_equal=dtype_equal,
                            different=different,
                            elements=got.size,
                            max_abs=max_abs,
                            reference_sha256=digest(want),
                        )
                records.append(rec)
                print(json.dumps(rec, sort_keys=True), flush=True)

    args.save.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.save), tensors, metadata={"format": "maple-w2-v1"})
    extra_reference = sorted(set(reference or {}) - set(tensors))
    if extra_reference:
        all_equal = False
        schema_failures += len(extra_reference)
    summary = {
        "type": "summary",
        "projections": len(tensors),
        "reference_projections": len(reference) if reference is not None else None,
        "missing": missing,
        "extra_reference": extra_reference,
        "schema_failures": schema_failures,
        "total_different": total_different,
        "all_array_equal": reference is None or all_equal,
    }
    records.append(summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    if not summary["all_array_equal"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
