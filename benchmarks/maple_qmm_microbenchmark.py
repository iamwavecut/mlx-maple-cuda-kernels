#!/usr/bin/env python3
"""Real-weight gathered affine-W2 projection microbenchmark for Maple."""

import argparse
import hashlib
import inspect
import json
import os
import statistics
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import maple

from maple_qmm_fingerprint import fixed_input, project


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()
    model, _tokenizer = load(
        str(args.model),
        model_config={"model_file": None, "use_flash_head": False},
        tokenizer_config={"trust_remote_code": False},
        trust_remote_code=False,
    )
    source = Path(inspect.getfile(type(model))).resolve()
    if source != Path(maple.__file__).resolve():
        raise RuntimeError("checkpoint-local model source was loaded")

    layers = []
    for layer_idx, decoder in enumerate(model.model.layers):
        moe = decoder.mlp.switch_mlp
        ids = [(layer_idx * 37 + slot * 29) % moe.up_gate_proj.num_experts for slot in range(8)]
        indices = mx.array(ids, dtype=mx.int32)
        up_x = fixed_input(moe.up_gate_proj.input_dims, 8, layer_idx, 1, False)
        down_x = fixed_input(moe.down_proj.input_dims, 8, layer_idx, 1, True)
        layers.append((moe.up_gate_proj, up_x, moe.down_proj, down_x, indices))
    mx.eval(*[value for entry in layers for value in (entry[1], entry[3], entry[4])])

    def execute(mode):
        outputs = []
        for _ in range(args.repeats):
            for up_layer, up_x, down_layer, down_x, indices in layers:
                if mode == "combined":
                    up_gate = project(up_x, up_layer, indices, False)
                    up, gate = mx.split(up_gate, 2, axis=-1)
                    activated = maple.clamped_swiglu(gate, up)
                    outputs.append(project(activated, down_layer, indices, True))
                elif mode == "up":
                    outputs.append(project(up_x, up_layer, indices, False))
                else:
                    outputs.append(project(down_x, down_layer, indices, True))
        mx.eval(*outputs)
        mx.synchronize()

    records = [
        {
            "type": "environment",
            "device": dict(mx.device_info(mx.gpu)),
            "mlx": mx.__version__,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "tile": {
                key: os.environ.get(key)
                for key in [
                    "MLX_QMM_NAIVE_TILE_M",
                    "MLX_QMM_NAIVE_TILE_N",
                    "MLX_QMM_NAIVE_TILE_K",
                ]
            },
            "layers": len(layers),
            "repeats": args.repeats,
        }
    ]
    for mode in ["up", "down", "combined"]:
        for _ in range(args.warmups):
            execute(mode)
        for trial in range(1, args.trials + 1):
            started = time.perf_counter()
            execute(mode)
            elapsed = time.perf_counter() - started
            records.append(
                {
                    "type": "trial",
                    "mode": mode,
                    "trial": trial,
                    "elapsed": elapsed,
                    "milliseconds_per_layer": elapsed * 1000.0 / (args.repeats * len(layers)),
                }
            )
    for mode in ["up", "down", "combined"]:
        values = [
            record["milliseconds_per_layer"]
            for record in records
            if record.get("type") == "trial" and record["mode"] == mode
        ]
        records.append(
            {
                "type": "summary",
                "mode": mode,
                "mean_milliseconds_per_layer": statistics.fmean(values),
                "median_milliseconds_per_layer": statistics.median(values),
                "min_milliseconds_per_layer": min(values),
                "max_milliseconds_per_layer": max(values),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    print(json.dumps(records[-1], sort_keys=True))


if __name__ == "__main__":
    main()
