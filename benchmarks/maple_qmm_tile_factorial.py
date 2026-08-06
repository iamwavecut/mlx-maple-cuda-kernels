#!/usr/bin/env python3
"""Within-process paired default-versus-candidate W2 CTA-tile benchmark."""

import argparse
import hashlib
import inspect
import json
import os
import statistics
from pathlib import Path

import mlx.core as mx
from maple_auto_benchmark import enable_fast, run
from mlx_lm import load
from mlx_lm.models import maple

TILE_KEYS = [
    "MLX_QMM_NAIVE_TILE_M",
    "MLX_QMM_NAIVE_TILE_N",
    "MLX_QMM_NAIVE_TILE_K",
]


def select(mode, tile):
    for key in TILE_KEYS:
        os.environ.pop(key, None)
    if mode == "candidate":
        for key, value in zip(TILE_KEYS, tile):
            os.environ[key] = str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile", type=int, nargs=3, metavar=("M", "N", "K"), required=True)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--generation-tokens", type=int, default=512)
    parser.add_argument("--blocks", type=int, default=10)
    args = parser.parse_args()
    if os.environ.get("MLX_USE_CUDA_GRAPHS") != "0":
        raise RuntimeError("paired runtime-tile switching requires MLX_USE_CUDA_GRAPHS=0")
    mx.random.seed(20260806)
    model, tokenizer, config = load(
        str(args.model),
        return_config=True,
        model_config={"model_file": None, "use_flash_head": False},
        tokenizer_config={"trust_remote_code": False},
        trust_remote_code=False,
    )
    source = Path(inspect.getfile(type(model))).resolve()
    if source != Path(maple.__file__).resolve():
        raise RuntimeError("checkpoint-local model source was loaded")
    tokenizer._eos_token_ids = {}
    vocab = config.get("vocab_size") or config["text_config"]["vocab_size"]
    prompt = mx.random.randint(0, vocab, (args.prompt_tokens,)).tolist()
    enable_fast(model)
    records = [
        {
            "type": "environment",
            "device": dict(mx.device_info(mx.gpu)),
            "mlx": mx.__version__,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "prompt_sha256": hashlib.sha256(
                ",".join(map(str, prompt)).encode()
            ).hexdigest(),
            "candidate_tile": args.tile,
        }
    ]
    expected = None
    for mode in ["default", "candidate"]:
        select(mode, args.tile)
        result = run(model, tokenizer, prompt, args.generation_tokens)
        if expected is None:
            expected = result["token_sha256"]
        if result["token_sha256"] != expected:
            raise RuntimeError(f"warm token mismatch for {mode}")
    fast_state = {
        "type": "fast_path_state",
        "add_rms_norm": model.model._fused_add_norm,
        "qk_norm": [layer.self_attn._fused_qk for layer in model.model.layers],
        "router": [layer.mlp.gate._fused for layer in model.model.layers],
        "cached_decode_lhs": maple._use_cached_decode_lhs,
        "router_indices_uint32": maple._cuda_router_indices_uint32,
        "ternary_up_gate": maple._use_cuda_ternary_up_gate,
        "approximate_router": maple._use_approximate_router,
        "approximate_add_rms": maple._use_approximate_add_rms,
    }
    if fast_state["add_rms_norm"] is None or any(
        value is None for value in fast_state["qk_norm"] + fast_state["router"]
    ):
        raise RuntimeError(f"a live strict probe did not resolve: {fast_state}")
    records.append(fast_state)
    for block in range(1, args.blocks + 1):
        order = ["default", "candidate"] if block % 2 else ["candidate", "default"]
        for position, mode in enumerate(order, 1):
            select(mode, args.tile)
            result = run(model, tokenizer, prompt, args.generation_tokens)
            if result["token_sha256"] != expected:
                raise RuntimeError(f"token mismatch in block {block} for {mode}")
            records.append(
                {
                    "type": "trial",
                    "block": block,
                    "position": position,
                    "mode": mode,
                    **{key: value for key, value in result.items() if key != "tokens"},
                }
            )
    for mode in ["default", "candidate"]:
        values = [
            item["generation_tps"]
            for item in records
            if item.get("type") == "trial" and item["mode"] == mode
        ]
        records.append(
            {
                "type": "summary",
                "mode": mode,
                "mean_generation_tps": statistics.fmean(values),
                "median_generation_tps": statistics.median(values),
            }
        )
    select("default", args.tile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in records)
    )
    print(json.dumps(records[-1], sort_keys=True))


if __name__ == "__main__":
    main()
