# Copyright © 2026 DeepGrove AI.

"""Measure every combination of Maple decode fast paths in one model load."""

import argparse
import json
import statistics
from itertools import product
from pathlib import Path

import mlx.core as mx
from maple_kernel_benchmark import _apply_router_override
from maple_model_benchmark import _environment, _run

from mlx_lm import load
from mlx_lm.models import maple


def _set_paths(model, add_rms_norm, qk_norm, router):
    # add-RMS and router=True are explicit semantic/experimental modes.
    maple._use_approximate_add_rms = add_rms_norm
    maple._use_approximate_router = router
    model.model._fused_add_norm = add_rms_norm
    for layer in model.model.layers:
        layer.self_attn._fused_qk = qk_norm
        layer.mlp.gate._fused = router


def _mode_name(add_rms_norm, qk_norm, router):
    enabled = [
        name
        for name, value in (
            ("add", add_rms_norm),
            ("qk", qk_norm),
            ("router", router),
        )
        if value
    ]
    return "reference" if not enabled else "+".join(enabled)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--generation-tokens", type=int, default=256)
    parser.add_argument("--equivalence-tokens", type=int, default=256)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--quality-only", action="store_true")
    parser.add_argument("--router-threads", type=int)
    parser.add_argument("--router-rows-per-warp", type=int)
    args = parser.parse_args()

    try:
        _apply_router_override(args.router_threads, args.router_rows_per_warp)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))

    mx.random.seed(20260806)
    model, tokenizer, config = load(
        str(args.model),
        return_config=True,
        tokenizer_config={"trust_remote_code": True},
        trust_remote_code=True,
    )
    tokenizer._eos_token_ids = {}
    vocab_size = config.get("vocab_size") or config["text_config"]["vocab_size"]
    prompt = mx.random.randint(0, vocab_size, (args.prompt_tokens,)).tolist()
    modes = [
        (add_rms_norm, qk_norm, router)
        for add_rms_norm, qk_norm, router in product((False, True), repeat=3)
    ]

    _set_paths(model, False, False, False)
    reference = _run(model, tokenizer, prompt, args.equivalence_tokens)
    records = [_environment(args.model, config)]
    for mode in modes:
        _set_paths(model, *mode)
        result = _run(model, tokenizer, prompt, args.equivalence_tokens)
        records.append(
            {
                "type": "equivalence",
                "mode": _mode_name(*mode),
                "add_rms_norm": mode[0],
                "qk_norm": mode[1],
                "router": mode[2],
                "tokens": args.equivalence_tokens,
                "tokens_equal": result["tokens"] == reference["tokens"],
                "reference_sha256": reference["token_sha256"],
                "token_sha256": result["token_sha256"],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    mismatches = [
        record
        for record in records
        if record["type"] == "equivalence" and not record["tokens_equal"]
    ]
    if mismatches:
        raise RuntimeError(
            "generation mismatch in " + ", ".join(r["mode"] for r in mismatches)
        )
    if args.quality_only:
        print("all fast-path combinations preserve the reference tokens")
        return

    for mode in modes:
        _set_paths(model, *mode)
        _run(model, tokenizer, prompt, args.generation_tokens)
        for trial in range(1, args.trials + 1):
            result = _run(model, tokenizer, prompt, args.generation_tokens)
            records.append(
                {
                    "type": "trial",
                    "mode": _mode_name(*mode),
                    "add_rms_norm": mode[0],
                    "qk_norm": mode[1],
                    "router": mode[2],
                    "trial": trial,
                    "prompt_tokens": args.prompt_tokens,
                    "generation_tokens": args.generation_tokens,
                    **{key: value for key, value in result.items() if key != "tokens"},
                }
            )

    for mode in modes:
        name = _mode_name(*mode)
        values = [
            record["generation_tps"]
            for record in records
            if record.get("type") == "trial" and record["mode"] == name
        ]
        records.append(
            {
                "type": "summary",
                "mode": name,
                "add_rms_norm": mode[0],
                "qk_norm": mode[1],
                "router": mode[2],
                "prompt_tokens": args.prompt_tokens,
                "generation_tokens": args.generation_tokens,
                "mean_generation_tps": statistics.fmean(values),
                "median_generation_tps": statistics.median(values),
                "min_generation_tps": min(values),
                "max_generation_tps": max(values),
            }
        )

    args.output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    for record in records:
        if record["type"] == "summary":
            print(record["mode"], f"{record['mean_generation_tps']:.3f} tok/s")


if __name__ == "__main__":
    main()
