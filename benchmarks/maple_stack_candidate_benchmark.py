#!/usr/bin/env python3
"""Experimental factorial for approximate router/add-RMS decode cleanup."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import mlx.core as mx

from mlx_lm import load, stream_generate
from mlx_lm.models import maple


def set_fast_paths(model):
    # Explicit semantic lane: these two kernels are not strict-array-exact.
    maple._use_approximate_router = True
    maple._use_approximate_add_rms = True
    model.model._fused_add_norm = True
    for layer in model.model.layers:
        layer.self_attn._fused_qk = True
        layer.mlp.gate._fused = True


def set_mode(use_lhs, use_uint32):
    maple._use_cached_decode_lhs = use_lhs
    maple._cuda_router_indices_uint32 = use_uint32


def run(model, tokenizer, prompt, max_tokens):
    tokens = []
    response = None
    for response in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=max_tokens,
        prefill_step_size=2048,
    ):
        tokens.append(int(response.token))
    if response is None:
        raise RuntimeError("generation produced no response")
    return {
        "tokens": tokens,
        "token_sha256": hashlib.sha256(
            ",".join(map(str, tokens)).encode()
        ).hexdigest(),
        "generation_tps": response.generation_tps,
        "peak_memory": response.peak_memory,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--generation-tokens", type=int, default=256)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()

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
    set_fast_paths(model)

    modes = [
        ("baseline", False, False),
        ("lhs_only", True, False),
        ("uint32_only", False, True),
        ("all", True, True),
    ]
    records = [{
        "type": "environment",
        "device": dict(mx.device_info(mx.gpu)),
        "mlx_version": mx.__version__,
        "prompt_sha256": hashlib.sha256(
            ",".join(map(str, prompt)).encode()
        ).hexdigest(),
    }]

    # Compile, probe and populate graph caches for each graph variant.
    for name, lhs, uint32 in modes:
        set_mode(lhs, uint32)
        run(model, tokenizer, prompt, args.generation_tokens)

    reference_tokens = None
    for name, lhs, uint32 in modes:
        set_mode(lhs, uint32)
        result = run(model, tokenizer, prompt, args.generation_tokens)
        if reference_tokens is None:
            reference_tokens = result["tokens"]
        records.append({
            "type": "equivalence",
            "mode": name,
            "tokens_equal": result["tokens"] == reference_tokens,
            "token_sha256": result["token_sha256"],
            "tokens": len(result["tokens"]),
        })

    for trial in range(args.trials):
        order = modes if trial % 2 == 0 else list(reversed(modes))
        for name, lhs, uint32 in order:
            set_mode(lhs, uint32)
            result = run(model, tokenizer, prompt, args.generation_tokens)
            records.append({
                "type": "trial",
                "trial": trial + 1,
                "mode": name,
                "generation_tps": result["generation_tps"],
                "token_sha256": result["token_sha256"],
                "peak_memory": result["peak_memory"],
            })

    baseline = None
    for name, _, _ in modes:
        vals = [r["generation_tps"] for r in records
                if r.get("type") == "trial" and r["mode"] == name]
        summary = {
            "type": "summary",
            "mode": name,
            "mean_generation_tps": statistics.fmean(vals),
            "median_generation_tps": statistics.median(vals),
            "min_generation_tps": min(vals),
            "max_generation_tps": max(vals),
        }
        if name == "baseline":
            baseline = summary["mean_generation_tps"]
        summary["gain_vs_baseline_pct"] = (
            100 * (summary["mean_generation_tps"] / baseline - 1)
            if baseline is not None else 0.0
        )
        records.append(summary)

    if not all(r["tokens_equal"] for r in records if r.get("type") == "equivalence"):
        raise RuntimeError("candidate token stream differs from baseline")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    for r in records:
        if r.get("type") == "summary":
            print(r)


if __name__ == "__main__":
    main()
