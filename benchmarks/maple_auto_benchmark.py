#!/usr/bin/env python3
"""Warm single-mode Maple decode benchmark with exact token hashes."""

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models import maple


def enable_fast(model):
    maple._use_cached_decode_lhs = True
    maple._cuda_router_indices_uint32 = False
    maple._use_cuda_ternary_up_gate = False
    maple._use_approximate_router = False
    maple._use_approximate_add_rms = False
    model.model._fused_add_norm = None
    for layer in model.model.layers:
        layer.self_attn._fused_qk = None
        layer.mlp.gate._fused = None


def run(model, tokenizer, prompt, max_tokens):
    tokens = []
    response = None
    tic = time.perf_counter()
    for response in stream_generate(
        model, tokenizer, prompt, max_tokens=max_tokens, prefill_step_size=2048
    ):
        tokens.append(int(response.token))
    elapsed = time.perf_counter() - tic
    if response is None:
        raise RuntimeError("no generation result")
    return {
        "tokens": tokens,
        "token_sha256": hashlib.sha256(
            ",".join(map(str, tokens)).encode()
        ).hexdigest(),
        "generation_tps": response.generation_tps,
        "prompt_tps": response.prompt_tps,
        "elapsed": elapsed,
        "peak_memory": response.peak_memory,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prompt-tokens", type=int, default=128)
    p.add_argument("--generation-tokens", type=int, default=256)
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--trials", type=int, default=3)
    args = p.parse_args()

    mx.random.seed(20260806)
    model, tokenizer, config = load(
        str(args.model), return_config=True,
        model_config={"model_file": None, "use_flash_head": False},
        tokenizer_config={"trust_remote_code": True}, trust_remote_code=True,
    )
    tokenizer._eos_token_ids = {}
    vocab = config.get("vocab_size") or config["text_config"]["vocab_size"]
    prompt = mx.random.randint(0, vocab, (args.prompt_tokens,)).tolist()
    enable_fast(model)
    records = [{
        "type": "environment",
        "device": dict(mx.device_info(mx.gpu)),
        "mlx": mx.__version__,
        "env": {k: os.environ.get(k) for k in [
            "MLX_USE_CUDA_GRAPHS", "MLX_CUDA_GRAPH_CACHE_SIZE",
            "MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER",
            "MLX_CUDA_USE_CUDNN_SDPA", "MLX_ENABLE_TF32",
        ]},
        "prompt_sha256": hashlib.sha256(
            ",".join(map(str, prompt)).encode()
        ).hexdigest(),
    }]
    for _ in range(args.warmups):
        run(model, tokenizer, prompt, args.generation_tokens)
    fast_state = {
        "add_rms_norm": model.model._fused_add_norm,
        "qk_norm": [layer.self_attn._fused_qk for layer in model.model.layers],
        "router": [layer.mlp.gate._fused for layer in model.model.layers],
    }
    if fast_state["add_rms_norm"] is None or any(
        value is None for value in fast_state["qk_norm"] + fast_state["router"]
    ):
        raise RuntimeError(f"an auto live probe did not resolve: {fast_state}")
    records.append({"type": "fast_path_state", **fast_state})
    expected = None
    for trial in range(args.trials):
        result = run(model, tokenizer, prompt, args.generation_tokens)
        if expected is None:
            expected = result["token_sha256"]
        if result["token_sha256"] != expected:
            raise RuntimeError("nondeterministic token stream")
        records.append({
            "type": "trial", "trial": trial + 1,
            **{k: v for k, v in result.items() if k != "tokens"},
        })
    values = [r["generation_tps"] for r in records if r.get("type") == "trial"]
    records.append({
        "type": "summary",
        "mean_generation_tps": statistics.fmean(values),
        "median_generation_tps": statistics.median(values),
        "min_generation_tps": min(values),
        "max_generation_tps": max(values),
        "token_sha256": expected,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    print(json.dumps(records[-1], sort_keys=True))


if __name__ == "__main__":
    main()
