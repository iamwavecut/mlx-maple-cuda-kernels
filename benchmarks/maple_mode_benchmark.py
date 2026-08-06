#!/usr/bin/env python3
"""Isolated Maple decode-mode benchmark with source and token provenance."""

import argparse
import hashlib
import inspect
import json
import os
import statistics
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models import maple


def enable_fast(model):
    maple._use_cuda_ternary_up_gate = False
    maple._use_approximate_router = False
    maple._use_approximate_add_rms = False
    model.model._fused_add_norm = None
    for layer in model.model.layers:
        layer.self_attn._fused_qk = None
        layer.mlp.gate._fused = None


def configure(lhs, uint32, lhs_shape):
    maple._use_cached_decode_lhs = lhs
    maple._cuda_router_indices_uint32 = uint32
    maple._decode_lhs_cache.clear()
    if lhs and lhs_shape == "broadcast":
        maple._decode_lhs_cache[8] = (
            mx.zeros((1, 1, 8), dtype=mx.uint32),
            mx.arange(8, dtype=mx.uint32).reshape(1, 1, 8),
        )
        mx.eval(*maple._decode_lhs_cache[8])


def run(model, tokenizer, prompt, max_tokens):
    tokens = []
    response = None
    for response in stream_generate(
        model, tokenizer, prompt, max_tokens=max_tokens, prefill_step_size=2048
    ):
        tokens.append(int(response.token))
    if response is None:
        raise RuntimeError("generation produced no result")
    return {
        "tokens": tokens,
        "token_sha256": hashlib.sha256(
            ",".join(map(str, tokens)).encode()
        ).hexdigest(),
        "generation_tps": response.generation_tps,
        "prompt_tps": response.prompt_tps,
        "peak_memory": response.peak_memory,
    }


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--lhs", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--uint32", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lhs-shape", choices=["flat", "broadcast"], default="flat")
    p.add_argument("--prompt-tokens", type=int, default=128)
    p.add_argument("--generation-tokens", type=int, default=512)
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--trials", type=int, default=5)
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
    configure(args.lhs, args.uint32, args.lhs_shape)

    model_source = Path(inspect.getfile(type(model)))
    maple_source = Path(inspect.getfile(maple))
    records = [{
        "type": "environment", "tag": args.tag,
        "device": dict(mx.device_info(mx.gpu)), "mlx": mx.__version__,
        "env": {k: os.environ.get(k) for k in [
            "MLX_USE_CUDA_GRAPHS", "MLX_CUDA_GRAPH_CACHE_SIZE",
            "MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER",
        ]},
        "lhs": args.lhs, "uint32": args.uint32,
        "lhs_shape": args.lhs_shape,
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "model_source": str(model_source.resolve()),
        "model_source_sha256": file_sha(model_source),
        "maple_module_source": str(maple_source.resolve()),
        "maple_module_sha256": file_sha(maple_source),
        "config_model_file": config.get("model_file"),
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
    for trial in range(1, args.trials + 1):
        result = run(model, tokenizer, prompt, args.generation_tokens)
        expected = expected or result["token_sha256"]
        if result["token_sha256"] != expected:
            raise RuntimeError("nondeterministic token stream")
        records.append({
            "type": "trial", "trial": trial, "tag": args.tag,
            **{k: v for k, v in result.items() if k != "tokens"},
        })
    vals = [r["generation_tps"] for r in records if r.get("type") == "trial"]
    records.append({
        "type": "summary", "tag": args.tag,
        "mean_generation_tps": statistics.fmean(vals),
        "median_generation_tps": statistics.median(vals),
        "min_generation_tps": min(vals), "max_generation_tps": max(vals),
        "stdev_generation_tps": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "token_sha256": expected,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    print(json.dumps(records[-1], sort_keys=True))


if __name__ == "__main__":
    main()
