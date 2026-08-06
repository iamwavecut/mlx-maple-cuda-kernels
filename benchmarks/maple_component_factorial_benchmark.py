#!/usr/bin/env python3
"""Within-process 2x2 exact-component benchmark: Q/K fusion x cached LHS."""

import argparse
import hashlib
import inspect
import json
import statistics
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models import maple

MODES = {
    "R": (False, False),
    "L": (False, True),
    "Q": (True, False),
    "QL": (True, True),
}
ORDERS = [
    ("R", "L", "Q", "QL"),
    ("QL", "Q", "L", "R"),
    ("L", "QL", "R", "Q"),
    ("Q", "R", "QL", "L"),
]


def configure(model, mode):
    qk, lhs = MODES[mode]
    maple._use_cached_decode_lhs = lhs
    maple._cuda_router_indices_uint32 = False
    maple._use_cuda_ternary_up_gate = False
    maple._use_approximate_router = False
    maple._use_approximate_add_rms = False
    model.model._fused_add_norm = False
    for layer in model.model.layers:
        layer.self_attn._fused_qk = qk
        layer.mlp.gate._fused = False


def run(model, tokenizer, prompt, max_tokens):
    mx.reset_peak_memory()
    tokens = []
    response = None
    for response in stream_generate(
        model, tokenizer, prompt, max_tokens=max_tokens, prefill_step_size=2048
    ):
        tokens.append(int(response.token))
    if response is None:
        raise RuntimeError("no generation response")
    return {
        "token_sha256": hashlib.sha256(",".join(map(str, tokens)).encode()).hexdigest(),
        "generated_tokens": len(tokens),
        "generation_tps": response.generation_tps,
        "prompt_tps": response.prompt_tps,
        "peak_memory": response.peak_memory,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prompt-tokens", type=int, default=128)
    p.add_argument("--generation-tokens", type=int, default=512)
    p.add_argument("--blocks", type=int, default=8)
    args = p.parse_args()
    mx.random.seed(20260806)
    model, tokenizer, config = load(
        str(args.model),
        return_config=True,
        model_config={"model_file": None, "use_flash_head": False},
        tokenizer_config={"trust_remote_code": False},
        trust_remote_code=False,
    )
    tokenizer._eos_token_ids = {}
    vocab = config.get("vocab_size") or config["text_config"]["vocab_size"]
    prompt = mx.random.randint(0, vocab, (args.prompt_tokens,)).tolist()
    source = Path(inspect.getfile(type(model))).resolve()
    module_source = Path(maple.__file__).resolve()
    if source != module_source:
        raise RuntimeError(f"loaded model source {source} differs from {module_source}")
    records = [
        {
            "type": "environment",
            "device": dict(mx.device_info(mx.gpu)),
            "model_source": str(source),
            "model_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "module_source_sha256": hashlib.sha256(
                module_source.read_bytes()
            ).hexdigest(),
            "config_model_file": config.get("model_file"),
            "prompt_sha256": hashlib.sha256(
                ",".join(map(str, prompt)).encode()
            ).hexdigest(),
        }
    ]
    expected = None
    for mode in MODES:
        configure(model, mode)
        result = run(model, tokenizer, prompt, args.generation_tokens)
        expected = result["token_sha256"] if expected is None else expected
        if result["token_sha256"] != expected:
            raise RuntimeError(f"warmup token mismatch in {mode}")
    for block in range(1, args.blocks + 1):
        order = ORDERS[(block - 1) % len(ORDERS)]
        if (block - 1) // len(ORDERS) % 2:
            order = tuple(reversed(order))
        for position, mode in enumerate(order, 1):
            configure(model, mode)
            result = run(model, tokenizer, prompt, args.generation_tokens)
            if result["token_sha256"] != expected:
                raise RuntimeError(f"token mismatch in block {block} mode {mode}")
            qk, lhs = MODES[mode]
            record = {
                "type": "trial",
                "block": block,
                "position": position,
                "mode": mode,
                "qk": qk,
                "lhs": lhs,
                **result,
            }
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
    for mode in MODES:
        vals = [r["generation_tps"] for r in records if r.get("mode") == mode]
        records.append(
            {
                "type": "summary",
                "mode": mode,
                "mean_generation_tps": statistics.fmean(vals),
                "median_generation_tps": statistics.median(vals),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
    )


if __name__ == "__main__":
    main()
