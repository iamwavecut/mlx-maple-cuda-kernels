#!/usr/bin/env python3
"""One fresh-process R/Q/QL timing block with exact token controls."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import mlx.core as mx
from maple_auto_benchmark import run
from mlx_lm import load
from mlx_lm.models import maple


def configure(model, mode, qk_state=None):
    maple._use_cached_decode_lhs = mode == "QL"
    maple._cuda_router_indices_uint32 = False
    maple._use_cuda_ternary_up_gate = False
    maple._use_approximate_router = False
    maple._use_approximate_add_rms = False
    model.model._fused_add_norm = False
    for index, layer in enumerate(model.model.layers):
        layer.self_attn._fused_qk = (
            False if mode == "R" else (qk_state[index] if qk_state else None)
        )
        layer.mlp.gate._fused = False


def path_state(model):
    return {
        "add_rms_norm": model.model._fused_add_norm,
        "qk_norm": [layer.self_attn._fused_qk for layer in model.model.layers],
        "router": [layer.mlp.gate._fused for layer in model.model.layers],
        "cached_decode_lhs": maple._use_cached_decode_lhs,
        "router_indices_uint32": maple._cuda_router_indices_uint32,
        "ternary_up_gate": maple._use_cuda_ternary_up_gate,
        "approximate_router": maple._use_approximate_router,
        "approximate_add_rms": maple._use_approximate_add_rms,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--order", nargs=3, choices=["R", "Q", "QL"], required=True)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--generation-tokens", type=int, default=512)
    args = parser.parse_args()
    if set(args.order) != {"R", "Q", "QL"}:
        raise RuntimeError("order must contain R, Q and QL exactly once")
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
    expected = None
    qk_state = None
    for mode in ["R", "Q", "QL"]:
        configure(model, mode, qk_state)
        result = run(model, tokenizer, prompt, args.generation_tokens)
        if mode == "Q":
            qk_state = path_state(model)["qk_norm"]
            if any(value is None for value in qk_state):
                raise RuntimeError("Q/K live probe did not resolve")
        expected = result["token_sha256"] if expected is None else expected
        if result["token_sha256"] != expected:
            raise RuntimeError(f"warm token mismatch for {mode}")
    records = [
        {
            "type": "environment",
            "block": args.block,
            "order": args.order,
            "device": dict(mx.device_info(mx.gpu)),
            "mlx": mx.__version__,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "prompt_sha256": hashlib.sha256(
                ",".join(map(str, prompt)).encode()
            ).hexdigest(),
            "qk_state": qk_state,
        }
    ]
    for position, mode in enumerate(args.order, 1):
        configure(model, mode, qk_state)
        result = run(model, tokenizer, prompt, args.generation_tokens)
        if result["token_sha256"] != expected:
            raise RuntimeError(f"timed token mismatch for {mode}")
        records.append(
            {
                "type": "trial",
                "block": args.block,
                "position": position,
                "mode": mode,
                **{key: value for key, value in result.items() if key != "tokens"},
            }
        )
    records.append({"type": "path_state", **path_state(model)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    print(json.dumps(records[-1], sort_keys=True))


if __name__ == "__main__":
    main()
