#!/usr/bin/env python3
"""Measure exact Maple decode throughput on an Apple Silicon MLX host."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(model, tokenizer, prompt, generation_tokens):
    tokens = []
    response = None
    mx.reset_peak_memory()
    for response in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=generation_tokens,
        prefill_step_size=2048,
    ):
        tokens.append(int(response.token))
    if response is None:
        raise RuntimeError("generation returned no response")
    return {
        "prompt_tps": response.prompt_tps,
        "generation_tps": response.generation_tps,
        "peak_memory_gb": response.peak_memory,
        "token_sha256": hashlib.sha256(",".join(map(str, tokens)).encode()).hexdigest(),
        "tokens": len(tokens),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", default="deepgrove/maple-preview-2bit-mlx")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--generation-tokens", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--gpu-cores", type=int, required=True)
    parser.add_argument(
        "--condition",
        choices=("quiescent", "interactive_non_quiescent"),
        required=True,
    )
    args = parser.parse_args()
    if args.trials < 1 or args.warmup < 0:
        parser.error("trials must be positive and warmup must be nonnegative")

    config_path = args.model / "config.json"
    model_source = args.model / "maple.py"
    if not config_path.is_file() or not model_source.is_file():
        parser.error("model directory must contain config.json and maple.py")

    mx.random.seed(20260806)
    model, tokenizer, config = load(
        str(args.model),
        return_config=True,
        tokenizer_config={"trust_remote_code": True},
        model_config={"use_flash_head": False},
        trust_remote_code=True,
    )
    tokenizer._eos_token_ids = {}
    vocab_size = config.get("vocab_size") or config["text_config"]["vocab_size"]
    prompt = mx.random.randint(0, vocab_size, (args.prompt_tokens,)).tolist()
    device = dict(mx.device_info(mx.gpu))
    records = [
        {
            "type": "environment",
            "backend": "metal",
            "device_name": device.get("device_name"),
            "device_architecture": device.get("architecture"),
            "gpu_cores": args.gpu_cores,
            "memory_gb": round(device.get("memory_size", 0) / 1e9, 3),
            "os": f"macOS {platform.mac_ver()[0]}",
            "python": platform.python_version(),
            "mlx": _version("mlx"),
            "mlx_lm": _version("mlx-lm"),
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "model_config_sha256": _sha256(config_path),
            "model_source_sha256": _sha256(model_source),
            "flash_head": False,
            "condition": args.condition,
            "prompt_tokens": args.prompt_tokens,
            "prompt_sha256": hashlib.sha256(
                ",".join(map(str, prompt)).encode()
            ).hexdigest(),
        }
    ]

    for generation_tokens in args.generation_tokens:
        for _ in range(args.warmup):
            _run(model, tokenizer, prompt, generation_tokens)
        for trial in range(1, args.trials + 1):
            result = _run(model, tokenizer, prompt, generation_tokens)
            record = {
                "type": "trial",
                "trial": trial,
                "prompt_tokens": args.prompt_tokens,
                "generation_tokens": generation_tokens,
                **result,
            }
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)

        selected = [
            record
            for record in records
            if record.get("type") == "trial"
            and record["generation_tokens"] == generation_tokens
        ]
        values = [record["generation_tps"] for record in selected]
        records.append(
            {
                "type": "summary",
                "prompt_tokens": args.prompt_tokens,
                "generation_tokens": generation_tokens,
                "trials": len(values),
                "mean_generation_tps": statistics.fmean(values),
                "median_generation_tps": statistics.median(values),
                "min_generation_tps": min(values),
                "max_generation_tps": max(values),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )


if __name__ == "__main__":
    main()
