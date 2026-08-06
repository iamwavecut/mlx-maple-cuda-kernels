# Copyright © 2026 DeepGrove AI.

"""Compare portable and custom-kernel Maple decode in one loaded process."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import statistics
import time
from pathlib import Path

import mlx.core as mx
from maple_kernel_benchmark import _apply_router_override, _git_sha
from mlx_lm import load, stream_generate
from mlx_lm.models import maple

ROOT = Path(__file__).resolve().parents[1]


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _environment(model_path, config):
    device = dict(mx.device_info(mx.gpu))
    profile = maple._cuda_profile()
    return {
        "type": "environment",
        "backend": maple._kernel_backend(),
        "profile": profile.name if profile is not None else None,
        "profile_parameters": (
            {
                "elementwise_threads": profile.elementwise_threads,
                "router_threads": profile.router_threads,
                "router_rows_per_warp": profile.router_rows_per_warp,
                "router_reference_gemv": profile.router_reference_gemv,
            }
            if profile is not None
            else None
        ),
        "device": device,
        "gpu_uuid": device.get("uuid"),
        "compute_capability": device.get("architecture"),
        "python": platform.python_version(),
        "mlx": _package_version("mlx"),
        "mlx_cuda": _package_version("mlx-cuda-12"),
        "git_sha": _git_sha(),
        "source_sha256": hashlib.sha256(Path(maple.__file__).read_bytes()).hexdigest(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model_path": str(model_path.resolve()),
        "model_name": config.get("_name_or_path"),
    }


def _set_fast_paths(model, enabled, state=None):
    strict_auto = enabled is not False
    maple._use_cached_decode_lhs = strict_auto
    maple._cuda_router_indices_uint32 = False
    maple._use_cuda_ternary_up_gate = False
    maple._use_approximate_router = False
    maple._use_approximate_add_rms = False
    if strict_auto and state is not None:
        model.model._fused_add_norm = state["add_rms_norm"]
        for layer, qk, router in zip(
            model.model.layers, state["qk_norm"], state["router"]
        ):
            layer.self_attn._fused_qk = qk
            layer.mlp.gate._fused = router
        return
    model.model._fused_add_norm = enabled
    for layer in model.model.layers:
        layer.self_attn._fused_qk = enabled
        layer.mlp.gate._fused = enabled


def _fast_path_state(model):
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


def _run(model, tokenizer, prompt, generation_tokens):
    mx.reset_peak_memory()
    started = time.perf_counter()
    tokens = []
    response = None
    for response in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=generation_tokens,
        prefill_step_size=2048,
    ):
        tokens.append(int(response.token))
    elapsed = time.perf_counter() - started
    if response is None:
        raise RuntimeError("generation returned no response")
    return {
        "tokens": tokens,
        "token_sha256": hashlib.sha256(",".join(map(str, tokens)).encode()).hexdigest(),
        "generated_tokens": len(tokens),
        "prompt_tps": response.prompt_tps,
        "generation_tps": response.generation_tps,
        "peak_memory": response.peak_memory,
        "total_time": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--generation-tokens", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--equivalence-tokens", type=int, default=256)
    parser.add_argument("--trials", type=int, default=5)
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
        model_config={"model_file": None, "use_flash_head": False},
        tokenizer_config={"trust_remote_code": False},
        trust_remote_code=False,
    )
    source = Path(inspect.getfile(type(model))).resolve()
    module_source = Path(maple.__file__).resolve()
    if source != module_source:
        raise RuntimeError(f"loaded model source {source} differs from {module_source}")
    tokenizer._eos_token_ids = {}
    vocab_size = config.get("vocab_size") or config["text_config"]["vocab_size"]
    prompt = mx.random.randint(0, vocab_size, (args.prompt_tokens,)).tolist()

    _set_fast_paths(model, False)
    reference_equivalence = _run(model, tokenizer, prompt, args.equivalence_tokens)
    _set_fast_paths(model, None)
    auto_equivalence = _run(model, tokenizer, prompt, args.equivalence_tokens)
    fast_state = _fast_path_state(model)
    all_resolved = fast_state["add_rms_norm"] is not None and all(
        value is not None
        for value in fast_state["qk_norm"] + fast_state["router"]
    )
    tokens_equal = reference_equivalence["tokens"] == auto_equivalence["tokens"]
    mismatch = None
    if not tokens_equal:
        mismatch = next(
            (
                index
                for index, (reference, auto) in enumerate(
                    zip(
                        reference_equivalence["tokens"],
                        auto_equivalence["tokens"],
                    )
                )
                if reference != auto
            ),
            min(
                len(reference_equivalence["tokens"]),
                len(auto_equivalence["tokens"]),
            ),
        )

    records = [
        _environment(args.model, config),
        {
            "type": "equivalence",
            "tokens_equal": tokens_equal,
            "mismatch_index": mismatch,
            "tokens": args.equivalence_tokens,
            "fast_path_state": fast_state,
            "reference_sha256": reference_equivalence["token_sha256"],
            "auto_sha256": auto_equivalence["token_sha256"],
        },
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    if not all_resolved:
        raise RuntimeError(f"a live fast-path probe failed: {fast_state}")
    if not tokens_equal:
        raise RuntimeError(f"reference and auto tokens diverged at index {mismatch}")
    for generation_tokens in args.generation_tokens:
        warmup_hash = None
        for mode, enabled in (("reference", False), ("auto", True)):
            _set_fast_paths(model, enabled, fast_state if enabled else None)
            result = _run(model, tokenizer, prompt, generation_tokens)
            warmup_hash = result["token_sha256"] if warmup_hash is None else warmup_hash
            if result["token_sha256"] != warmup_hash:
                raise RuntimeError(f"warmup token mismatch for {generation_tokens} {mode}")

        for trial in range(args.trials):
            order = (("reference", False), ("auto", True))
            if trial % 2:
                order = tuple(reversed(order))
            for mode, enabled in order:
                _set_fast_paths(model, enabled, fast_state if enabled else None)
                result = _run(model, tokenizer, prompt, generation_tokens)
                if result["token_sha256"] != warmup_hash:
                    raise RuntimeError(
                        f"timed token mismatch for {generation_tokens} trial {trial + 1} {mode}"
                    )
                records.append(
                    {
                        "type": "trial",
                        "mode": mode,
                        "trial": trial + 1,
                        "prompt_tokens": args.prompt_tokens,
                        "generation_tokens": generation_tokens,
                        **{
                            key: value
                            for key, value in result.items()
                            if key != "tokens"
                        },
                    }
                )

    for generation_tokens in args.generation_tokens:
        for mode in ("reference", "auto"):
            selected = [
                record
                for record in records
                if record.get("type") == "trial"
                and record["mode"] == mode
                and record["generation_tokens"] == generation_tokens
            ]
            values = [record["generation_tps"] for record in selected]
            records.append(
                {
                    "type": "summary",
                    "mode": mode,
                    "prompt_tokens": args.prompt_tokens,
                    "generation_tokens": generation_tokens,
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
    for record in records:
        if record["type"] == "summary":
            print(
                record["mode"],
                record["generation_tokens"],
                f"{record['mean_generation_tps']:.3f} tok/s",
            )


if __name__ == "__main__":
    main()
