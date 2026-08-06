#!/usr/bin/env python3
"""Canonical strict Maple generation entry point for an NVIDIA process."""

import argparse
import inspect
import json
import os
from pathlib import Path


_REQUIRED_ENV = {
    "MLX_CUDA_USE_CUDNN_SDPA": "0",
    "MLX_ENABLE_TF32": "0",
    "MLX_USE_CUDA_GRAPHS": "1",
    "MLX_CUDA_GRAPH_CACHE_SIZE": "400",
    "MLX_MAX_OPS_PER_BUFFER": "100",
    "MLX_MAX_MB_PER_BUFFER": "100",
}


def _qk_state(model):
    values = [layer.self_attn._fused_qk for layer in model.model.layers]
    return {
        "cuda_active_layers": sum(value is True for value in values),
        "portable_fallback_layers": sum(value is False for value in values),
        "unresolved_layers": sum(value is None for value in values),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run exact-head Maple inference from the patched package source."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--cached-lhs",
        action="store_true",
        help=(
            "Enable the validated warm B=1/L=1 cached-LHS experiment. Use only "
            "in a fresh, single-device process; the cache is keyed only by top-k."
        ),
    )
    args = parser.parse_args()

    if args.max_tokens < 1:
        parser.error("--max-tokens must be at least 1")
    wrong_env = {
        name: os.environ.get(name)
        for name, expected in _REQUIRED_ENV.items()
        if os.environ.get(name) != expected
    }
    if wrong_env:
        raise RuntimeError(
            f"strict process environment mismatch: {wrong_env}; expected {_REQUIRED_ENV}"
        )

    # Import MLX only after the process contract has been validated.
    from mlx_lm import generate, load
    from mlx_lm.models import maple

    # Keep all known approximate paths out of this entry point.
    maple._use_approximate_router = False
    maple._use_approximate_add_rms = False
    maple._use_cuda_ternary_up_gate = False
    maple._cuda_router_indices_uint32 = False
    maple._use_cached_decode_lhs = args.cached_lhs

    model, tokenizer = load(
        str(args.model),
        model_config={"model_file": None, "use_flash_head": False},
        tokenizer_config={"trust_remote_code": False},
        trust_remote_code=False,
    )

    loaded_source = Path(inspect.getfile(type(model))).resolve()
    package_source = Path(maple.__file__).resolve()
    if loaded_source != package_source:
        raise RuntimeError(
            f"loaded checkpoint-local source {loaded_source}; expected {package_source}"
        )

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=args.max_tokens,
        prefill_step_size=2048,
        verbose=True,
    )

    print(
        "strict_path_state="
        + json.dumps(
            {
                "model_source": str(package_source),
                "exact_lm_head": True,
                "cached_lhs": args.cached_lhs,
                "qk_norm_rope": _qk_state(model),
                "approximate_router": False,
                "approximate_add_rms": False,
                "ternary_up_gate": False,
                "kv_quantization": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
