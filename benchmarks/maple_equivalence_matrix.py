#!/usr/bin/env python3
"""Long multi-seed correctness-only gate for conservative strict-auto Maple."""

import argparse
import hashlib
import inspect
import json
import random
from pathlib import Path

import mlx.core as mx
from maple_common_slice_benchmark import run
from mlx_lm import load
from mlx_lm.models import maple


def configure(model, optimized, state=None):
    if optimized and state is not None:
        model.model._fused_add_norm = state["add_rms_norm"]
        for layer, qk, router in zip(
            model.model.layers, state["qk_norm"], state["router"]
        ):
            layer.self_attn._fused_qk = qk
            layer.mlp.gate._fused = router
    else:
        value = None if optimized else False
        model.model._fused_add_norm = value
        for layer in model.model.layers:
            layer.self_attn._fused_qk = value
            layer.mlp.gate._fused = value
    maple._use_cached_decode_lhs = False
    maple._cuda_router_indices_uint32 = False
    maple._use_cuda_ternary_up_gate = False
    maple._use_approximate_router = False
    maple._use_approximate_add_rms = False


def state(model):
    return {
        "add_rms_norm": model.model._fused_add_norm,
        "qk_norm": [layer.self_attn._fused_qk for layer in model.model.layers],
        "router": [layer.mlp.gate._fused for layer in model.model.layers],
        "cached_decode_lhs": maple._use_cached_decode_lhs,
        "router_indices_uint32": maple._cuda_router_indices_uint32,
        "approximate_router": maple._use_approximate_router,
        "approximate_add_rms": maple._use_approximate_add_rms,
        "ternary_up_gate": maple._use_cuda_ternary_up_gate,
    }


def mismatch(a, b):
    if a == b:
        return None
    for index, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return index
    return min(len(a), len(b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model, tokenizer, config = load(
        str(args.model),
        return_config=True,
        model_config={"model_file": None, "use_flash_head": False},
        tokenizer_config={"trust_remote_code": False},
        trust_remote_code=False,
    )
    source = Path(inspect.getfile(type(model))).resolve()
    module = Path(maple.__file__).resolve()
    if source != module:
        raise RuntimeError(f"loaded {source}; expected patched module {module}")
    tokenizer._eos_token_ids = {}
    vocab = config.get("vocab_size") or config["text_config"]["vocab_size"]
    cases = [
        (20260806, 128, 1024),
        (20260807, 17, 2048),
        (20260808, 511, 1024),
    ]
    records = [
        {
            "type": "environment",
            "device": dict(mx.device_info(mx.gpu)),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "config_model_file": config.get("model_file"),
            "cases": [
                {"seed": seed, "prompt_tokens": prompt_len, "max_tokens": cap}
                for seed, prompt_len, cap in cases
            ],
        }
    ]
    strict_state = None
    ok = True
    for case_index, (seed, prompt_len, cap) in enumerate(cases, 1):
        rng = random.Random(seed)
        prompt = [rng.randrange(vocab) for _ in range(prompt_len)]
        outputs = {}
        for mode in ("reference_1", "strict", "reference_2"):
            optimized = mode == "strict"
            configure(model, optimized, strict_state if optimized else None)
            result = run(model, tokenizer, prompt, cap)
            if optimized and strict_state is None:
                strict_state = state(model)
            outputs[mode] = result
            records.append(
                {
                    "type": "case_mode",
                    "case": case_index,
                    "seed": seed,
                    "prompt_tokens": prompt_len,
                    "max_tokens": cap,
                    "prompt_sha256": hashlib.sha256(
                        ",".join(map(str, prompt)).encode()
                    ).hexdigest(),
                    "mode": mode,
                    "generated_tokens": len(result["tokens"]),
                    **{
                        key: value
                        for key, value in result.items()
                        if key not in {"tokens", "text"}
                    },
                }
            )
        artifact_keys = [
            "token_sha256",
            "text_sha256",
            "selected_logprob_sha256",
            "top1_sha256",
            "finish_reason",
        ]
        reference_stable = all(
            outputs["reference_1"][key] == outputs["reference_2"][key]
            for key in artifact_keys
        ) and outputs["reference_1"]["tokens"] == outputs["reference_2"]["tokens"]
        strict_equal = all(
            outputs["reference_1"][key] == outputs["strict"][key]
            for key in artifact_keys
        ) and outputs["reference_1"]["tokens"] == outputs["strict"]["tokens"]
        ok = ok and reference_stable and strict_equal
        records.append(
            {
                "type": "case_comparison",
                "case": case_index,
                "reference_stable": reference_stable,
                "strict_equal": strict_equal,
                "strict_first_token_mismatch": mismatch(
                    outputs["reference_1"]["tokens"], outputs["strict"]["tokens"]
                ),
                "reference_first_token_mismatch": mismatch(
                    outputs["reference_1"]["tokens"],
                    outputs["reference_2"]["tokens"],
                ),
            }
        )
    records.append({"type": "strict_path_state", **strict_state})
    records.append({"type": "summary", "all_cases_exact": ok, "cases": len(cases)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    print(json.dumps(records[-1], sort_keys=True))
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
