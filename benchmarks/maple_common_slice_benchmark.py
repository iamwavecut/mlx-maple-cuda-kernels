#!/usr/bin/env python3
"""Run a fixed 20-question cognitive regression slice in paired Maple modes."""

import argparse
import hashlib
import inspect
import json
import os
import re
import struct
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models import maple


def set_mode(model, optimized, probe=False, state=None):
    """Select portable or strict-auto paths without bypassing failed probes."""
    if optimized and state is not None:
        model.model._fused_add_norm = state["add_rms_norm"]
        for layer, qk, router in zip(
            model.model.layers, state["qk_norm"], state["router"]
        ):
            layer.self_attn._fused_qk = qk
            layer.mlp.gate._fused = router
    else:
        value = None if optimized and probe else False
        model.model._fused_add_norm = value
        for layer in model.model.layers:
            layer.self_attn._fused_qk = value
            layer.mlp.gate._fused = value
    maple._use_cached_decode_lhs = optimized
    # The uint32 shortcut belongs to the disabled approximate router.
    maple._cuda_router_indices_uint32 = False
    # Approximate kernels are separate semantic lanes, never strict auto.
    maple._use_cuda_ternary_up_gate = False
    maple._use_approximate_router = False
    maple._use_approximate_add_rms = False


def fast_state(model):
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


def prompt_for(case):
    if case["choices"]:
        choices = "\n".join(
            f"{chr(65 + i)}. {choice}" for i, choice in enumerate(case["choices"])
        )
        answer_rule = "Return the option letter on a final line exactly as: Answer: X"
        body = f"{case['question']}\n\nChoices:\n{choices}\n\n{answer_rule}"
    else:
        body = (
            f"{case['question']}\n\n"
            "Return the requested nonnegative integer on a final line exactly as: "
            "Answer: N"
        )
    return [
        {
            "role": "system",
            "content": (
                "Solve the benchmark problem carefully. You may reason before the "
                "answer, but obey the required final-answer format."
            ),
        },
        {"role": "user", "content": body},
    ]


def extract_answer(text, choices):
    """Strict official extraction; loose tail guesses are intentionally excluded."""
    if choices:
        matches = re.findall(r"(?im)^\s*(?:final\s+)?answer\s*:\s*([A-J])\s*$", text)
        if not matches:
            return None, "missing_final_answer"
        picked = matches[-1].upper()
        if ord(picked) - ord("A") >= len(choices):
            return None, "choice_out_of_range"
        return picked, "final_answer_line"
    matches = re.findall(r"(?im)^\s*(?:final\s+)?answer\s*:\s*([0-9]+)\s*$", text)
    if not matches:
        return None, "missing_final_answer"
    return str(int(matches[-1])), "final_answer_line"


def run(model, tokenizer, prompt, max_tokens):
    mx.reset_peak_memory()
    tokens = []
    pieces = []
    selected_lp_hash = hashlib.sha256()
    top1_hash = hashlib.sha256()
    response = None
    started = time.perf_counter()
    for response in stream_generate(
        model, tokenizer, prompt, max_tokens=max_tokens, prefill_step_size=2048
    ):
        token = int(response.token)
        tokens.append(token)
        pieces.append(response.text)
        selected = float(response.logprobs[token].item())
        top1 = int(mx.argmax(response.logprobs).item())
        selected_lp_hash.update(struct.pack("!d", selected))
        top1_hash.update(struct.pack("!I", top1))
    elapsed = time.perf_counter() - started
    if response is None:
        raise RuntimeError("generation produced no response")
    return {
        "tokens": tokens,
        "text": "".join(pieces),
        "token_sha256": hashlib.sha256(
            ",".join(map(str, tokens)).encode()
        ).hexdigest(),
        "text_sha256": hashlib.sha256("".join(pieces).encode()).hexdigest(),
        "selected_logprob_sha256": selected_lp_hash.hexdigest(),
        "top1_sha256": top1_hash.hexdigest(),
        "generation_tps": response.generation_tps,
        "prompt_tps": response.prompt_tps,
        "peak_memory": response.peak_memory,
        "finish_reason": getattr(response, "finish_reason", None),
        "elapsed": elapsed,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument(
        "--manifest", type=Path,
        default=Path(__file__).parent / "data/maple_common_slice_20.json",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--limit", type=int, default=20)
    args = p.parse_args()

    manifest = json.loads(args.manifest.read_text())
    cases = manifest["cases"][: args.limit]
    model, tokenizer, config = load(
        str(args.model), return_config=True,
        model_config={"model_file": None, "use_flash_head": False},
        tokenizer_config={"trust_remote_code": False}, trust_remote_code=False,
    )
    source = Path(inspect.getfile(type(model)))
    module_source = Path(inspect.getfile(maple))
    if source.resolve() != module_source.resolve():
        raise RuntimeError(
            f"loaded model source {source} differs from worktree module {module_source}"
        )
    records = [{
        "type": "environment", "device": dict(mx.device_info(mx.gpu)),
        "mlx": mx.__version__, "model": str(args.model.resolve()),
        "model_source": str(source.resolve()),
        "model_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "module_source": str(module_source.resolve()),
        "module_source_sha256": hashlib.sha256(module_source.read_bytes()).hexdigest(),
        "mlx_cuda_use_cudnn_sdpa": os.environ.get("MLX_CUDA_USE_CUDNN_SDPA"),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "manifest_name": manifest["name"],
        "upstream_commit": manifest["upstream_commit"],
        "config_model_file": config.get("model_file"),
        "max_tokens": args.max_tokens, "cases": len(cases),
    }]
    strict_state = None
    paired = []
    for case_idx, case in enumerate(cases):
        rendered = tokenizer.apply_chat_template(
            prompt_for(case), tokenize=True, add_generation_prompt=True
        )
        order = ["reference", "strict"] if case_idx % 2 == 0 else ["strict", "reference"]
        case_results = {}
        for mode in order:
            optimized = mode == "strict"
            probe = optimized and strict_state is None
            set_mode(model, optimized, probe, strict_state if optimized else None)
            result = run(model, tokenizer, rendered, args.max_tokens)
            if probe:
                state = fast_state(model)
                resolved = state["add_rms_norm"] is not None and all(
                    value is not None for value in state["qk_norm"] + state["router"]
                )
                approximate = any(
                    state[key] for key in (
                        "ternary_up_gate", "approximate_router", "approximate_add_rms"
                    )
                )
                if not resolved or approximate:
                    raise RuntimeError(f"strict live probe failed: {state}")
                strict_state = state
                records.append({"type": "strict_probe", "fast_path_state": state})
            picked, extraction_method = extract_answer(result["text"], case["choices"])
            expected = case["answer"].upper() if case["choices"] else str(int(case["answer"]))
            record = {
                "type": "case", "mode": mode, "case_index": case_idx + 1,
                "source": case["source"], "source_id": case["source_id"],
                "domain": case.get("domain"), "expected": expected,
                "picked": picked, "extraction_method": extraction_method,
                "correct": picked == expected,
                "prompt_tokens": len(rendered),
                "prompt_sha256": hashlib.sha256(
                    ",".join(map(str, rendered)).encode()
                ).hexdigest(),
                "generated_tokens": len(result["tokens"]),
                **{k: v for k, v in result.items() if k not in ("tokens", "text")},
                "output_text": result["text"],
            }
            records.append(record)
            case_results[mode] = {"record": record, "tokens": result["tokens"]}
            print(json.dumps({k: v for k, v in record.items() if k != "output_text"}, sort_keys=True), flush=True)
        ref = case_results["reference"]
        opt = case_results["strict"]
        rt, ot = ref["tokens"], opt["tokens"]
        first_mismatch = next(
            (i for i, pair in enumerate(zip(rt, ot)) if pair[0] != pair[1]),
            None,
        )
        if first_mismatch is None and len(rt) != len(ot):
            first_mismatch = min(len(rt), len(ot))
        comparison = {
            "type": "comparison", "case_index": case_idx + 1,
            "source": case["source"], "source_id": case["source_id"],
            "tokens_equal": rt == ot, "first_token_mismatch": first_mismatch,
            "picked_equal": ref["record"]["picked"] == opt["record"]["picked"],
            "text_hash_equal": ref["record"]["text_sha256"] == opt["record"]["text_sha256"],
            "selected_logprobs_equal": (
                ref["record"]["selected_logprob_sha256"]
                == opt["record"]["selected_logprob_sha256"]
            ),
            "top1_equal": ref["record"]["top1_sha256"] == opt["record"]["top1_sha256"],
            "grade_equal": ref["record"]["correct"] == opt["record"]["correct"],
        }
        records.append(comparison)
        paired.append(comparison)

    for mode in ("reference", "strict"):
        selected = [r for r in records if r.get("type") == "case" and r["mode"] == mode]
        by_source = defaultdict(lambda: [0, 0])
        for r in selected:
            by_source[r["source"]][1] += 1
            by_source[r["source"]][0] += int(r["correct"])
        records.append({
            "type": "summary", "mode": mode,
            "correct": sum(int(r["correct"]) for r in selected),
            "total": len(selected),
            "accuracy": sum(int(r["correct"]) for r in selected) / len(selected),
            "by_source": {
                key: {"correct": val[0], "total": val[1], "accuracy": val[0] / val[1]}
                for key, val in sorted(by_source.items())
            },
        })
    records.append({
        "type": "paired_summary", "cases": len(paired),
        "all_tokens_equal": all(r["tokens_equal"] for r in paired),
        "all_selected_logprobs_equal": all(r["selected_logprobs_equal"] for r in paired),
        "all_top1_equal": all(r["top1_equal"] for r in paired),
        "all_grades_equal": all(r["grade_equal"] for r in paired),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records))
    print(json.dumps(records[-3:], ensure_ascii=False, sort_keys=True, indent=2))
    exact_fields = (
        "tokens_equal", "text_hash_equal", "selected_logprobs_equal", "top1_equal"
    )
    if not all(all(item[field] for field in exact_fields) for item in paired):
        raise RuntimeError("strict auto changed at least one required decode artifact")


if __name__ == "__main__":
    main()
