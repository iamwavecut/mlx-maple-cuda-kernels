# Copyright © 2026 DeepGrove AI.

"""Benchmark Maple's decode-only custom kernels against portable MLX ops."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402

from mlx_lm.models import maple  # noqa: E402


def _model_args(layer_type="sliding_attention", num_experts=256):
    return maple.ModelArgs(
        num_hidden_layers=1,
        num_experts=num_experts,
        num_experts_per_tok=8,
        hidden_size=2048,
        moe_intermediate_size=512,
        vocab_size=1024,
        layer_types=[layer_type],
    )


def _outputs(value):
    return value if isinstance(value, (tuple, list)) else (value,)


def _evaluate(call):
    outputs = _outputs(call())
    mx.eval(*outputs)
    mx.synchronize()
    return outputs




def _benchmark_matches(fast, reference, tol=2e-2):
    """Numerical candidate gate; strict auto-probes in maple.py are exact."""
    try:
        got, want = fast(), reference()
        mx.eval(got, want)
    except Exception:
        return False
    return len(got) == len(want) and all(
        g.shape == w.shape
        and bool(
            mx.allclose(g.astype(mx.float32), w.astype(mx.float32), rtol=tol, atol=tol)
        )
        for g, w in zip(got, want)
    )


def _percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _measure(call, warmup, trials):
    _evaluate(call)  # JIT/compile outside warmup and measurement.
    for _ in range(warmup):
        _evaluate(call)
    samples = []
    for _ in range(trials):
        mx.synchronize()
        start = time.perf_counter_ns()
        _evaluate(call)
        samples.append((time.perf_counter_ns() - start) / 1_000.0)
    return {
        "median_us": statistics.median(samples),
        "p10_us": _percentile(samples, 0.10),
        "p90_us": _percentile(samples, 0.90),
        "samples_us": samples,
    }


def _add_rms_case():
    dim, eps = 2048, 1e-6
    x = (mx.random.normal((1, 1, dim)) * 0.5).astype(mx.bfloat16)
    residual = (mx.random.normal((1, 1, dim)) * 0.5).astype(mx.bfloat16)
    weight = (mx.random.normal((dim,)) * 0.1 + 1.0).astype(mx.bfloat16)
    mx.eval(x, residual, weight)

    def fast():
        return maple._add_rms_norm(x, residual, weight, eps)

    def reference():
        summed = x + residual
        return (
            summed,
            mx.fast.rms_norm(
                summed.astype(mx.float32), weight.astype(mx.float32), eps
            ).astype(mx.bfloat16),
        )

    if not _benchmark_matches(fast, reference):
        raise RuntimeError("add_rms_norm failed correctness gate")
    return fast, reference


def _qk_case(use_rope):
    args = _model_args("sliding_attention" if use_rope else "full_attention")
    attention = maple.MapleAttention(args, 0)
    attention.q_norm.weight = (mx.random.normal((args.head_dim,)) * 0.1 + 1.0).astype(
        mx.bfloat16
    )
    attention.k_norm.weight = (mx.random.normal((args.head_dim,)) * 0.1 + 1.0).astype(
        mx.bfloat16
    )
    heads = args.num_attention_heads + args.num_key_value_heads
    qk = (mx.random.normal((heads, args.head_dim)) * 0.5).astype(mx.bfloat16)
    mx.eval(attention.parameters(), qk)

    def fast():
        return (attention._qk_fused(qk, 613),)

    def reference():
        return (attention._qk_reference(qk, 613),)

    if not _benchmark_matches(fast, reference):
        raise RuntimeError("qk_norm failed correctness gate")
    return fast, reference


def _router_case():
    args = _model_args(num_experts=256)
    gate = maple.MapleGate(args)
    gate.weight = (
        mx.random.normal((args.num_experts, args.hidden_size)) * 0.05
    ).astype(mx.bfloat16)
    x = (mx.random.normal((1, 1, args.hidden_size)) * 0.5).astype(mx.bfloat16)
    mx.eval(gate.weight, x)

    def fast():
        return gate._fused_call(x)

    def reference():
        return gate._reference(x)

    inds, scores = _evaluate(fast)
    ref_inds, ref_scores = _evaluate(reference)
    if not (
        bool(mx.all((inds >= 0) & (inds < args.num_experts)))
        and bool(
            mx.allclose(mx.sort(scores), mx.sort(ref_scores), rtol=1e-5, atol=1e-5)
        )
    ):
        raise RuntimeError("router failed correctness gate")
    return fast, reference


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_sha():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return os.environ.get("MAPLE_SOURCE_SHA")


def _environment():
    source = ROOT / "mlx_lm/models/maple.py"
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
        "platform": platform.platform(),
        "mlx": _package_version("mlx"),
        "mlx_cuda": _package_version("mlx-cuda-12"),
        "git_sha": _git_sha(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _apply_router_override(threads, rows_per_warp):
    if threads is None and rows_per_warp is None:
        return
    if threads is None or rows_per_warp is None:
        raise ValueError("router override requires both threads and rows-per-warp")
    profile = maple._cuda_profile()
    if profile is None:
        raise RuntimeError("router override requires a supported CUDA device")
    maple._cuda_profile_cache = maple._CudaProfile(
        f"{profile.name}_t{threads}_r{rows_per_warp}",
        elementwise_threads=profile.elementwise_threads,
        router_threads=threads,
        router_rows_per_warp=rows_per_warp,
        router_reference_gemv=profile.router_reference_gemv,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--router-threads", type=int)
    parser.add_argument("--router-rows-per-warp", type=int)
    args = parser.parse_args()
    if args.warmup < 0 or args.trials < 1:
        parser.error("warmup must be nonnegative and trials must be positive")

    try:
        _apply_router_override(args.router_threads, args.router_rows_per_warp)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))

    backend = maple._kernel_backend()
    if backend is None or (backend == "cuda" and maple._cuda_profile() is None):
        raise RuntimeError("no supported custom-kernel backend")

    mx.random.seed(20260806)
    cases = (
        ("add_rms_norm", _add_rms_case()),
        ("qk_norm_rope", _qk_case(True)),
        ("qk_norm_nope", _qk_case(False)),
        ("router", _router_case()),
    )
    environment = _environment()
    records = [environment]
    for name, (fast, reference) in cases:
        fast_timing = _measure(fast, args.warmup, args.trials)
        reference_timing = _measure(reference, args.warmup, args.trials)
        records.append(
            {
                "type": "result",
                "kernel": name,
                "backend": environment["backend"],
                "profile": environment["profile"],
                **fast_timing,
                "reference_median_us": reference_timing["median_us"],
                "reference_p10_us": reference_timing["p10_us"],
                "reference_p90_us": reference_timing["p90_us"],
                "reference_samples_us": reference_timing["samples_us"],
                "speedup": reference_timing["median_us"] / fast_timing["median_us"],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    for record in records[1:]:
        print(
            f"{record['kernel']}: {record['median_us']:.2f} us, "
            f"{record['speedup']:.2f}x"
        )


if __name__ == "__main__":
    main()
