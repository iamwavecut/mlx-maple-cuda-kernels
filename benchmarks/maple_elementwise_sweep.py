# Copyright © 2026 DeepGrove AI.

"""Sweep CUDA residual-add/RMSNorm block sizes with correctness gates."""

import argparse
import json
from pathlib import Path

import mlx.core as mx

from mlx_lm.models import maple

from maple_kernel_benchmark import (
    _add_rms_case,
    _environment,
    _evaluate,
    _measure,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--stress-dispatches", type=int, default=128)
    args = parser.parse_args()

    if maple._kernel_backend() != "cuda" or maple._cuda_profile() is None:
        raise RuntimeError("elementwise profile sweep requires a supported CUDA device")
    base_profile = maple._cuda_profile()
    records = [_environment()]
    for threads in args.threads:
        record = {"type": "elementwise_sweep", "threads": threads}
        try:
            maple._cuda_profile_cache = maple._CudaProfile(
                f"{base_profile.name}_e{threads}",
                elementwise_threads=threads,
                router_threads=base_profile.router_threads,
                router_rows_per_warp=base_profile.router_rows_per_warp,
                router_reference_gemv=base_profile.router_reference_gemv,
            )
            mx.random.seed(20260806)
            fast, reference = _add_rms_case()
            for dispatch in range(args.stress_dispatches):
                fast_residual, fast_norm = _evaluate(fast)
                ref_residual, ref_norm = _evaluate(reference)
                if not bool(mx.array_equal(fast_residual, ref_residual)):
                    raise RuntimeError(f"dispatch {dispatch}: residual mismatch")
                if not bool(mx.allclose(fast_norm, ref_norm, rtol=8e-3, atol=8e-3)):
                    raise RuntimeError(f"dispatch {dispatch}: norm mismatch")
            fast_timing = _measure(fast, args.warmup, args.trials)
            reference_timing = _measure(reference, args.warmup, args.trials)
            record.update(
                status="ok",
                median_us=fast_timing["median_us"],
                p10_us=fast_timing["p10_us"],
                p90_us=fast_timing["p90_us"],
                reference_median_us=reference_timing["median_us"],
                speedup=reference_timing["median_us"] / fast_timing["median_us"],
                stress_dispatches=args.stress_dispatches,
            )
        except Exception as error:
            record.update(status="error", error=f"{type(error).__name__}: {error}")
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    maple._cuda_profile_cache = base_profile
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    valid = [record for record in records if record.get("status") == "ok"]
    if not valid:
        raise RuntimeError("no elementwise profile passed its correctness gate")
    best = min(valid, key=lambda record: record["median_us"])
    print(f"best: threads={best['threads']} median={best['median_us']:.3f} us")


if __name__ == "__main__":
    main()
