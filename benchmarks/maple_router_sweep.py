# Copyright © 2026 DeepGrove AI.

"""Sweep CUDA router launch profiles with correctness and stress gates."""

import argparse
import json
from pathlib import Path

import mlx.core as mx

from mlx_lm.models import maple

from maple_kernel_benchmark import (
    _environment,
    _evaluate,
    _measure,
    _router_case,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--rows-per-warp", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--stress-dispatches", type=int, default=128)
    args = parser.parse_args()

    if maple._kernel_backend() != "cuda" or maple._cuda_profile() is None:
        raise RuntimeError("router profile sweep requires a supported CUDA device")
    base_profile = maple._cuda_profile()
    records = [_environment()]
    rows_values = [1] if base_profile.router_reference_gemv else args.rows_per_warp
    for threads in args.threads:
        for rows_per_warp in rows_values:
            record = {
                "type": "router_sweep",
                "threads": threads,
                "rows_per_warp": rows_per_warp,
            }
            try:
                maple._cuda_profile_cache = maple._CudaProfile(
                    f"{base_profile.name}_t{threads}_r{rows_per_warp}",
                    elementwise_threads=base_profile.elementwise_threads,
                    router_threads=threads,
                    router_rows_per_warp=rows_per_warp,
                    router_reference_gemv=base_profile.router_reference_gemv,
                )
                mx.random.seed(20260806)
                fast, reference = _router_case()
                for dispatch in range(args.stress_dispatches):
                    indices, scores = _evaluate(fast)
                    ref_indices, ref_scores = _evaluate(reference)
                    if not bool(mx.all((indices >= 0) & (indices < 256))):
                        raise RuntimeError(f"dispatch {dispatch}: invalid expert index")
                    if not bool(
                        mx.allclose(
                            mx.sort(scores),
                            mx.sort(ref_scores),
                            rtol=1e-5,
                            atol=1e-5,
                        )
                    ):
                        raise RuntimeError(f"dispatch {dispatch}: score mismatch")
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
        raise RuntimeError("no router profile passed its correctness gate")
    best = min(valid, key=lambda record: record["median_us"])
    print(
        f"best: threads={best['threads']} rows={best['rows_per_warp']} "
        f"median={best['median_us']:.3f} us"
    )


if __name__ == "__main__":
    main()
