#!/usr/bin/env python3
"""Stress same-gate CUDA router calls in one lazy graph."""

import argparse
import hashlib
import inspect
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import maple


def check_pair(got, reference, dispatch):
    inds, scores = got
    ref_inds, ref_scores = reference
    if not bool(mx.array_equal(inds, ref_inds)):
        raise RuntimeError(f"dispatch {dispatch}: expert order mismatch")
    if not bool(mx.allclose(scores, ref_scores, rtol=1e-5, atol=1e-5)):
        raise RuntimeError(f"dispatch {dispatch}: score mismatch")


def batch(gate, inputs, validate):
    fast = [gate._fused_call(x) for x in inputs]
    refs = [gate._reference(x) for x in inputs] if validate else []
    mx.eval(*(a for pair in fast for a in pair), *(a for pair in refs for a in pair))
    if validate:
        for i, pair in enumerate(zip(fast, refs)):
            check_pair(*pair, i)
    return fast


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dispatches", type=int, default=32)
    p.add_argument("--warmups", type=int, default=10)
    p.add_argument("--trials", type=int, default=50)
    args = p.parse_args()

    mx.random.seed(20260806)
    model = load(
        str(args.model), tokenizer_config={"trust_remote_code": True},
        trust_remote_code=True,
    )[0]
    gate = model.model.layers[0].mlp.gate
    inputs = [
        (mx.random.normal((1, 1, gate.hidden_size)) * (0.1 + i / 37.0)).astype(
            mx.bfloat16
        )
        for i in range(args.dispatches)
    ]
    mx.eval(*inputs)
    batch(gate, inputs, True)
    for _ in range(args.warmups):
        batch(gate, inputs, False)
    mx.synchronize()
    samples = []
    for _ in range(args.trials):
        tic = time.perf_counter_ns()
        batch(gate, inputs, True)
        mx.synchronize()
        samples.append((time.perf_counter_ns() - tic) / 1000.0)

    source = Path(inspect.getfile(maple))
    record = {
        "type": "summary", "dispatches": args.dispatches,
        "trials": args.trials, "all_exact_indices": True,
        "all_scores_close": True, "median_batch_us": statistics.median(samples),
        "mean_batch_us": statistics.fmean(samples),
        "p10_batch_us": sorted(samples)[int(0.1 * len(samples))],
        "p90_batch_us": sorted(samples)[min(len(samples) - 1, int(0.9 * len(samples)))],
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "device": dict(mx.device_info(mx.gpu)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
