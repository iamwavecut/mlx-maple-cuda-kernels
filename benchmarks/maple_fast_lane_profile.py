"""Exclusive host time per sub-block of a decode step, fast lane edition.

The 0.4.0 breakdown (attention 57.9%) was measured before the megakernel
became the default, so it no longer describes the shipped configuration.
This probe times the fast lane's remaining Python: the fused attention
sub-steps, the megakernel dispatch itself, the lm_head and the sampling tail.

Timers wrap the *construction* of each sub-graph; the step is evaluated once
at the end, so the split is host time, which is the currency that matters for
a host-bound decode.

    python benchmarks/maple_fast_lane_profile.py --model model-cuda
"""

import argparse
import json
import statistics
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import maple
from mlx_lm.models.cache import make_prompt_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--warm", type=int, default=30)
    args = ap.parse_args()

    model, _, cfg = load(
        args.model,
        return_config=True,
        model_config={"model_file": None, "use_flash_head": False},
        tokenizer_config={"trust_remote_code": False},
        trust_remote_code=False,
    )
    vocab = cfg.get("vocab_size") or cfg["text_config"]["vocab_size"]
    inner = model.model

    buckets = {}

    def timed(name, fn):
        def wrapper(*a, **k):
            t0 = time.perf_counter()
            out = fn(*a, **k)
            buckets[name] = buckets.get(name, 0.0) + (time.perf_counter() - t0)
            return out
        return wrapper

    # Attention is invoked through the class, so instance-level wrapping does
    # not intercept it; patch the class once.  Cache updates run *inside*
    # attention, so that bucket is nested and reported separately below.
    maple.MapleAttention.__call__ = timed(
        "attention_incl_kv", maple.MapleAttention.__call__
    )
    maple._moe_megakernel_call = timed("megakernel", maple._moe_megakernel_call)

    real_decode = inner._decode_fused

    def decode_timed(h, cache, full_mask, swa_mask, fuse):
        return real_decode(h, cache, full_mask, swa_mask, timed("fuse", fuse))

    inner._decode_fused = decode_timed

    mx.random.seed(20260806)
    cache = make_prompt_cache(model)
    for c in cache:
        if hasattr(c, "update_and_fetch"):
            c.update_and_fetch = timed("kv_cache", c.update_and_fetch)
    mx.eval(model(mx.random.randint(0, vocab, (1, 128)), cache=cache))
    mx.synchronize()

    lm_head_t = {"t": 0.0}
    y = mx.random.randint(0, vocab, (1, 1))
    mx.eval(y)
    pending = None
    totals = []
    for i in range(args.warm + args.steps):
        if i == args.warm:
            for k in list(buckets):
                buckets[k] = 0.0
            lm_head_t["t"] = 0.0
            totals.clear()
        t0 = time.perf_counter()
        logits = model(y, cache=cache)
        t1 = time.perf_counter()
        nxt = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.async_eval(nxt)
        if pending is not None:
            mx.eval(pending)
        pending = nxt
        y = nxt
        totals.append(time.perf_counter() - t0)
        lm_head_t["t"] += time.perf_counter() - t1
    mx.eval(pending)
    mx.synchronize()

    n = args.steps
    total = sum(totals)
    kv = buckets.get("kv_cache", 0.0)
    attn = buckets.get("attention_incl_kv", 0.0)
    flat = {
        "attention_excl_kv": attn - kv,
        "kv_cache": kv,
        "megakernel": buckets.get("megakernel", 0.0),
        "fuse": buckets.get("fuse", 0.0),
        "sample_tail": lm_head_t["t"],
    }
    flat["other"] = total - sum(flat.values())
    report = {
        "steps": n,
        "step_ms": round(statistics.median(totals) * 1e3, 4),
        "us_per_step": {k: round(v / n * 1e6, 1) for k, v in flat.items()},
        "share_percent": {
            k: round(v / total * 100.0, 1) for k, v in flat.items()
        },
    }
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
