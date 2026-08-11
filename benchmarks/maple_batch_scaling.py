"""Aggregate throughput versus batch size.

Decode is host-bound: the step cost is the sum of per-operation host costs, and
the operation count does not depend on how many sequences ride through the
layer.  If that holds on the batch axis the way it partly held on the sequence
axis, serving several streams at once is nearly free and multiplies aggregate
throughput without touching latency-per-step much.
"""
import json, statistics, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

model, tok, cfg = load("model-cuda", return_config=True,
    model_config={"model_file": None, "use_flash_head": False},
    tokenizer_config={"trust_remote_code": False}, trust_remote_code=False)
vocab = cfg.get("vocab_size") or cfg["text_config"]["vocab_size"]

for B in (1, 2, 4, 8, 16, 32):
    try:
        cache = make_prompt_cache(model)
        mx.random.seed(20260806)
        prompt = mx.random.randint(0, vocab, (B, 128))
        lg = model(prompt, cache=cache)
        mx.eval(lg); mx.synchronize()
        y = mx.argmax(lg[:, -1, :], axis=-1, keepdims=True)
        mx.async_eval(y)
        build, total = [], []
        WARM, STEPS = 20, 60
        for i in range(WARM + STEPS):
            t0 = time.perf_counter()
            ny = mx.argmax(model(y, cache=cache)[:, -1, :], axis=-1, keepdims=True)
            t1 = time.perf_counter()
            mx.async_eval(ny)
            mx.eval(y)
            t2 = time.perf_counter()
            y = ny
            if i >= WARM:
                build.append(t1 - t0); total.append(t2 - t0)
        md = statistics.median
        step = md(total)
        print(json.dumps({
            "batch": B,
            "host_build_ms": round(md(build) * 1e3, 4),
            "step_ms": round(step * 1e3, 4),
            "tps_per_stream": round(1 / step, 1),
            "aggregate_tps": round(B / step, 1),
        }), flush=True)
        del cache
    except Exception as e:
        print(json.dumps({"batch": B, "error": str(e)[:160]}), flush=True)
