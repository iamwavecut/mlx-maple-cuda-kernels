"""Is decode host-bound? Reproduce generate_step's async pattern and split the
step into (a) Python graph construction, (b) submission, (c) waiting for the GPU.

If the GPU wait is near zero, the GPU always finishes before Python has built
the next step, and no kernel optimization can move the wall clock.
"""
import argparse, json, statistics, time
from pathlib import Path
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import maple
from mlx_lm.models.cache import make_prompt_cache

ap = argparse.ArgumentParser()
ap.add_argument("--model", type=Path, required=True)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--warmup", type=int, default=50)
ap.add_argument("--prompt-tokens", type=int, default=128)
ap.add_argument("--label", default="")
ap.add_argument("--mode", choices=["off", "strict", "fast"], default="off")
a = ap.parse_args()

maple._use_fused_add_rms = a.mode != "off"
maple._use_fused_qkv = a.mode != "off"
maple._use_moe_megakernel = a.mode == "fast"

mx.random.seed(20260806)
model, tok, cfg = load(str(a.model), return_config=True,
    model_config={"model_file": None, "use_flash_head": False},
    tokenizer_config={"trust_remote_code": False}, trust_remote_code=False)
vocab = cfg.get("vocab_size") or cfg["text_config"]["vocab_size"]
cache = make_prompt_cache(model)
prompt = mx.random.randint(0, vocab, (1, a.prompt_tokens))
logits = model(prompt, cache=cache)
mx.eval(logits)
mx.synchronize()

def step(y):
    lg = model(y, cache=cache)[:, -1, :]
    return mx.argmax(lg, axis=-1, keepdims=True)

y = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
mx.async_eval(y)
build, submit, wait, total = [], [], [], []
for i in range(a.warmup + a.steps):
    t0 = time.perf_counter()
    ny = step(y)                 # Python: trace the graph for the next token
    t1 = time.perf_counter()
    mx.async_eval(ny)            # hand it to the GPU without blocking
    t2 = time.perf_counter()
    mx.eval(y)                   # block until the *previous* token is done
    t3 = time.perf_counter()
    y = ny
    if i >= a.warmup:
        build.append(t1 - t0); submit.append(t2 - t1)
        wait.append(t3 - t2); total.append(t3 - t0)

med = statistics.median
out = {
  "label": a.label, "steps": a.steps,
  "host_build_ms": med(build) * 1e3,
  "host_submit_ms": med(submit) * 1e3,
  "gpu_wait_ms": med(wait) * 1e3,
  "step_ms": med(total) * 1e3,
  "tps": 1.0 / med(total),
  "host_fraction": (med(build) + med(submit)) / med(total),
}
out["verdict"] = "host-bound" if out["host_fraction"] > 0.7 else (
    "gpu-bound" if out["host_fraction"] < 0.3 else "mixed")
print(json.dumps(out, indent=1), flush=True)
