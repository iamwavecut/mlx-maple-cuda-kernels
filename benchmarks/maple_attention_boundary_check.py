"""Bit identity of the attention lane across the kL=1024 boundary.

Five cases against the pure stock stream (greedy, argmax-stable region):

  A_cross_1024        prefill ~1000, decode 60   -- crosses 1024 mid-decode
                      (full-layer cap grows 1024 -> 2048 in flight)
  B_start_past_1024   prefill  1500, decode 40   -- first fused decode is
                      already in the kernel's 2-pass branch
  C_writeback_regrow  prefill ~1000, decode 40, prefill 200, decode 40
                      -- write-back past the boundary, stock prefill on a
                      rotated ring (concat-tail state), fused re-entry
  D_start_at_4096     prefill  3800, decode 40   -- lands on the 4096 tier
  E_grow_2048_4096    prefill  1900, decode 260  -- crosses 2048 in flight

    python benchmarks/maple_attention_boundary_check.py --model <path>
"""

import argparse
import json

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import maple
from mlx_lm.models.cache import make_prompt_cache

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--long", action="store_true",
                help="include the slow D/E growth cases")
args = ap.parse_args()

MODEL = args.model
model, tok, cfg = load(MODEL, return_config=True,
    model_config={"model_file": None, "use_flash_head": False},
    tokenizer_config={"trust_remote_code": False}, trust_remote_code=False)
inner = model.model
vocab = cfg.get("vocab_size") or cfg["text_config"]["vocab_size"]


def run(attn_on, seed, plen, steps, plen2=0, steps2=0):
    maple._use_attention_megakernel = False
    for l in inner.layers:
        if hasattr(l.self_attn, "_mega_state"):
            del l.self_attn._mega_state
    mx.random.seed(seed)
    p1 = mx.random.randint(0, vocab, (1, plen))
    p2 = mx.random.randint(0, vocab, (1, plen2)) if plen2 else None
    mx.eval(p1) if p2 is None else mx.eval(p1, p2)
    cache = make_prompt_cache(model)
    out = model(p1, cache=cache); mx.eval(out)
    y = mx.argmax(out[:, -1, :], axis=-1, keepdims=True); mx.eval(y)
    out = model(y, cache=cache); mx.eval(out)  # warm probes stock
    y = mx.argmax(out[:, -1, :], axis=-1, keepdims=True); mx.eval(y)
    maple._use_attention_megakernel = attn_on
    toks = []
    for _ in range(steps):
        out = model(y, cache=cache); mx.eval(out)
        y = mx.argmax(out[:, -1, :], axis=-1, keepdims=True); mx.eval(y)
        toks.append(int(y.item()))
    if p2 is not None:
        out = model(p2, cache=cache); mx.eval(out)
        y = mx.argmax(out[:, -1, :], axis=-1, keepdims=True); mx.eval(y)
        toks.append(int(y.item()))
        for _ in range(steps2):
            out = model(y, cache=cache); mx.eval(out)
            y = mx.argmax(out[:, -1, :], axis=-1, keepdims=True); mx.eval(y)
            toks.append(int(y.item()))
    return toks


CASES = {
    "A_cross_1024": dict(plen=1000, steps=60),
    "B_start_past_1024": dict(plen=1500, steps=40),
    "C_writeback_regrow": dict(plen=1000, steps=40, plen2=200, steps2=40),
}


def run_fresh_cache_pair(attn_on, seed):
    """Two independent requests on one model: fresh caches, live state.

    This is the service pattern that leaked one user's context into the
    next answer: the megakernel state outlives the per-request cache
    object, and an unbound write-back used to inject the previous
    request's KV history into the new empty cache.
    """
    maple._use_attention_megakernel = attn_on
    for l in inner.layers:
        if hasattr(l.self_attn, "_mega_state"):
            del l.self_attn._mega_state
    outs = []
    for r in range(2):
        mx.random.seed(seed + 17 * r)
        p = mx.random.randint(0, vocab, (1, 96 + 40 * r))
        mx.eval(p)
        cache = make_prompt_cache(model)
        out = model(p, cache=cache); mx.eval(out)
        y = mx.argmax(out[:, -1, :], axis=-1, keepdims=True); mx.eval(y)
        toks = [int(y.item())]
        for _ in range(24):
            out = model(y, cache=cache); mx.eval(out)
            y = mx.argmax(out[:, -1, :], axis=-1, keepdims=True); mx.eval(y)
            toks.append(int(y.item()))
        outs.append(toks)
    return outs
if args.long:
    CASES["D_start_at_4096"] = dict(plen=3800, steps=40)
    CASES["E_grow_2048_4096"] = dict(plen=1900, steps=260)
rep = {}
ref_pair = run_fresh_cache_pair(False, 8800)
got_pair = run_fresh_cache_pair(True, 8800)
rep["F_fresh_cache_reuse"] = {
    "request2_identical": ref_pair[1] == got_pair[1],
    "request1_identical": ref_pair[0] == got_pair[0],
}
for name, kw in CASES.items():
    seed = 4200 + len(name)
    ref = run(False, seed, **kw)
    ref2 = run(False, seed, **kw)
    stable = next(
        (j for j, (a, b) in enumerate(zip(ref, ref2)) if a != b), len(ref))
    got = run(True, seed, **kw)
    m = next((j for j in range(stable) if got[j] != ref[j]), None)
    rep[name] = {"stable": stable, "identical": m is None, "first_div": m}
print(json.dumps(rep), flush=True)
