"""Speculation v0: the acceptance harness on single-step verify.

Prompt-lookup drafts L tokens; verification runs them through ordinary
sequential fused steps (bit-trivially equal to plain decode); a mismatch
rolls the caches back with _attn_mega_rollback and the corrected token
is emitted. No new kernels — this proves the DRAFT/ACCEPT/ROLLBACK
plumbing end to end and measures live acceptance. The M=L kernels then
drop into this harness for the actual speedup.

    python benchmarks/maple_spec_loop_proto.py --model <path>
"""
import argparse
import json
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import maple
from mlx_lm.models.cache import make_prompt_cache

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--gens", type=int, default=96)
ap.add_argument("--L", type=int, default=8)
ap.add_argument("--k", type=int, default=2)
args = ap.parse_args()

model, tok, cfg = load(args.model, return_config=True,
    model_config={"model_file": None, "use_flash_head": False},
    tokenizer_config={"trust_remote_code": False}, trust_remote_code=False)
inner = model.model

PROMPTS = [
    "Rewrite this function to handle status 'urgent' too:\n"
    "def f(orders):\n    return [o for o in orders if o['status'] == 'pending']",
    "Convert each line to JSON with ts and level fields:\n"
    "2026-08-13T10:02:11 ERROR timeout\n2026-08-13T10:02:14 WARN retry",
    "Explain briefly why rivers meander.",
]


def toks_for(p):
    return list(tok.apply_chat_template(
        [{"role": "user", "content": p}], add_generation_prompt=True))


def reset():
    for l in inner.layers:
        if hasattr(l.self_attn, "_mega_state"):
            del l.self_attn._mega_state


def draft(ctx, k, L):
    if len(ctx) < k + 1:
        return []
    key = tuple(ctx[-k:])
    for s in range(len(ctx) - k - 1, -1, -1):
        if tuple(ctx[s:s + k]) == key:
            return list(ctx[s + k:s + k + L])
    return []


def plain_decode(prompt_ids, gens):
    reset()
    cache = make_prompt_cache(model)
    out = model(mx.array([prompt_ids]), cache=cache); mx.eval(out)
    y = mx.argmax(out[:, -1, :], axis=-1, keepdims=True); mx.eval(y)
    toks = [int(y.item())]
    for _ in range(gens - 1):
        out = model(y, cache=cache); mx.eval(out)
        y = mx.argmax(out[:, -1, :], axis=-1, keepdims=True); mx.eval(y)
        toks.append(int(y.item()))
    return toks


def rollback_all(cache, to_offset):
    ok = True
    for layer, c in zip(inner.layers, cache):
        st = getattr(layer.self_attn, "_mega_state", None)
        if st is not None and st.bound_to(c) and st.synced_offset >= 0:
            if not maple._attn_mega_rollback(layer.self_attn, c, to_offset):
                ok = False
        else:
            # stock-managed layer: counters only
            c.offset = to_offset
            if hasattr(c, "_idx"):
                c._idx = to_offset
    return ok


def spec_decode(prompt_ids, gens, k, L):
    reset()
    cache = make_prompt_cache(model)
    out = model(mx.array([prompt_ids]), cache=cache); mx.eval(out)
    y = mx.argmax(out[:, -1, :], axis=-1, keepdims=True); mx.eval(y)
    ctx = list(prompt_ids) + [int(y.item())]
    toks = [int(y.item())]
    passes = 0
    accepted_total = 0
    while len(toks) < gens:
        d = draft(ctx, k, L)
        passes += 1
        if not d:
            out = model(y, cache=cache); mx.eval(out)
            y = mx.argmax(out[:, -1, :], axis=-1, keepdims=True); mx.eval(y)
            t = int(y.item()); toks.append(t); ctx.append(t)
            continue
        base_off = cache[0].offset
        emitted = []
        cur = y
        ok_prefix = 0
        for di, dtok in enumerate(d):
            out = model(cur, cache=cache); mx.eval(out)
            cur = mx.argmax(out[:, -1, :], axis=-1, keepdims=True)
            mx.eval(cur)
            t = int(cur.item())
            emitted.append(t)
            if t == dtok:
                ok_prefix += 1
            else:
                break
        accepted_total += ok_prefix
        keep = ok_prefix + 1                      # accepted + correction
        keep = min(keep, len(emitted))
        if keep < len(emitted) or len(emitted) < len(d):
            pass
        # rewind cache to base + keep steps
        target = base_off + keep
        if cache[0].offset != target:
            if not rollback_all(cache, target):
                return None, None, None
        take = emitted[:keep]
        toks.extend(take)
        ctx.extend(take)
        if len(toks) > gens:
            del toks[gens:]
        y = mx.array([[toks[-1]]])
        mx.eval(y)
    return toks, passes, accepted_total


report = {}
for pi, prompt in enumerate(PROMPTS):
    ids = toks_for(prompt)
    ref = plain_decode(ids, args.gens)
    t0 = time.perf_counter()
    got, passes, acc = spec_decode(ids, args.gens, args.k, args.L)
    dt = time.perf_counter() - t0
    if got is None:
        report[f"p{pi}"] = {"error": "rollback refused"}
        continue
    m = next((i for i, (a, b) in enumerate(zip(ref, got)) if a != b), None)
    report[f"p{pi}"] = {
        "identical": m is None, "first_div": m,
        "passes": passes, "accepted": acc,
        "tokens_per_pass": round(len(got) / max(passes, 1), 2),
    }
    print(json.dumps({f"p{pi}": report[f"p{pi}"]}), flush=True)
print(json.dumps(report), flush=True)
