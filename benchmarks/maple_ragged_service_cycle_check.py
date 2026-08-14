"""The canary's exact shape on the testbench: solo round (B=1 lane,
LRU inserts) -> pooled round over LRU-hit caches -> byte compare.

Covers what the isolated gates miss: the B=1 mega-state and the ragged
state touching the SAME caches through the LRU, exact-hit re-admission
(trim+tail-token prefill through the B=1 lane), and a mid-pool join.

    python benchmarks/maple_ragged_service_cycle_check.py
"""

import argparse
import json

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import maple
from mlx_lm.models.cache import (LRUPromptCache, make_prompt_cache,
                                 trim_prompt_cache)

PROMPTS = [
    "Write a haiku about a maple grove.",
    "Explain what a mutex is in one sentence.",
    "List three uses for a paperclip.",
    "What is the capital of Portugal? Answer briefly.",
]
STEPS = 48


def toks(tok, p):
    return list(tok.apply_chat_template(
        [{"role": "user", "content": p}], add_generation_prompt=True))


def solo_serve(model, lru, ids, steps):
    """The service's solo generate: LRU fetch, prefill, decode, insert."""
    cache, rest = lru.fetch_nearest_cache(id(model), ids)
    if cache is None:
        cache = make_prompt_cache(model)
        rest = ids
    elif not rest:
        trim_prompt_cache(cache, 1)
        rest = ids[-1:]
    out = model(mx.array([rest]), cache=cache)
    mx.eval(out)
    y = mx.argmax(out[:, -1, :], axis=-1, keepdims=True)
    mx.eval(y)
    got = [int(y.item())]
    for _ in range(steps - 1):
        out = model(y, cache=cache)
        y = mx.argmax(out[:, -1, :], axis=-1, keepdims=True)
        mx.eval(y)
        got.append(int(y.item()))
    lru.insert_cache(id(model), ids + got, cache)
    return got


def pool_admit(model, lru, ids):
    cache, rest = lru.fetch_nearest_cache(id(model), ids)
    kind = "cold"
    if cache is None:
        cache = make_prompt_cache(model)
        rest = ids
    elif not rest:
        trim_prompt_cache(cache, 1)
        rest = ids[-1:]
        kind = "exact"
    else:
        kind = "partial"
    out = model(mx.array([rest]), cache=cache)
    mx.eval(out)
    y = mx.argmax(out[:, -1, :], axis=-1, keepdims=True)
    mx.eval(y)
    return cache, int(y.item()), kind


def materialize_row(model, row_cache):
    for layer in model.model.layers:
        state = getattr(layer.self_attn, "_ragged_state", None)
        if state is None:
            continue
        for r in range(state.rows):
            ref = state.cache_refs[r]
            if ref is not None and any(ref() is c for c in row_cache):
                state.materialize_row(r)
                state.synced[r] = -1
                state.cache_refs[r] = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model-cuda")
    args = ap.parse_args()
    model, tok = load(
        args.model,
        model_config={"model_file": None, "use_flash_head": False},
        tokenizer_config={"trust_remote_code": False},
        trust_remote_code=False,
    )
    rows = [toks(tok, p) for p in PROMPTS]
    lru = LRUPromptCache(max_size=8)

    # solo baseline round -- the canary's round 1, B=1 lane + LRU inserts
    solos = [solo_serve(model, lru, ids, STEPS) for ids in rows]

    # pooled round -- admits over LRU hits, then one ragged loop; row 3
    # joins two steps late (the mid-pool join path)
    for layer in model.model.layers:
        layer.self_attn._ragged_state = None
    admits = [pool_admit(model, lru, ids) for ids in rows[:3]]
    kinds = [a[2] for a in admits]
    caches = [a[0] for a in admits]
    streams = [[a[1]] for a in admits]
    y = mx.array([[a[1]] for a in admits])
    for step in range(STEPS - 1):
        if step == 2:
            cache_d, first_d, kind_d = pool_admit(model, lru, rows[3])
            kinds.append(kind_d)
            caches.append(cache_d)
            streams.append([first_d])
            y = mx.concatenate([y, mx.array([[first_d]])], axis=0)
        out = maple.ragged_decode_step(model, y, caches)
        y = mx.argmax(out[:, -1, :], axis=-1, keepdims=True)
        mx.eval(y)
        for rr, v in enumerate(y[:, 0].tolist()):
            streams[rr].append(int(v))
    for rr in range(len(caches)):
        materialize_row(model, caches[rr])
        lru.insert_cache(id(model), rows[rr] + streams[rr], caches[rr])

    matches = [streams[rr] == solos[rr][: len(streams[rr])]
               for rr in range(4)]
    report = {
        "admit_kinds": kinds,
        "rows_equal_solo": f"{sum(matches)}/4",
        "per_row": matches,
        "verdict": "PASS" if all(matches) else "FAIL",
    }
    for rr, ok in enumerate(matches):
        if not ok:
            report[f"row{rr}_first_div"] = next(
                (i for i, (x, z) in enumerate(zip(streams[rr], solos[rr]))
                 if x != z), None)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
