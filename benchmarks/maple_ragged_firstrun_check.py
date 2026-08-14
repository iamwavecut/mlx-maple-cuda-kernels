"""Reproduce the service's first-session order in a clean process.

Sequence under test (the production incident's shape): ghost warmup ->
first real session: 4 admits, the short row finishes early and leaves,
a 5th request joins the freed slot -> every row must equal its solo
stream. Run this in a FRESH process per trial (the incident only ever
fires in the first pool session after a restart).

    python benchmarks/maple_ragged_firstrun_check.py --trial N
"""

import argparse
import json

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import maple
from mlx_lm.models.cache import make_prompt_cache

PROMPTS = [
    "What is the capital of France? One word.",
    "Name the third planet from the sun. One word.",
    "Name the chemical symbol for gold. Just the symbol.",
    "What year did the Berlin Wall fall? Just the year.",
    "Write a Python lambda that doubles a number. Code only.",
]
STEPS = 40


def toks(tok, p):
    return list(tok.apply_chat_template(
        [{"role": "user", "content": p}], add_generation_prompt=True))


def solo_stream(model, ids, steps, eos):
    cache = make_prompt_cache(model)
    out = model(mx.array([ids]), cache=cache)
    mx.eval(out)
    y = mx.argmax(out[:, -1, :], axis=-1, keepdims=True)
    mx.eval(y)
    got = [int(y.item())]
    for _ in range(steps - 1):
        if got[-1] in eos:
            break
        out = model(y, cache=cache)
        y = mx.argmax(out[:, -1, :], axis=-1, keepdims=True)
        mx.eval(y)
        got.append(int(y.item()))
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model-cuda")
    ap.add_argument("--trial", type=int, default=0)
    args = ap.parse_args()
    model, tok = load(
        args.model,
        model_config={"model_file": None, "use_flash_head": False},
        tokenizer_config={"trust_remote_code": False},
        trust_remote_code=False,
    )
    eos = set(tok.eos_token_ids)
    rows = [toks(tok, p) for p in PROMPTS]
    refs = [solo_stream(model, ids, STEPS, eos) for ids in rows]

    # ---- ghost warmup, exactly like the service load ------------------
    for layer in model.model.layers:
        layer.self_attn._ragged_state = None
    ghosts = [make_prompt_cache(model) for _ in range(4)]
    y = mx.array([[0]] * 4)
    for _ in range(3):
        out = maple.ragged_decode_step(model, y, ghosts)
        mx.eval(mx.argmax(out[:, -1, :], axis=-1))
    ghosts = [make_prompt_cache(model) for _ in range(4)]

    # ---- first real session ------------------------------------------
    def admit(ids):
        cache = make_prompt_cache(model)
        n = len(ids)
        for s0 in range(0, n - 1, 2048):
            out = model(mx.array([ids[s0:min(s0 + 2048, n - 1)]]),
                        cache=cache)
            mx.eval(out)
        out = model(mx.array([ids[n - 1:]]), cache=cache)
        mx.eval(out)
        yv = mx.argmax(out[:, -1, :], axis=-1)
        mx.eval(yv)
        return cache, int(yv.item())

    live = []
    for r in range(4):
        cache, first = admit(rows[r])
        live.append({"idx": r, "cache": cache, "stream": [first],
                     "done": first in eos})
    fifth_joined = False
    for step in range(STEPS):
        active = [e for e in live if not e["done"]]
        if not active:
            break
        pad = 4 - len(active)
        y = mx.array([[e["stream"][-1]] for e in active] + [[0]] * pad)
        caches = [e["cache"] for e in active] + ghosts[:pad]
        out = maple.ragged_decode_step(model, y, caches)
        picked = mx.argmax(out[:len(active), -1, :], axis=-1)
        mx.eval(picked)
        for e, v in zip(active, picked.tolist()):
            e["stream"].append(int(v))
            if int(v) in eos:
                e["done"] = True
        if not fifth_joined and any(e["done"] for e in live):
            cache, first = admit(rows[4])
            live.append({"idx": 4, "cache": cache, "stream": [first],
                         "done": first in eos})
            fifth_joined = True

    report = {"trial": args.trial}
    bad = []
    for e in live:
        ref = refs[e["idx"]]
        m = min(len(e["stream"]), len(ref))
        div = next((i for i in range(m)
                    if e["stream"][i] != ref[i]), None)
        if div is not None:
            bad.append({"row": e["idx"], "first_div": div,
                        "got": e["stream"][max(0, div - 1):div + 3],
                        "want": ref[max(0, div - 1):div + 3]})
    report["rows_ok"] = f"{5 - len(bad)}/5"
    report["bad"] = bad
    report["verdict"] = "PASS" if not bad else "FAIL"
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
