"""Gate for pool-composition changes on the ragged lane.

The serving pool shrinks mid-flight: a finished row leaves, the row
list compacts, and every surviving row shifts one plane left. This
must not move anyone's bits. Scenario (the canary's shape):

  1. Prefill A, B, C at different prompt lengths; decode K1 steps
     ragged as [A, B, C].
  2. B finishes: materialize B's plane into its cache (the pool's
     finalize discipline), drop it, decode K2 steps as [A, C].
  3. A and C must equal their solo streams for all K1+K2 steps, and
     B must equal solo for its K1 steps AND continue solo bit-equal
     from its materialized cache afterward (the LRU reuse path).

    python benchmarks/maple_ragged_leave_check.py
"""

import argparse
import json

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import maple
from mlx_lm.models.cache import make_prompt_cache

PROMPTS = [
    "Write a haiku about a maple grove.",
    "List three uses for a paperclip.",
    "Explain what a mutex is in one sentence, then give a tiny example "
    "in Python with a short explanation of each line.",
]


def toks(tok, p):
    return list(tok.apply_chat_template(
        [{"role": "user", "content": p}], add_generation_prompt=True))


def solo_stream(model, ids, steps):
    cache = make_prompt_cache(model)
    out = model(mx.array([ids]), cache=cache)
    mx.eval(out)
    y = mx.argmax(out[:, -1, :], axis=-1, keepdims=True)
    mx.eval(y)
    got = [int(y.item())]
    for _ in range(steps - 1):
        out = model(y, cache=cache)
        y = mx.argmax(out[:, -1, :], axis=-1, keepdims=True)
        mx.eval(y)
        got.append(int(y.item()))
    return got


def materialize_pool_row(model, row_cache):
    """The pool's finalize discipline, verbatim."""
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
    ap.add_argument("--k1", type=int, default=12)
    ap.add_argument("--k2", type=int, default=12)
    args = ap.parse_args()
    K1, K2 = args.k1, args.k2

    model, tok = load(
        args.model,
        model_config={"model_file": None, "use_flash_head": False},
        tokenizer_config={"trust_remote_code": False},
        trust_remote_code=False,
    )
    rows = [toks(tok, p) for p in PROMPTS]
    refs = [solo_stream(model, ids, K1 + K2) for ids in rows]

    for layer in model.model.layers:
        layer.self_attn._ragged_state = None
    caches, first = [], []
    for ids in rows:
        cache = make_prompt_cache(model)
        out = model(mx.array([ids]), cache=cache)
        mx.eval(out)
        y = mx.argmax(out[:, -1, :], axis=-1, keepdims=True)
        mx.eval(y)
        first.append(int(y.item()))
        caches.append(cache)

    streams = [[v] for v in first]
    y = mx.array([[v] for v in first])
    for _ in range(K1 - 1):
        out = maple.ragged_decode_step(model, y, caches)
        y = mx.argmax(out[:, -1, :], axis=-1, keepdims=True)
        mx.eval(y)
        for rr, v in enumerate(y[:, 0].tolist()):
            streams[rr].append(int(v))

    # --- B (row 1) leaves: materialize, drop, continue [A, C] -----------
    materialize_pool_row(model, caches[1])
    b_cache = caches[1]
    b_stream_k1 = list(streams[1])
    caches = [caches[0], caches[2]]
    y = mx.array([[streams[0][-1]], [streams[2][-1]]])
    for _ in range(K2):
        out = maple.ragged_decode_step(model, y, caches)
        y = mx.argmax(out[:, -1, :], axis=-1, keepdims=True)
        mx.eval(y)
        streams[0].append(int(y[0, 0].item()))
        streams[2].append(int(y[1, 0].item()))

    a_ok = streams[0] == refs[0][: len(streams[0])]
    c_ok = streams[2] == refs[2][: len(streams[2])]
    b_k1_ok = b_stream_k1 == refs[1][:K1]

    # --- B resumes solo from its materialized cache (the LRU path) ------
    yb = mx.array([[b_stream_k1[-1]]])
    b_resumed = list(b_stream_k1)
    for _ in range(K2):
        out = model(yb, cache=b_cache)
        yb = mx.argmax(out[:, -1, :], axis=-1, keepdims=True)
        mx.eval(yb)
        b_resumed.append(int(yb.item()))
    b_resume_ok = b_resumed == refs[1][: len(b_resumed)]

    report = {
        "A_rows_equal_solo": a_ok,
        "C_rows_equal_solo": c_ok,
        "B_k1_equal_solo": b_k1_ok,
        "B_resume_from_materialized": b_resume_ok,
        "verdict": "PASS" if all(
            [a_ok, c_ok, b_k1_ok, b_resume_ok]) else "FAIL",
    }
    if not a_ok:
        report["A_first_div"] = next(
            (i for i, (x, z) in enumerate(zip(streams[0], refs[0]))
             if x != z), None)
    if not c_ok:
        report["C_first_div"] = next(
            (i for i, (x, z) in enumerate(zip(streams[2], refs[2]))
             if x != z), None)
    if not b_resume_ok:
        report["B_first_div"] = next(
            (i for i, (x, z) in enumerate(zip(b_resumed, refs[1]))
             if x != z), None)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
