"""E2E gate for ragged batched decode (requests at different offsets).

B prompts of DIFFERENT lengths, each prefilled solo into its own cache;
then ONE ragged loop decodes all of them per step via
`maple.ragged_decode_step`. Every row must emit exactly its solo stream.
Also reports aggregate tok/s vs serving them sequentially.

    python benchmarks/maple_ragged_e2e_check.py --steps 48
"""

import argparse
import json
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import maple
from mlx_lm.models.cache import make_prompt_cache

PROMPTS = [
    "Write a haiku about a maple grove.",
    "Explain what a mutex is in one sentence, then give a tiny example "
    "in Python with a short explanation of each line.",
    "List three uses for a paperclip.",
    "Describe rain to someone who has never seen it, in two or three "
    "sentences that avoid the word water entirely if you can manage.",
    "Name a prime number between 90 and 100.",
    "Summarize photosynthesis in one line.",
    "Give one tip for writing readable code and one for naming things.",
    "What is the capital of Portugal? Answer briefly.",
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


def sequential_walltime(model, rows, steps):
    t0 = time.perf_counter()
    for ids in rows:
        solo_stream(model, ids, steps)
    dt = time.perf_counter() - t0
    return round(len(rows) * steps / dt, 1)


def ragged_run(model, rows, steps, warmup=4):
    caches, first = [], []
    for ids in rows:
        cache = make_prompt_cache(model)
        out = model(mx.array([ids]), cache=cache)
        mx.eval(out)
        y = mx.argmax(out[:, -1, :], axis=-1, keepdims=True)
        mx.eval(y)
        first.append(int(y.item()))
        caches.append(cache)
    for l in model.model.layers:
        if getattr(l.self_attn, "_ragged_state", None) is not None:
            l.self_attn._ragged_state = None
    y = mx.array([[v] for v in first])
    streams = [[v] for v in first]
    t0 = None
    for step in range(steps - 1):
        if step == warmup:
            t0 = time.perf_counter()
        out = maple.ragged_decode_step(model, y, caches)
        y = mx.argmax(out[:, -1, :], axis=-1, keepdims=True)
        mx.eval(y)
        for rr, v in enumerate(y[:, 0].tolist()):
            streams[rr].append(int(v))
    dt = time.perf_counter() - t0
    tps = len(rows) * (steps - 1 - warmup) / dt
    return streams, round(tps, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model-cuda")
    ap.add_argument("--steps", type=int, default=48)
    args = ap.parse_args()

    model, tok = load(
        args.model,
        model_config={"model_file": None, "use_flash_head": False},
        tokenizer_config={"trust_remote_code": False},
        trust_remote_code=False,
    )

    report = {}
    all_ok = True
    for B in (2, 4, 8):
        rows = [toks(tok, p) for p in PROMPTS[:B]]
        lens = sorted({len(x) for x in rows})
        refs = [solo_stream(model, ids, args.steps) for ids in rows]
        got, tps = ragged_run(model, rows, args.steps)
        row_ok = [got[rr] == refs[rr] for rr in range(B)]
        all_ok = all_ok and all(row_ok)
        seq_tps = sequential_walltime(model, rows, args.steps)
        report[f"B{B}"] = {
            "distinct_prompt_lens": lens,
            "rows_equal_solo": f"{sum(row_ok)}/{B}",
            "ragged_tps": tps,
            "sequential_tps": seq_tps,
        }
    report["verdict"] = "PASS" if all_ok else "FAIL"
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
