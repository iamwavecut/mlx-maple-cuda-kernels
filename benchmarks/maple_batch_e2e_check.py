"""E2E gate for the batch decode lane (MAPLE_BATCH_MEGAKERNELS=1).

The lane's promise: batched decode of B independent prompts emits, per
row, EXACTLY the token stream a solo run of that prompt emits. So the
reference is B solo runs (stock path, lanes off), and the candidate is
one batched run with the lane on -- token IDs compared per row. Run
twice, lane off/on, in ONE process (module flags flipped in place).

    python benchmarks/maple_batch_e2e_check.py --steps 48
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
    "Explain what a mutex is in one sentence.",
    "List three uses for a paperclip.",
    "What is the capital of Portugal? Answer briefly.",
    "Give one tip for writing readable code.",
    "Describe rain to someone who has never seen it.",
    "Name a prime number between 90 and 100.",
    "Summarize photosynthesis in one line.",
]


def toks(tok, p):
    return tok.apply_chat_template(
        [{"role": "user", "content": p}], add_generation_prompt=True)


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


def merged_solo_prefill(model, ids_rows):
    """Prefill each row SOLO (bit-equal to its solo run by construction),
    then assemble one batch cache: concat per-layer K/V on the batch axis.
    Equal prompt lengths keep offsets/_idx shared."""
    from mlx_lm.models.cache import KVCache, RotatingKVCache
    solo_caches, first_ys = [], []
    for ids in ids_rows:
        cache = make_prompt_cache(model)
        out = model(mx.array([ids]), cache=cache)
        mx.eval(out)
        y = mx.argmax(out[:, -1, :], axis=-1, keepdims=True)
        mx.eval(y)
        first_ys.append(int(y.item()))
        solo_caches.append(cache)
    merged = make_prompt_cache(model)
    for li, mc in enumerate(merged):
        parts = [sc[li] for sc in solo_caches]
        keys = mx.concatenate(
            [mx.contiguous(p.keys) for p in parts], axis=0)
        values = mx.concatenate(
            [mx.contiguous(p.values) for p in parts], axis=0)
        mx.eval(keys, values)
        mc.keys, mc.values = keys, values
        mc.offset = parts[0].offset
        if isinstance(mc, RotatingKVCache):
            mc._idx = parts[0]._idx
    return merged, first_ys


def batch_stream(model, ids_rows, steps, warmup=3):
    cache, first = merged_solo_prefill(model, ids_rows)
    y = mx.array([[v] for v in first])
    got = [[v] for v in first]
    t0 = None
    for step in range(steps - 1):
        if step == warmup:
            mx.eval(y)
            t0 = time.perf_counter()
        out = model(y, cache=cache)
        y = mx.argmax(out[:, -1, :], axis=-1, keepdims=True)
        mx.eval(y)
        for rr, v in enumerate(y[:, 0].tolist()):
            got[rr].append(int(v))
    dt = time.perf_counter() - t0
    tps = len(ids_rows) * (steps - 1 - warmup) / dt
    return got, round(tps, 1)


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
        cut = min(len(x) for x in rows)
        rows = [list(x)[:cut] for x in rows]

        maple._use_batch_megakernels = False
        refs = [solo_stream(model, ids, args.steps) for ids in rows]

        maple._use_batch_megakernels = False
        stock_b, stock_tps = batch_stream(model, rows, args.steps)

        maple._use_batch_megakernels = True
        got, mega_tps = batch_stream(model, rows, args.steps)
        maple._use_batch_megakernels = False

        row_ok = [got[rr] == refs[rr] for rr in range(B)]
        stock_row_ok = [stock_b[rr] == refs[rr] for rr in range(B)]
        all_ok = all_ok and all(row_ok)
        report[f"B{B}"] = {
            "mega_rows_equal_solo": f"{sum(row_ok)}/{B}",
            "stock_batch_rows_equal_solo": f"{sum(stock_row_ok)}/{B}",
            "mega_tps": mega_tps, "stock_tps": stock_tps,
        }

    report["verdict"] = "PASS" if all_ok else "FAIL"
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
