"""Isolation gate for the batch lane against LRU-stored histories.

The dangerous pattern: a solo request's cache is stored (LRU), the batch
lane then reuses the same layer modules (rebinding the per-attention
states to a different cache object and row count), and the stored cache
is fetched later to continue the first conversation. The stored history
must not rot while the batch lane overwrites the persistent buffers --
the continuation must equal an uninterrupted solo run bit for bit.

    python benchmarks/maple_batch_lru_check.py
"""

import json

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import maple
from mlx_lm.models.cache import make_prompt_cache

PROMPT_A = "Tell a short story about a lighthouse keeper."
BATCH_PROMPTS = [
    "List three uses for a paperclip.",
    "What is the capital of Portugal? Answer briefly.",
    "Give one tip for writing readable code.",
    "Name a prime number between 90 and 100.",
]
SEG = 24


def toks(tok, p):
    return list(tok.apply_chat_template(
        [{"role": "user", "content": p}], add_generation_prompt=True))


def decode(model, y, cache, steps):
    got = []
    for _ in range(steps):
        out = model(y, cache=cache)
        y = mx.argmax(out[:, -1, :], axis=-1, keepdims=True)
        mx.eval(y)
        got.append(int(y.item()))
    return got, y


def batch_run(model, tok, steps):
    maple._use_batch_megakernels = True
    rows = [toks(tok, p) for p in BATCH_PROMPTS]
    cut = min(len(x) for x in rows)
    rows = [x[:cut] for x in rows]
    cache = make_prompt_cache(model)
    out = model(mx.array(rows), cache=cache)
    mx.eval(out)
    y = mx.argmax(out[:, -1, :], axis=-1, keepdims=True)
    mx.eval(y)
    for _ in range(steps):
        out = model(y, cache=cache)
        y = mx.argmax(out[:, -1, :], axis=-1, keepdims=True)
        mx.eval(y)
    maple._use_batch_megakernels = False


def main():
    model, tok = load(
        "model-cuda",
        model_config={"model_file": None, "use_flash_head": False},
        tokenizer_config={"trust_remote_code": False},
        trust_remote_code=False,
    )
    ids_a = toks(tok, PROMPT_A)

    # Reference: uninterrupted solo run, 2*SEG steps.
    cache = make_prompt_cache(model)
    out = model(mx.array([ids_a]), cache=cache)
    mx.eval(out)
    y = mx.argmax(out[:, -1, :], axis=-1, keepdims=True)
    mx.eval(y)
    first = int(y.item())
    ref_tokens, _ = decode(model, y, cache, 2 * SEG)

    # Interrupted: SEG steps, store (the LRU keeps the object), run the
    # batch lane over the same modules, fetch and continue SEG steps.
    cache = make_prompt_cache(model)
    out = model(mx.array([ids_a]), cache=cache)
    mx.eval(out)
    y = mx.argmax(out[:, -1, :], axis=-1, keepdims=True)
    mx.eval(y)
    assert int(y.item()) == first
    got1, y = decode(model, y, cache, SEG)
    stored = cache          # what an LRU keeps: the object, maybe views
    batch_run(model, tok, SEG)
    got2, _ = decode(model, y, stored, SEG)

    got = got1 + got2
    ok = got == ref_tokens
    div = next((i for i, (a, b) in enumerate(zip(ref_tokens, got))
                if a != b), None)
    print(json.dumps({
        "continuation_bit_equal": ok,
        "first_divergence_step": div,
        "verdict": "PASS" if ok else "FAIL",
    }), flush=True)


if __name__ == "__main__":
    main()
