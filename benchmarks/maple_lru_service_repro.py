"""Reproduce the prod cross-request leak with the service's own LRU flow.

Mirrors maple_service: per request — fetch_nearest_cache, (make if None),
prefill rest, greedy decode, then store the cache back into the LRU.
Compares the second/third requests' tokens between the attention lane on
and off. Any divergence is the leak.

    python lab/lru_service_repro.py
"""
import json

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import maple
from mlx_lm.models.cache import (LRUPromptCache, make_prompt_cache,
                                 trim_prompt_cache)

MODEL = "model-cuda"
model, tok, cfg = load(MODEL, return_config=True,
    model_config={"model_file": None, "use_flash_head": False},
    tokenizer_config={"trust_remote_code": False}, trust_remote_code=False)
inner = model.model

DOC = ("The quarterly report shows revenue of 4.2 million dollars in Q1, "
       "rising to 4.8 million dollars in Q2. Operating costs stayed flat.")
CODE = ("Rewrite this function to handle status 'urgent' too:\n"
        "def f(orders):\n    return [o for o in orders if o['status'] == 'pending']")
PROMPTS = [
    DOC + " Extract every number as JSON.",
    CODE,
    DOC + " Summarize in one sentence.",
    DOC + " Render the figures as a markdown table.",
]


def toks_for(p):
    msgs = [{"role": "user", "content": p}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True)


def serve(attn_on, gens=48):
    maple._use_attention_megakernel = attn_on
    for l in inner.layers:
        if hasattr(l.self_attn, "_mega_state"):
            del l.self_attn._mega_state
    lru = LRUPromptCache(max_size=8)
    outs = []
    for p in PROMPTS:
        ids = list(toks_for(p))
        cache, rest = lru.fetch_nearest_cache(id(model), ids)
        used_cached = cache is not None
        if cache is None:
            cache = make_prompt_cache(model)
            rest = ids
        elif not rest:
            trim_prompt_cache(cache, 1)
            rest = ids[-1:]
        r = mx.array([rest])
        out = model(r, cache=cache); mx.eval(out)
        y = mx.argmax(out[:, -1, :], axis=-1, keepdims=True); mx.eval(y)
        got = [int(y.item())]
        for _ in range(gens - 1):
            out = model(y, cache=cache); mx.eval(out)
            y = mx.argmax(out[:, -1, :], axis=-1, keepdims=True); mx.eval(y)
            got.append(int(y.item()))
        lru.insert_cache(id(model), ids + got, cache)
        outs.append({"cached": used_cached, "toks": got})
    return outs


ref = serve(False)
got = serve(True)
rep = {}
for i, (a, b) in enumerate(zip(ref, got)):
    m = next((j for j, (x, z) in enumerate(zip(a["toks"], b["toks"]))
              if x != z), None)
    rep[f"req{i}"] = {"cached_ref": a["cached"], "cached_got": b["cached"],
                      "identical": m is None, "first_div": m}
print(json.dumps(rep), flush=True)
