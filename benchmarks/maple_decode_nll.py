"""Teacher-forced negative log-likelihood through the decode path.

Token equality answers "did the greedy path change"; it says nothing about how
far the distribution moved.  The document has to be stepped one token at a time
against a cache: feeding it whole runs the prefill path, where none of the
decode fusions are active, and every mode then agrees to eight decimals while
measuring nothing.

    python benchmarks/maple_decode_nll.py --model model-cuda --mode fast
"""

import argparse
import json

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import maple
from mlx_lm.models.cache import make_prompt_cache

TEXTS = [
    "The cache hierarchy of a modern processor exists because main memory is "
    "far slower than the arithmetic units it feeds. Each level trades capacity "
    "for latency, and the compiler and hardware cooperate to keep the working "
    "set as close to the core as possible.",
    "A mixture-of-experts layer replaces a single feed-forward network with a "
    "set of expert networks and a router that selects a small subset for each "
    "token. Only the selected experts run, so the parameter count grows "
    "without a proportional increase in computation.",
    "def quicksort(items):\n    if len(items) <= 1:\n        return items\n"
    "    pivot = items[len(items) // 2]\n"
    "    left = [x for x in items if x < pivot]\n"
    "    middle = [x for x in items if x == pivot]\n"
    "    right = [x for x in items if x > pivot]\n"
    "    return quicksort(left) + middle + quicksort(right)\n",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["off", "strict", "fast"], required=True)
    ap.add_argument("--top1-head", type=int, default=40)
    args = ap.parse_args()

    maple._use_fused_add_rms = args.mode != "off"
    maple._use_fused_qkv = args.mode != "off"
    maple._use_moe_megakernel = args.mode == "fast"

    model, tok, _ = load(
        args.model,
        return_config=True,
        model_config={"model_file": None, "use_flash_head": False},
        tokenizer_config={"trust_remote_code": False},
        trust_remote_code=False,
    )

    docs = []
    for index, text in enumerate(TEXTS):
        ids = tok.encode(text)
        cache = make_prompt_cache(model)
        nll_sum = 0.0
        count = 0
        top1 = []
        y = mx.array([[ids[0]]])
        for target in ids[1:]:
            logits = model(y, cache=cache)[:, -1, :].astype(mx.float32)
            lse = mx.logsumexp(logits, axis=-1)
            picked = logits[0, target]
            mx.eval(lse, picked)
            nll_sum += float(lse.item()) - float(picked.item())
            top1.append(int(mx.argmax(logits, axis=-1).item()))
            count += 1
            y = mx.array([[target]])
        docs.append({
            "doc": index,
            "tokens": count,
            "mean_nll": round(nll_sum / count, 10),
            "top1_head": top1[: args.top1_head],
        })
        del cache

    print(json.dumps({"mode": args.mode, "docs": docs}), flush=True)


if __name__ == "__main__":
    main()
