"""Teacher-forced quality over a varied corpus, through the decode path.

Token equality answers "did the greedy path change". It does not answer "is the
model worse", and for a lane that is within ~1 ULP rather than array-exact that
is the question that matters.

Every document is stepped one token at a time against a cache, so the fused
decode kernels are actually exercised — feeding a document whole runs the
prefill path, where none of them are active and every mode trivially agrees.

Reported per mode: total negative log-likelihood per token over the corpus,
per-document deltas against the reference mode, and how often the top-1
prediction changes. A lane that only reorders the last bit should move the
aggregate by far less than its per-document spread, and in both directions.

    python benchmarks/maple_quality_suite.py --model model-cuda --mode off
    python benchmarks/maple_quality_suite.py --model model-cuda --mode fast
"""

import argparse
import json

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import maple
from mlx_lm.models.cache import make_prompt_cache

DOCUMENTS = [
    # expository prose
    "The cache hierarchy of a modern processor exists because main memory is far "
    "slower than the arithmetic units it feeds. Each level trades capacity for "
    "latency, and the compiler and the hardware cooperate to keep the working set "
    "as close to the core as possible. A miss that reaches DRAM costs hundreds of "
    "cycles, which is long enough to retire a great deal of unrelated work.",
    "A mixture-of-experts layer replaces a single feed-forward network with a set "
    "of expert networks and a router that selects a small subset for each token. "
    "Only the selected experts run, so the parameter count grows without a "
    "proportional increase in computation. The difficulty moves from arithmetic to "
    "memory: the weights of every selected expert must still be read.",
    "Rotary position embeddings encode position by rotating pairs of dimensions in "
    "the query and key vectors. Because the rotation is applied before the dot "
    "product, the resulting attention score depends only on the relative distance "
    "between two positions, which is what makes the scheme extrapolate more "
    "gracefully than a learned absolute table.",
    # instructions and dialogue
    "To reproduce the issue, start the server with logging at debug level, send a "
    "request with an empty payload, and watch for the retry loop. If the loop "
    "starts, capture the process state before restarting anything; a restart "
    "discards exactly the evidence needed to explain the failure.",
    "Q: Why did the deployment succeed in staging but fail in production?\n"
    "A: The two environments disagreed about a default. Staging had the feature "
    "flag enabled from an earlier experiment, so the new code path was already "
    "warm there and never ran cold.",
    # code
    "def quicksort(items):\n    if len(items) <= 1:\n        return items\n"
    "    pivot = items[len(items) // 2]\n"
    "    left = [x for x in items if x < pivot]\n"
    "    middle = [x for x in items if x == pivot]\n"
    "    right = [x for x in items if x > pivot]\n"
    "    return quicksort(left) + middle + quicksort(right)\n",
    "class RingBuffer:\n    def __init__(self, capacity):\n"
    "        self._items = [None] * capacity\n        self._head = 0\n"
    "        self._size = 0\n\n    def push(self, value):\n"
    "        index = (self._head + self._size) % len(self._items)\n"
    "        self._items[index] = value\n"
    "        if self._size < len(self._items):\n            self._size += 1\n"
    "        else:\n            self._head = (self._head + 1) % len(self._items)\n",
    "SELECT customer_id, COUNT(*) AS orders, SUM(total) AS revenue\n"
    "FROM orders\nWHERE created_at >= DATE '2026-01-01'\n"
    "GROUP BY customer_id\nHAVING COUNT(*) > 3\nORDER BY revenue DESC\nLIMIT 50;\n",
    # structured text
    "Status codes worth memorizing: 200 means the request succeeded. 201 means a "
    "resource was created. 204 means success with no body. 301 is a permanent "
    "redirect, 302 a temporary one. 400 means the request was malformed, 401 that "
    "it was unauthenticated, 403 that it was authenticated and still refused, and "
    "404 that the resource does not exist.",
    "Ingredients: two cups of flour, one teaspoon of salt, three tablespoons of "
    "olive oil, and about three quarters of a cup of warm water. Combine the dry "
    "ingredients, add the oil, then add water gradually until the dough pulls away "
    "from the sides of the bowl.",
    # numbers and reasoning
    "If a train leaves at 14:20 travelling at 90 kilometres per hour and the next "
    "station is 150 kilometres away, it arrives at roughly 16:00. A ten minute "
    "delay at departure pushes arrival past the connection at 16:05, so the "
    "connection is missed unless the following train also runs late.",
    "The sequence begins 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89. Each term is the "
    "sum of the two before it, and the ratio between consecutive terms approaches "
    "the golden ratio, roughly 1.618, from alternating sides.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["off", "strict", "fast", "exact"], required=True)
    args = ap.parse_args()

    maple._use_fused_add_rms = args.mode != "off"
    maple._use_fused_qkv = args.mode != "off"
    maple._use_moe_megakernel = args.mode == "fast"
    maple._use_moe_megakernel_exact = args.mode == "exact"

    model, tok, _ = load(
        args.model,
        return_config=True,
        model_config={"model_file": None, "use_flash_head": False},
        tokenizer_config={"trust_remote_code": False},
        trust_remote_code=False,
    )

    docs = []
    total_nll = 0.0
    total_tokens = 0
    for index, text in enumerate(DOCUMENTS):
        ids = tok.encode(text)
        if len(ids) < 4:
            continue
        cache = make_prompt_cache(model)
        nll_sum = 0.0
        count = 0
        top1 = []
        y = mx.array([[ids[0]]])
        for target in ids[1:]:
            logits = model(y, cache=cache)[:, -1, :].astype(mx.float32)
            lse = mx.logsumexp(logits, axis=-1)
            picked = logits[0, target]
            arg = mx.argmax(logits, axis=-1)
            mx.eval(lse, picked, arg)
            nll_sum += float(lse.item()) - float(picked.item())
            top1.append(int(arg.item()))
            count += 1
            y = mx.array([[target]])
        docs.append({"doc": index, "tokens": count,
                     "mean_nll": nll_sum / count, "top1": top1})
        total_nll += nll_sum
        total_tokens += count
        del cache

    print(json.dumps({
        "mode": args.mode,
        "documents": len(docs),
        "tokens": total_tokens,
        "corpus_mean_nll": total_nll / total_tokens,
        "corpus_perplexity": float(mx.exp(mx.array(total_nll / total_tokens)).item()),
        "docs": docs,
    }), flush=True)


if __name__ == "__main__":
    main()
