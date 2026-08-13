"""How many UNIQUE experts does a pack of L consecutive tokens route to?

MoE-verify's GPU cost is (unique experts in the pack) / 8 of a single
step's expert phase — shared experts ride the free MMA rows. This runs
stock decode (fused lanes off) over corpus-like prompts, records every
layer's top-8 per token, and reports the sharing factor for pack sizes
L=4/8/16.

    python benchmarks/maple_expert_overlap_probe.py --model <path>
"""
import argparse
import json
import statistics

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import maple
from mlx_lm.models.cache import make_prompt_cache

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--gens", type=int, default=96)
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
    "The quarterly report shows revenue of 4.2 million dollars in Q1, "
    "rising to 4.8 million in Q2. Summarize as three bullets quoting figures.",
    "Explain briefly why rivers meander.",
]

maple._use_moe_megakernel = False
maple._use_moe_megakernel_exact = False
maple._use_attention_megakernel = False

step_inds = []           # per decode step: list of per-layer ind refs
collecting = False
orig = maple.MapleSparseMoeBlock.__call__


def tap(self, x):
    inds, scores = self.gate(x)
    if collecting:
        step_inds[-1].append(inds)
    y = self.switch_mlp(x, inds)
    return maple.aggregate_expert_outputs(y, scores)


maple.MapleSparseMoeBlock.__call__ = tap

per_prompt = {}
for pi, prompt in enumerate(PROMPTS):
    ids = list(tok.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True))
    cache = make_prompt_cache(model)
    out = model(mx.array([ids]), cache=cache); mx.eval(out)
    y = mx.argmax(out[:, -1, :], axis=-1, keepdims=True); mx.eval(y)
    step_inds.clear()
    collecting = True
    for _ in range(args.gens):
        step_inds.append([])
        out = model(y, cache=cache); mx.eval(out)
        y = mx.argmax(out[:, -1, :], axis=-1, keepdims=True); mx.eval(y)
    collecting = False
    # materialize: steps x layers x 8
    grid = []
    for step in step_inds:
        row = []
        for t in step:
            mx.eval(t)
            row.append([int(v) for v in t.reshape(-1).tolist()])
        grid.append(row)
    n_layers = len(grid[0])
    res = {}
    for L in (4, 8, 16):
        factors = []
        for s0 in range(0, len(grid) - L + 1, L):
            for li in range(n_layers):
                uniq = set()
                for s in range(s0, s0 + L):
                    uniq.update(grid[s][li])
                factors.append(len(uniq) / (8 * L))
        res[f"L{L}"] = {
            "mean_unique_frac": round(statistics.mean(factors), 3),
            "moe_cost_vs_single": round(
                statistics.mean(factors) * L, 2),
        }
    per_prompt[f"p{pi}"] = res
    print(json.dumps({f"p{pi}": res}), flush=True)

print(json.dumps(per_prompt), flush=True)
