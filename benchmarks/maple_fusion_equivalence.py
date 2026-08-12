"""Token equality of every shipped mode against the all-off path.

The stock reference is not always reproducible run to run, so each prompt is
generated three times with the fusions off and a candidate only counts as
divergent inside the region where those three agree.
"""
import argparse, json
from pathlib import Path
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models import maple

ap = argparse.ArgumentParser()
ap.add_argument("--model", type=Path, required=True)
ap.add_argument("--gen", type=int, default=512)
ap.add_argument("--repeats", type=int, default=3)
a = ap.parse_args()

model, tok, cfg = load(str(a.model), return_config=True,
    model_config={"model_file": None, "use_flash_head": False},
    tokenizer_config={"trust_remote_code": False}, trust_remote_code=False)
tok._eos_token_ids = {}
vocab = cfg.get("vocab_size") or cfg["text_config"]["vocab_size"]

TEXTS = [
    "Explain how a write-back cache decides which line to evict.",
    "Summarize the difference between latency and throughput for an inference server.",
    "Write a short Python function that merges two sorted lists.",
]
prompts = []
for i in range(5):
    mx.random.seed(5000 + i)
    prompts.append((f"random{i}", mx.random.randint(0, vocab, (128,)).tolist()))
for i, t in enumerate(TEXTS):
    try:
        p = tok.apply_chat_template([{"role": "user", "content": t}],
                                    add_generation_prompt=True)
    except Exception:
        p = t
    prompts.append((f"text{i}", p))

def configure(mode):
    maple._use_fused_add_rms = mode != "off"
    maple._use_fused_qkv = mode != "off"
    maple._use_compiled_router = mode != "off"
    maple._use_moe_megakernel = mode == "fast"
    maple._use_moe_megakernel_exact = mode in ("exact", "exact+attn")
    maple._use_attention_megakernel = mode == "exact+attn"
    for l in model.model.layers:
        if hasattr(l.self_attn, "_mega_state"):
            del l.self_attn._mega_state
    model.model._exact_add_norm = None
    for l in model.model.layers:
        l.self_attn._split_qkv = None
        l.mlp.gate._compiled_ok = None
        if hasattr(l.mlp, "_megakernel_plan"):
            del l.mlp._megakernel_plan
        if hasattr(l.mlp, "_exact_megakernel_plan"):
            del l.mlp._exact_megakernel_plan

def gen(p):
    return [int(r.token) for r in
            stream_generate(model, tok, p, max_tokens=a.gen, prefill_step_size=2048)]

summary = {"strict": {"prompts": 0, "identical": 0, "mismatches": []},
           "fast": {"prompts": 0, "identical": 0, "mismatches": []},
           "exact": {"prompts": 0, "identical": 0, "mismatches": []},
           "exact+attn": {"prompts": 0, "identical": 0, "mismatches": []}}
unstable = []
for name, prompt in prompts:
    configure("off")
    runs = [gen(prompt) for _ in range(a.repeats)]
    stable = a.gen
    for r in runs[1:]:
        m = next((j for j, (x, y) in enumerate(zip(runs[0], r)) if x != y), None)
        if m is not None:
            stable = min(stable, m)
    if stable < a.gen:
        unstable.append({"prompt": name, "reference_stable_up_to": stable})
    for mode in ("strict", "fast", "exact", "exact+attn"):
        configure(mode)
        cand = gen(prompt)
        s = summary[mode]
        s["prompts"] += 1
        m = next((j for j in range(stable) if cand[j] != runs[0][j]), None)
        if m is None:
            s["identical"] += 1
        else:
            s["mismatches"].append({"prompt": name, "index": m})
print(json.dumps({"reference_unstable": unstable}), flush=True)
for mode in ("strict", "fast", "exact", "exact+attn"):
    print(json.dumps({"mode": mode, **summary[mode]}), flush=True)
