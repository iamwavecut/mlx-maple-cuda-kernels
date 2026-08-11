"""Release benchmark for the integrated source.

Modes map to the flags in maple.py, so this measures the shipped code paths
rather than monkeypatches:

  off     every fusion disabled -- reproduces the previous release behaviour
  strict  the three array-exact fusions, which are the new defaults
  fast    strict plus the opt-in MoE megakernel
"""
import argparse, hashlib, json
from pathlib import Path
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models import maple

ap = argparse.ArgumentParser()
ap.add_argument("--model", type=Path, required=True)
ap.add_argument("--mode", required=True,
                help="off | norm | qkv | router | strict | fast, or a + joined set")
ap.add_argument("--prompt-tokens", type=int, default=128)
ap.add_argument("--generation-tokens", type=int, default=512)
ap.add_argument("--warmup-tokens", type=int, default=64)
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--prompt-seed", type=int, default=20260806)
a = ap.parse_args()

parts = set(a.mode.split("+"))
if "strict" in parts or "fast" in parts:
    parts |= {"norm", "qkv"}
maple._use_fused_add_rms = "norm" in parts
maple._use_fused_qkv = "qkv" in parts
maple._use_compiled_router = "router" in parts
maple._use_moe_megakernel = "fast" in parts
if "fast" in parts:
    # the megakernel absorbs the router, so the compiled one never runs
    maple._use_compiled_router = False

mx.random.seed(20260806)
model, tok, cfg = load(str(a.model), return_config=True,
    model_config={"model_file": None, "use_flash_head": False},
    tokenizer_config={"trust_remote_code": False}, trust_remote_code=False)
tok._eos_token_ids = {}
vocab = cfg.get("vocab_size") or cfg["text_config"]["vocab_size"]
mx.random.seed(a.prompt_seed)
prompt = mx.random.randint(0, vocab, (a.prompt_tokens,)).tolist()

def run(n):
    toks, r = [], None
    for r in stream_generate(model, tok, prompt, max_tokens=n, prefill_step_size=2048):
        toks.append(int(r.token))
    return r.generation_tps, toks, r.peak_memory

run(a.warmup_tokens); mx.synchronize()
best, toks, peak = 0.0, None, 0.0
for _ in range(a.repeats):
    tps, toks, peak = run(a.generation_tokens)
    best = max(best, tps); mx.synchronize()

inner = model.model
print(json.dumps({
    "mode": a.mode,
    "generation_tps": best,
    "peak_memory_gb": round(peak, 4),
    "token_sha256": hashlib.sha256(",".join(map(str, toks)).encode()).hexdigest(),
    "state": {
        "exact_add_norm": bool(getattr(inner, "_exact_add_norm", False)),
        "qkv_split_layers": sum(
            1 for l in inner.layers if getattr(l.self_attn, "_split_qkv", None) is True),
        "compiled_router_layers": sum(
            1 for l in inner.layers if getattr(l.mlp.gate, "_compiled_ok", None) is True),
        "megakernel_layers": sum(
            1 for l in inner.layers
            if getattr(l.mlp, "_megakernel_plan", None) not in (None, False)),
    },
}), flush=True)
