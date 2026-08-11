"""How much does a forward pass cost as a function of tokens per step?

Decode is host-bound: the step cost is the sum of per-operation host costs, and
the operation count does not depend on how many tokens ride through the layer.
If that held exactly, verifying k speculative tokens in one pass would cost
what one token costs.  It does not hold exactly, and the measured marginal cost
is what decides whether speculative decoding can pay for itself here.

    python benchmarks/maple_length_scaling.py --model model-cuda --mode strict
"""

import argparse
import json
import statistics
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import maple
from mlx_lm.models.cache import make_prompt_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["off", "strict", "fast"], default="strict")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--warm", type=int, default=20)
    ap.add_argument("--lengths", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32])
    args = ap.parse_args()

    maple._use_fused_add_rms = args.mode != "off"
    maple._use_fused_qkv = args.mode != "off"
    maple._use_moe_megakernel = args.mode == "fast"

    model, _, cfg = load(
        args.model,
        return_config=True,
        model_config={"model_file": None, "use_flash_head": False},
        tokenizer_config={"trust_remote_code": False},
        trust_remote_code=False,
    )
    vocab = cfg.get("vocab_size") or cfg["text_config"]["vocab_size"]

    for length in args.lengths:
        cache = make_prompt_cache(model)
        mx.random.seed(20260806)
        mx.eval(model(mx.random.randint(0, vocab, (1, 128)), cache=cache))
        mx.synchronize()

        build, total = [], []
        y = mx.random.randint(0, vocab, (1, length))
        mx.eval(y)
        pending = None
        for i in range(args.warm + args.steps):
            start = time.perf_counter()
            logits = model(y, cache=cache)
            nxt = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            built = time.perf_counter()
            mx.async_eval(nxt)
            if pending is not None:
                mx.eval(pending)
            pending = nxt
            done = time.perf_counter()
            if i >= args.warm:
                build.append(built - start)
                total.append(done - start)
        mx.eval(pending)
        mx.synchronize()

        step = statistics.median(total)
        print(json.dumps({
            "mode": args.mode,
            "tokens_per_pass": length,
            "host_build_ms": round(statistics.median(build) * 1e3, 4),
            "step_ms": round(step * 1e3, 4),
            "ms_per_token_if_all_accepted": round(step * 1e3 / length, 4),
            "max_tps_if_all_accepted": round(length / step, 1),
        }), flush=True)


if __name__ == "__main__":
    main()
