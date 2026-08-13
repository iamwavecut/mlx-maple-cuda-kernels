"""Bit gate for the batch (M=B) exact-MoE megakernel.

Reference: B independent PRODUCTION dispatches (`_moe_exact_megakernel_call`
on real layer weights), each row its own random hidden/residual pair.
Candidate: ONE `_moe_batch_megakernel` dispatch over the stacked rows.
Every output surface must match bit for bit per row: out (the next-norm
activation), hout (the residual stream), and the routing tail (top-8
indices + renormed scores read back from scratch).

    python benchmarks/maple_moe_batch_check.py [--layers 0,3,7]
"""

import argparse
import json

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import maple


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model-cuda")
    ap.add_argument("--layers", default="0,3,7")
    ap.add_argument("--trials", type=int, default=6)
    args = ap.parse_args()

    model, _tok = load(
        args.model,
        model_config={"model_file": None, "use_flash_head": False},
        tokenizer_config={"trust_remote_code": False},
        trust_remote_code=False,
    )
    layers = [model.model.layers[int(i)] for i in args.layers.split(",")]
    next_norms = model.model._megakernel_next_norms()

    report = {}
    total = ok = 0
    for li, layer in zip(args.layers.split(","), layers):
        block = layer.mlp
        ln = layer.post_attention_layernorm
        next_w = next_norms[int(li)]
        for B in (2, 4, 8):
            hits = {"out": 0, "hout": 0, "route": 0}
            for t in range(args.trials):
                mx.random.seed(91000 + 977 * B + 31 * int(li) + t)
                h = (mx.random.normal((B, 2048)) * 0.5).astype(mx.bfloat16)
                r = (mx.random.normal((B, 2048)) * 0.5).astype(mx.bfloat16)
                mx.eval(h, r)

                pplan = maple._moe_exact_megakernel_plan(block, ln, h.dtype)
                assert pplan is not False, "production plan refused"
                pkernel, pkwargs = pplan
                mlp0 = block.switch_mlp
                ug0, dp0 = mlp0.up_gate_proj, mlp0.down_proj
                refs = []
                for row in range(B):
                    got = pkernel(
                        inputs=[h[row].reshape(1, 1, -1),
                                r[row].reshape(1, 1, -1), ln.weight,
                                block.gate.weight, ug0.weight, ug0.scales,
                                ug0.biases, dp0.weight, dp0.scales,
                                dp0.biases, next_w],
                        **pkwargs,
                    )
                    refs.append(got)  # (out, hout, scratch)
                mx.eval(*[a for trip in refs for a in trip])

                plan = maple._moe_batch_megakernel_plan(
                    block, ln, h.dtype, B)
                assert plan is not False, "batch plan refused"
                kernel, kwargs = plan
                mlp = block.switch_mlp
                ug, dp = mlp.up_gate_proj, mlp.down_proj
                bout, bhout, bscr = kernel(
                    inputs=[h, r, ln.weight, block.gate.weight, ug.weight,
                            ug.scales, ug.biases, dp.weight, dp.scales,
                            dp.biases, next_w],
                    **kwargs,
                )
                mx.eval(bout, bhout, bscr)

                rows_out = all(
                    bool(mx.array_equal(
                        bout[row, 0], refs[row][0].reshape(-1)).item())
                    for row in range(B))
                rows_hout = all(
                    bool(mx.array_equal(
                        bhout[row, 0], refs[row][1].reshape(-1)).item())
                    for row in range(B))
                # routing tail: production scratch holds idx[8] at float
                # offset 16 and the renormed scores at 24 -- compare the
                # batch planes bit for bit, kernel against kernel
                bidx = bscr[16:16 + B * 8].reshape(B, 8)
                bsco = bscr[16 + B * 8:16 + 2 * B * 8].reshape(B, 8)
                route_ok = all(
                    bool(mx.array_equal(
                        bidx[row], refs[row][2][16:24]).item())
                    and bool(mx.array_equal(
                        bsco[row], refs[row][2][24:32]).item())
                    for row in range(B))
                hits["out"] += int(rows_out)
                hits["hout"] += int(rows_hout)
                hits["route"] += int(route_ok)
                total += 3
                ok += int(rows_out) + int(rows_hout) + int(route_ok)
            report[f"L{li}_B{B}"] = {
                k: f"{v}/{args.trials}" for k, v in hits.items()}

    report["total"] = f"{ok}/{total}"
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
