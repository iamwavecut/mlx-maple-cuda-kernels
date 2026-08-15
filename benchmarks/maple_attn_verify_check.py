"""M=L attention verify megakernel vs L sequential production dispatches.

Real layer weights, a random prefilled cache, L random inputs; the
production megakernel is already bit-exact vs stock, so equality here
makes the verify kernel sequential-exact by transitivity. Checks outputs,
appended K/V slots and the live counters, on one RoPE and one NoPE layer.

    python benchmarks/maple_attn_verify_check.py --model <path>
"""
import argparse
import json

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import maple

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
args = ap.parse_args()

model, tok, cfg = load(args.model, return_config=True,
    model_config={"model_file": None, "use_flash_head": False},
    tokenizer_config={"trust_remote_code": False}, trust_remote_code=False)
inner = model.model

report = {}
profile = maple._cuda_profile().name

for lname, lidx in (("rope_sliding", 0), ("nope_full", 3)):
    layer = inner.layers[lidx]
    attn = layer.self_attn
    if attn._qk_w is None:
        attn._ensure_qk_constants()
    qkv, op = attn.qkv_proj, attn.o_proj
    kh = qkv.weight.shape[1] * 16  # 2-bit packed uint32: 16 elems per word
    cap = 512
    base = 173
    prod = maple._attn_megakernel(
        profile, attn.use_rope, float(attn.scale),
        float(attn._eps), float(attn._rope_log2_base))
    vab, vcd = maple._attn_verify_kernels(
        profile, attn.use_rope, float(attn.scale),
        float(attn._eps), float(attn._rope_log2_base))
    rd = getattr(attn, "_rope_dim", 0) if attn.use_rope else 0

    for L in (4, 8, 16):
        hits = {"out": 0, "kv": 0, "ctr": 0}
        for t in range(6):
            mx.random.seed(81000 + lidx * 97 + L * 7 + t)
            kb0 = mx.random.normal(
                (1, attn.num_key_value_heads, cap, 128)).astype(mx.bfloat16)
            vb0 = mx.random.normal(
                (1, attn.num_key_value_heads, cap, 128)).astype(mx.bfloat16)
            hns = mx.random.normal((L, kh)).astype(mx.bfloat16)
            mx.eval(kb0, vb0, hns)

            def run_seq():
                kb = mx.contiguous(kb0); vb = mx.contiguous(vb0)
                ctr = mx.zeros((8,), mx.float32)
                mx.eval(kb, vb, ctr)
                ctr[0] = float(base); ctr[1] = float(base + 1)
                ctr[2] = float(base)
                mx.eval(ctr)
                outs = []
                for r in range(L):
                    pscr = mx.zeros(
                        (16 + (attn.num_attention_heads
                               + 2 * attn.num_key_value_heads) * 128
                         + attn.num_attention_heads * 128 * 2
                         + attn.num_attention_heads * 32 * 130,),
                        mx.float32)
                    mx.eval(pscr)
                    (o,) = prod(
                        inputs=[hns[r].reshape(-1), qkv.weight, qkv.scales,
                                qkv.biases, attn._qk_w, op.weight, op.scales,
                                op.biases, kb, vb, ctr, pscr],
                        template=[("T_", hns.dtype), ("KH_", kh),
                                  ("NQ_", attn.num_attention_heads),
                                  ("NKV_", attn.num_key_value_heads),
                                  ("CAP_", cap),
                                  ("ROPE_", 1 if attn.use_rope else 0),
                                  ("RD_", rd), ("THREADS_", 1024),
                                  ("GRID_", 64)],
                        grid=(64 * 1024, 1, 1), threadgroup=(1024, 1, 1),
                        output_shapes=[(1, 1, kh)],
                        output_dtypes=[hns.dtype])
                    outs.append(o.reshape(1, kh))
                    mx.eval(o)
                return mx.concatenate(outs, axis=0), kb, vb, ctr

            def run_ver():
                kb = mx.contiguous(kb0); vb = mx.contiguous(vb0)
                ctr = mx.zeros((8,), mx.float32)
                mx.eval(kb, vb, ctr)
                ctr[0] = float(base); ctr[1] = float(base + 1)
                ctr[2] = float(base)
                mx.eval(ctr)
                nq, nkv = attn.num_attention_heads, attn.num_key_value_heads
                tmpl = [("T_", hns.dtype), ("KH_", kh),
                        ("NQ_", nq), ("NKV_", nkv), ("CAP_", cap),
                        ("ROPE_", 1 if attn.use_rope else 0),
                        ("RD_", rd), ("ROWS_", L), ("RAGGED_", 0),
                        ("BATCH_", 0), ("GRID_", 64)]
                scr_shape = (16 + L * ((nq + 2 * nkv) * 128
                                       + nq * 128 * 2 + kh),)
                (scr,) = vab(
                    inputs=[hns.reshape(-1), qkv.weight, qkv.scales,
                            qkv.biases, attn._qk_w, kb, vb, ctr],
                    template=tmpl + [("THREADS_", 512)],
                    grid=(64 * 512, 1, 1), threadgroup=(512, 1, 1),
                    output_shapes=[scr_shape],
                    output_dtypes=[mx.float32], init_value=0)
                (o,) = vcd(
                    inputs=[scr, op.weight, op.scales, op.biases,
                            kb, vb, ctr],
                    template=tmpl + [("THREADS_", 1024)],
                    grid=(64 * 1024, 1, 1), threadgroup=(1024, 1, 1),
                    output_shapes=[(L, kh)],
                    output_dtypes=[hns.dtype])
                mx.eval(o)
                return o, kb, vb, ctr

            so, skb, svb, sctr = run_seq()
            vo, vkb, vvb, vctr = run_ver()
            mx.eval(so, vo, skb, vkb, svb, vvb, sctr, vctr)
            n = base + L
            hits["out"] += int(mx.array_equal(so, vo).item())
            hits["kv"] += int(
                mx.array_equal(skb[..., :n, :], vkb[..., :n, :]).item()
                and mx.array_equal(svb[..., :n, :], vvb[..., :n, :]).item())
            hits["ctr"] += int(
                mx.array_equal(sctr[:3], vctr[:3]).item())
        report[f"{lname}_L{L}"] = {k: f"{v}/6" for k, v in hits.items()}
        print(json.dumps({f"{lname}_L{L}": report[f"{lname}_L{L}"]}),
              flush=True)

print(json.dumps(report), flush=True)
