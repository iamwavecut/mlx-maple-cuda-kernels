"""Batch attention (M=B) vs B independent production streams.

Each batch row carries its OWN random cache prefix at a shared offset;
the batch pair (BATCH_=1: constant pos/slot/kL, per-row cache planes)
must reproduce every row's solo production-dispatch result bit for bit:
outputs, the appended K/V slot in each plane, and the shared counters.

    python benchmarks/maple_attn_batch_check.py --model <path>
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
profile = maple._cuda_profile().name
report = {}

for lname, lidx in (("rope_sliding", 0), ("nope_full", 3)):
    layer = inner.layers[lidx]
    attn = layer.self_attn
    if attn._qk_w is None:
        attn._ensure_qk_constants()
    qkv, op = attn.qkv_proj, attn.o_proj
    kh = qkv.weight.shape[1] * 16
    nq, nkv = attn.num_attention_heads, attn.num_key_value_heads
    cap, base = 512, 173
    rd = getattr(attn, "_rope_dim", 0) if attn.use_rope else 0
    prod = maple._attn_megakernel(
        profile, attn.use_rope, float(attn.scale),
        float(attn._eps), float(attn._rope_log2_base))
    bab, bcd = maple._attn_verify_kernels(
        profile, attn.use_rope, float(attn.scale),
        float(attn._eps), float(attn._rope_log2_base), batch=True)

    for B in (2, 4, 8):
        hits = {"out": 0, "kv": 0, "ctr": 0}
        for t in range(6):
            mx.random.seed(91000 + lidx * 131 + B * 11 + t)
            kb0 = mx.random.normal((B, nkv, cap, 128)).astype(mx.bfloat16)
            vb0 = mx.random.normal((B, nkv, cap, 128)).astype(mx.bfloat16)
            hns = mx.random.normal((B, kh)).astype(mx.bfloat16)
            mx.eval(kb0, vb0, hns)

            souts, skbs, svbs = [], [], []
            sctr = None
            for r in range(B):
                kb = mx.contiguous(kb0[r:r + 1])
                vb = mx.contiguous(vb0[r:r + 1])
                ctr = mx.zeros((8,), mx.float32)
                mx.eval(kb, vb, ctr)
                ctr[0] = float(base); ctr[1] = float(base + 1)
                ctr[2] = float(base)
                mx.eval(ctr)
                pscr = mx.zeros(
                    (16 + (nq + 2 * nkv) * 128 + nq * 128 * 2
                     + nq * 32 * 130,), mx.float32)
                mx.eval(pscr)
                (o,) = prod(
                    inputs=[hns[r].reshape(-1), qkv.weight, qkv.scales,
                            qkv.biases, attn._qk_w, op.weight, op.scales,
                            op.biases, kb, vb, ctr, pscr],
                    template=[("T_", hns.dtype), ("KH_", kh),
                              ("NQ_", nq), ("NKV_", nkv), ("CAP_", cap),
                              ("ROPE_", 1 if attn.use_rope else 0),
                              ("RD_", rd), ("THREADS_", 1024),
                              ("GRID_", 64)],
                    grid=(64 * 1024, 1, 1), threadgroup=(1024, 1, 1),
                    output_shapes=[(1, 1, kh)],
                    output_dtypes=[hns.dtype])
                mx.eval(o)
                souts.append(o.reshape(1, kh))
                skbs.append(kb); svbs.append(vb); sctr = ctr
            sout = mx.concatenate(souts, axis=0)

            kbb = mx.contiguous(kb0); vbb = mx.contiguous(vb0)
            bctr = mx.zeros((8,), mx.float32)
            mx.eval(kbb, vbb, bctr)
            bctr[0] = float(base); bctr[1] = float(base + 1)
            bctr[2] = float(base)
            mx.eval(bctr)
            tmpl = [("T_", hns.dtype), ("KH_", kh), ("NQ_", nq),
                    ("NKV_", nkv), ("CAP_", cap),
                    ("ROPE_", 1 if attn.use_rope else 0), ("RD_", rd),
                    ("ROWS_", B), ("BATCH_", 1), ("RAGGED_", 0), ("GRID_", 64)]
            scr_shape = (16 + B * ((nq + 2 * nkv) * 128
                                   + nq * 128 * 2 + kh
                                   + nq * 32 * (128 + 2)),)
            (scr,) = bab(
                inputs=[hns.reshape(-1), qkv.weight, qkv.scales,
                        qkv.biases, attn._qk_w, kbb, vbb, bctr],
                template=tmpl + [("THREADS_", 512)],
                grid=(64 * 512, 1, 1), threadgroup=(512, 1, 1),
                output_shapes=[scr_shape], output_dtypes=[mx.float32],
                init_value=0)
            (bo,) = bcd(
                inputs=[scr, op.weight, op.scales, op.biases,
                        kbb, vbb, bctr],
                template=tmpl + [("THREADS_", 1024)],
                grid=(64 * 1024, 1, 1), threadgroup=(1024, 1, 1),
                output_shapes=[(B, kh)], output_dtypes=[hns.dtype])
            mx.eval(bo)

            hits["out"] += int(mx.array_equal(sout, bo).item())
            kv_ok = True
            for r in range(B):
                n = base + 1
                kv_ok = kv_ok and bool(mx.array_equal(
                    skbs[r][0, :, :n, :], kbb[r, :, :n, :]).item())
                kv_ok = kv_ok and bool(mx.array_equal(
                    svbs[r][0, :, :n, :], vbb[r, :, :n, :]).item())
            hits["kv"] += int(kv_ok)
            hits["ctr"] += int(mx.array_equal(sctr[:3], bctr[:3]).item())
        report[f"{lname}_B{B}"] = {k: f"{v}/6" for k, v in hits.items()}
        print(json.dumps({f"{lname}_B{B}": report[f"{lname}_B{B}"]}),
              flush=True)
print(json.dumps(report), flush=True)
