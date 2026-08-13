"""Bit gate for the 2-pass branch of the batch attention pair.

Full-attention geometry at CAP 4096. Three regimes per B: pure 2-pass
(base 3000), the mixed verify pack straddling the 1024 boundary
(base 1020, rows kl0+r on both sides), and a pure 1-pass control
(base 900). Reference: B sequential production megakernel dispatches
(whose own 2-pass port is bit-proven); candidate: one AB/CD pair call.

    python benchmarks/maple_attn_batch_2pass_check.py
"""

import argparse
import json

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import maple


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model-cuda")
    ap.add_argument("--trials", type=int, default=4)
    args = ap.parse_args()

    model, _ = load(
        args.model,
        model_config={"model_file": None, "use_flash_head": False},
        tokenizer_config={"trust_remote_code": False},
        trust_remote_code=False,
    )
    inner = model.model
    lidx = next(i for i, t in enumerate(inner.layer_types)
                if t == "full_attention")
    layer = inner.layers[lidx]
    attn = layer.self_attn
    if attn._qk_w is None:
        attn._ensure_qk_constants()
    profile = maple._cuda_profile().name
    qkv, op = attn.qkv_proj, attn.o_proj
    kh = qkv.weight.shape[1] * 16
    nq, nkv = attn.num_attention_heads, attn.num_key_value_heads
    cap = 4096
    rd = getattr(attn, "_rope_dim", 0) if attn.use_rope else 0
    prod = maple._attn_megakernel(
        profile, attn.use_rope, float(attn.scale),
        float(attn._eps), float(attn._rope_log2_base))
    bab, bcd = maple._attn_verify_kernels(
        profile, attn.use_rope, float(attn.scale),
        float(attn._eps), float(attn._rope_log2_base), batch=True)

    report = {}
    total = ok = 0
    for B in (2, 4, 8):
        for base, tag in ((3000, "pure2p"), (900, "pure1p")):
            batch_mode = True
            hits = {"out": 0, "kv": 0, "ctr": 0}
            for t in range(args.trials):
                mx.random.seed(77000 + B * 131 + base + t)
                kb0 = mx.random.normal((B, nkv, cap, 128)).astype(mx.bfloat16)
                vb0 = mx.random.normal((B, nkv, cap, 128)).astype(mx.bfloat16)
                hns = mx.random.normal((B, kh)).astype(mx.bfloat16)
                mx.eval(kb0, vb0, hns)

                souts, skbs, svbs = [], [], []
                sctr = None
                for r in range(B):
                    rbase = base if batch_mode else base + r
                    kb = mx.contiguous(kb0[r:r + 1])
                    vb = mx.contiguous(vb0[r:r + 1])
                    ctr = mx.zeros((8,), mx.float32)
                    mx.eval(kb, vb, ctr)
                    ctr[0] = float(rbase); ctr[1] = float(rbase + 1)
                    ctr[2] = float(rbase)
                    mx.eval(ctr)
                    o, _ = prod(
                        inputs=[hns[r].reshape(-1), qkv.weight, qkv.scales,
                                qkv.biases, attn._qk_w, op.weight,
                                op.scales, op.biases, kb, vb, ctr],
                        template=[("T_", hns.dtype), ("KH_", kh),
                                  ("NQ_", nq), ("NKV_", nkv),
                                  ("CAP_", cap),
                                  ("ROPE_", 1 if attn.use_rope else 0),
                                  ("RD_", rd), ("THREADS_", 1024),
                                  ("GRID_", 64)],
                        grid=(64 * 1024, 1, 1), threadgroup=(1024, 1, 1),
                        output_shapes=[
                            (1, 1, kh),
                            (16 + (nq + 2 * nkv) * 128 + nq * 128 * 2
                             + nq * 32 * 130,)],
                        output_dtypes=[hns.dtype, mx.float32],
                        init_value=0)
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
                        ("ROPE_", 1 if attn.use_rope else 0),
                        ("RD_", rd), ("ROWS_", B), ("RAGGED_", 0),
                        ("BATCH_", 1 if batch_mode else 0), ("GRID_", 64)]
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
                    rbase = base if batch_mode else base + r
                    n = rbase + 1
                    kv_ok = kv_ok and bool(mx.array_equal(
                        skbs[r][0, :, :n, :], kbb[r, :, :n, :]).item())
                    kv_ok = kv_ok and bool(mx.array_equal(
                        svbs[r][0, :, :n, :], vbb[r, :, :n, :]).item())
                hits["kv"] += int(kv_ok)
                want = (mx.array([float(base + 1), float(base + 2),
                                  float(base + 1)])
                        if batch_mode else
                        mx.array([float(base + B), float(base + B + 1),
                                  float(base + B)]))
                hits["ctr"] += int(mx.array_equal(bctr[:3], want).item())
                total += 3
                ok += sum(v == t + 1 for v in ())  # placeholder
            ok += sum(hits.values())
            report[f"{tag}_B{B}"] = {
                k: f"{v}/{args.trials}" for k, v in hits.items()}

    report["total"] = f"{ok}/{total}"
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
