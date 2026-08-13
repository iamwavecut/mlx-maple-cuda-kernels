"""Bit gate for the RAGGED_ mode of the batch attention pair.

Every row carries its OWN (pos, kL, slot) -- the continuous-batching
shape, where concurrent requests sit at different offsets. Reference:
B production dispatches, each seeded with its row's counters. Candidate:
one pair call with the per-row counter array (live[3 + r*3 + 0..2]).
Covers the sliding geometry (cap 512, wrapped and unwrapped rings) and
the full geometry (cap 4096, rows straddling the 1024 2-pass boundary).

    python benchmarks/maple_attn_ragged_check.py
"""

import argparse
import json

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import maple


def run_case(attn, profile, cap, offsets, wrapped, trials, seed0):
    qkv, op = attn.qkv_proj, attn.o_proj
    kh = qkv.weight.shape[1] * 16
    nq, nkv = attn.num_attention_heads, attn.num_key_value_heads
    rd = getattr(attn, "_rope_dim", 0) if attn.use_rope else 0
    B = len(offsets)
    prod = maple._attn_megakernel(
        profile, attn.use_rope, float(attn.scale),
        float(attn._eps), float(attn._rope_log2_base))
    bab, bcd = maple._attn_verify_kernels(
        profile, attn.use_rope, float(attn.scale),
        float(attn._eps), float(attn._rope_log2_base), batch=True)

    hits = {"out": 0, "kv": 0, "ctr": 0}
    for t in range(trials):
        mx.random.seed(seed0 + t)
        kb0 = mx.random.normal((B, nkv, cap, 128)).astype(mx.bfloat16)
        vb0 = mx.random.normal((B, nkv, cap, 128)).astype(mx.bfloat16)
        hns = mx.random.normal((B, kh)).astype(mx.bfloat16)
        mx.eval(kb0, vb0, hns)

        rows_meta = []
        for r, off in enumerate(offsets):
            if wrapped:
                kl = cap
                slot = off % cap
            else:
                kl = min(off + 1, cap)
                slot = off
            rows_meta.append((float(off), float(kl), float(slot)))

        souts, skbs, svbs, sctrs = [], [], [], []
        for r in range(B):
            kb = mx.contiguous(kb0[r:r + 1])
            vb = mx.contiguous(vb0[r:r + 1])
            ctr = mx.zeros((8,), mx.float32)
            mx.eval(kb, vb, ctr)
            ctr[0], ctr[1], ctr[2] = rows_meta[r]
            mx.eval(ctr)
            o, _ = prod(
                inputs=[hns[r].reshape(-1), qkv.weight, qkv.scales,
                        qkv.biases, attn._qk_w, op.weight, op.scales,
                        op.biases, kb, vb, ctr],
                template=[("T_", hns.dtype), ("KH_", kh), ("NQ_", nq),
                          ("NKV_", nkv), ("CAP_", cap),
                          ("ROPE_", 1 if attn.use_rope else 0),
                          ("RD_", rd), ("THREADS_", 1024), ("GRID_", 64)],
                grid=(64 * 1024, 1, 1), threadgroup=(1024, 1, 1),
                output_shapes=[
                    (1, 1, kh),
                    (16 + (nq + 2 * nkv) * 128 + nq * 128 * 2
                     + nq * 32 * 130,)],
                output_dtypes=[hns.dtype, mx.float32], init_value=0)
            mx.eval(o)
            souts.append(o.reshape(1, kh))
            skbs.append(kb); svbs.append(vb); sctrs.append(ctr)
        sout = mx.concatenate(souts, axis=0)

        kbb = mx.contiguous(kb0); vbb = mx.contiguous(vb0)
        live = mx.zeros((3 + 3 * B,), mx.float32)
        mx.eval(kbb, vbb, live)
        for r in range(B):
            live[3 + r * 3 + 0] = rows_meta[r][0]
            live[3 + r * 3 + 1] = rows_meta[r][1]
            live[3 + r * 3 + 2] = rows_meta[r][2]
        mx.eval(live)
        tmpl = [("T_", hns.dtype), ("KH_", kh), ("NQ_", nq),
                ("NKV_", nkv), ("CAP_", cap),
                ("ROPE_", 1 if attn.use_rope else 0), ("RD_", rd),
                ("ROWS_", B), ("BATCH_", 0), ("RAGGED_", 1),
                ("GRID_", 64)]
        scr_shape = (16 + B * ((nq + 2 * nkv) * 128 + nq * 128 * 2 + kh
                               + nq * 32 * (128 + 2)),)
        (scr,) = bab(
            inputs=[hns.reshape(-1), qkv.weight, qkv.scales, qkv.biases,
                    attn._qk_w, kbb, vbb, live],
            template=tmpl + [("THREADS_", 512)],
            grid=(64 * 512, 1, 1), threadgroup=(512, 1, 1),
            output_shapes=[scr_shape], output_dtypes=[mx.float32],
            init_value=0)
        (bo,) = bcd(
            inputs=[scr, op.weight, op.scales, op.biases, kbb, vbb, live],
            template=tmpl + [("THREADS_", 1024)],
            grid=(64 * 1024, 1, 1), threadgroup=(1024, 1, 1),
            output_shapes=[(B, kh)], output_dtypes=[hns.dtype])
        mx.eval(bo)

        hits["out"] += int(mx.array_equal(sout, bo).item())
        kv_ok = True
        ctr_ok = True
        for r in range(B):
            n = int(rows_meta[r][1])
            kv_ok = kv_ok and bool(mx.array_equal(
                skbs[r][0, :, :n, :], kbb[r, :, :n, :]).item())
            kv_ok = kv_ok and bool(mx.array_equal(
                svbs[r][0, :, :n, :], vbb[r, :, :n, :]).item())
            ctr_ok = ctr_ok and bool(mx.array_equal(
                sctrs[r][:3], live[3 + r * 3:3 + r * 3 + 3]).item())
        hits["kv"] += int(kv_ok)
        hits["ctr"] += int(ctr_ok)
    return hits


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
    profile = maple._cuda_profile().name
    sl = next(i for i, t in enumerate(inner.layer_types)
              if t != "full_attention")
    fl = next(i for i, t in enumerate(inner.layer_types)
              if t == "full_attention")
    for li in (sl, fl):
        a = inner.layers[li].self_attn
        if a._qk_w is None:
            a._ensure_qk_constants()

    report = {}
    total = ok = 0
    cases = [
        ("sliding_unwrapped", sl, 512, [37, 120, 260, 490], False),
        ("sliding_wrapped_mix", sl, 512, [700, 900, 300, 480], True),
        ("full_straddle_1024", fl, 4096, [400, 900, 1500, 3000], False),
        ("full_all_2pass", fl, 4096, [1200, 1900, 2600, 3900], False),
        ("sliding_B8", sl, 512, [10, 60, 111, 200, 280, 350, 411, 505],
         False),
    ]
    for name, li, cap, offs, wrapped in cases:
        attn = inner.layers[li].self_attn
        hits = run_case(attn, profile, cap, offs, wrapped,
                        args.trials, 88000 + li * 131 + cap)
        report[name] = {k: f"{v}/{args.trials}" for k, v in hits.items()}
        total += 3 * args.trials
        ok += sum(hits.values())
    report["total"] = f"{ok}/{total}"
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
