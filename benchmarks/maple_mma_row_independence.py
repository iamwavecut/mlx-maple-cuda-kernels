"""The make-or-break probe for free-MMA-row speculation.

The exact expert recipe feeds one activation row into the m16n8k16 bf16
MMA atom (rows 1..15 zero). Speculative verification wants rows 1..L-1
filled with OTHER tokens' activations. The whole idea lives or dies on
one question: does filling those rows change row 0's bits?

PTX semantics say no — D = A@B + C is computed per-row with no cross-row
reduction — but the recipe's bits were pinned empirically, so this is
pinned empirically too: run the same k-ordered tile loop twice, rows
1..15 zeroed vs filled with random activations, and compare row 0
bitwise. Also compare each filled row r against the SAME recipe run with
that row's activation placed in row 0 alone — if equal, a speculative
verify pass IS the sequential M=1 pass, bit for bit, for every draft
position at once.

    python lab/mma_row_independence.py
"""
import json

import mlx.core as mx

SRC = r"""
    // One warp per output tile of 8 columns; K_ is a multiple of 128.
    // A-rows: ROWS_ activations at stride K_; rows >= ROWS_ read zeros.
    const int lane = threadIdx.x & 31;
    const int tile = blockIdx.x;          // column tile: 8 outputs
    const int col0 = tile * 8;

    // fragment accumulators for m16n8k16: each lane holds 4 floats
    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    for (int k0 = 0; k0 < K_; k0 += 16) {
        // A fragment: m16n8k16.row.col A is 16x16 bf16: lane l supplies
        // a0,a1 (rows l/4, cols ...) per the PTX layout. Build via ldmatrix
        // semantics manually: a[r][c] = act[r*K_ + k0 + c] for r < ROWS_.
        unsigned a0, a1, a2, a3;
        {
            const int r = lane >> 2;          // 0..7
            const int c = (lane & 3) * 2;     // 0,2,4,6 pairs
            __nv_bfloat16 v0, v1, v2, v3, v4, v5, v6, v7;
            v0 = (r < ROWS_) ? act[r * K_ + k0 + c] : __nv_bfloat16(0.f);
            v1 = (r < ROWS_) ? act[r * K_ + k0 + c + 1] : __nv_bfloat16(0.f);
            v2 = (r + 8 < ROWS_) ? act[(r + 8) * K_ + k0 + c] : __nv_bfloat16(0.f);
            v3 = (r + 8 < ROWS_) ? act[(r + 8) * K_ + k0 + c + 1] : __nv_bfloat16(0.f);
            v4 = (r < ROWS_) ? act[r * K_ + k0 + c + 8] : __nv_bfloat16(0.f);
            v5 = (r < ROWS_) ? act[r * K_ + k0 + c + 9] : __nv_bfloat16(0.f);
            v6 = (r + 8 < ROWS_) ? act[(r + 8) * K_ + k0 + c + 8] : __nv_bfloat16(0.f);
            v7 = (r + 8 < ROWS_) ? act[(r + 8) * K_ + k0 + c + 9] : __nv_bfloat16(0.f);
            a0 = (unsigned(__bfloat16_as_ushort(v1)) << 16) | __bfloat16_as_ushort(v0);
            a1 = (unsigned(__bfloat16_as_ushort(v3)) << 16) | __bfloat16_as_ushort(v2);
            a2 = (unsigned(__bfloat16_as_ushort(v5)) << 16) | __bfloat16_as_ushort(v4);
            a3 = (unsigned(__bfloat16_as_ushort(v7)) << 16) | __bfloat16_as_ushort(v6);
        }
        // B fragment: 16x8 bf16, col-major per row.col: b[k][n] = w[(col0+n)*K_ + k0+k]
        unsigned b0, b1;
        {
            const int n = lane >> 2;
            const int k = (lane & 3) * 2;
            __nv_bfloat16 w0 = wt[(col0 + n) * K_ + k0 + k];
            __nv_bfloat16 w1 = wt[(col0 + n) * K_ + k0 + k + 1];
            __nv_bfloat16 w2 = wt[(col0 + n) * K_ + k0 + k + 8];
            __nv_bfloat16 w3 = wt[(col0 + n) * K_ + k0 + k + 9];
            b0 = (unsigned(__bfloat16_as_ushort(w1)) << 16) | __bfloat16_as_ushort(w0);
            b1 = (unsigned(__bfloat16_as_ushort(w3)) << 16) | __bfloat16_as_ushort(w2);
        }
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }

    // write the 16x8 fp32 result: lane layout of D matches C
    {
        const int r = lane >> 2;
        const int c = (lane & 3) * 2;
        out[(long long)(r) * N_ + col0 + c] = acc[0];
        out[(long long)(r) * N_ + col0 + c + 1] = acc[1];
        out[(long long)(r + 8) * N_ + col0 + c] = acc[2];
        out[(long long)(r + 8) * N_ + col0 + c + 1] = acc[3];
    }
"""


def run(kern, act_rows, K, N):
    (out,) = kern(
        inputs=[act_rows.reshape(-1), W.reshape(-1)],
        template=[("K_", K), ("N_", N), ("ROWS_", act_rows.shape[0])],
        grid=((N // 8) * 32, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(16, N)], output_dtypes=[mx.float32])
    mx.eval(out)
    return out


K, N = 2048, 512
mx.random.seed(777)
W = (mx.random.normal((N, K)) * 0.05).astype(mx.bfloat16)
acts = mx.random.normal((16, K)).astype(mx.bfloat16)
mx.eval(W, acts)

kern = mx.fast.cuda_kernel(
    name="mma_row_probe", input_names=["act", "wt"], output_names=["out"],
    source=SRC)

report = {}
# 1) row-0 invariance: rows 1..15 zero vs filled
solo = run(kern, acts[:1], K, N)
full = run(kern, acts, K, N)
report["row0_invariant_when_rows_filled"] = bool(
    mx.array_equal(solo[0], full[0]).item())

# 2) every filled row r == the same recipe with that activation in row 0
per_row = []
for r in range(16):
    alone = run(kern, acts[r:r + 1], K, N)
    per_row.append(bool(mx.array_equal(alone[0], full[r]).item()))
report["rows_equal_their_solo_M1"] = per_row
report["all_rows_bitexact"] = all(per_row)
print(json.dumps(report), flush=True)
