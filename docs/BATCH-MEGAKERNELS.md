# Batch megakernels (B ≤ 8): the work map

Measured on sm89 (sync loop): B=1 runs the fused lanes at 289 aggregate;
B=2 falls to 236 — BELOW B=1 — because everything fused gates on a
single vector; stock scales to 472@4 and 642@8. Generalizing the two
megakernels to small batches multiplies the fused rate across streams.

## The single gate

`MapleModel.__call__` enters the fused decode only when
`h.size == h.shape[-1]` (B=1 ∧ S=1). The batch entry is
`h.shape[-2] == 1 and h.shape[0] <= 8` with shapes (B, 1, KH) carried as
(B, KH) through `_decode_fused`.

## Why the bits are already proven

Every per-row recipe is pinned row-independent: router 48/48, qmv 48/48
(`maple_lrow_semantics.py`), expert MMA rows 16/16
(`maple_mma_row_independence.py`), the exact split/RoPE and 1-pass SDPA
per row (the verify pair, 108/108 vs sequential production dispatches).
A batch row is bit-wise the same computation as a lone B=1 stream at the
same offset — the batch gates compare each row against its own solo run.

## Attention M=B — simpler than the verify pair

The mlx-lm batch cache is ONE object with a leading B axis and a SHARED
offset/_idx: every row appends into the same slot. So attention M=B is
the verify pair with kL CONSTANT across rows (no per-row kl0+r), buffers
(B, kvh, cap, 128), the state's counters unchanged (one pos/kl/slot for
all rows), and phase C's task grid (head × row). The AB/CD split already
carries ROWS_; deriving BATCH kernels is mostly deleting the per-row kL
logic and adding the B axis to the cache pointers. Rollback and
LRU-materialization semantics carry over unchanged (one set of counters).

## Exact MoE M=B

Stateless — scratch regions gain a ROWS_ factor and the phases become
(row-major) task grids exactly like the attention generalization:
- router: B rows through the pinned fp32 gemv/softmax/top-8/renorm;
- experts: the gather covers all B rows' top-8 (up to 8B expert blocks);
  rows sharing an expert ride the free MMA rows of one pass — for
  INDEPENDENT streams sharing is irrelevant to correctness, purely a
  bandwidth amortization (this is where the batch win lives: one weight
  read serves every subscribed row);
- SwiGLU/aggregation/tail norms: per-row, template-scaled.

## Gates before default

Per-B bit suites (each row vs its solo stream), the batch curve re-run
(the target: B=4 aggregate well above stock's 472), the LRU/service
suites at B>1, and per-arch defaults only where measured faster —
the standing discipline applies unchanged.

## Exact-MoE M=B: the concrete decomposition (post-attention-proof)

Attention M=B is proven (108/108, `maple_attn_batch_check.py`) — with a
battle scar worth its own line: a fresh compilation contracted the
upper-half RoPE differently from production sm86 (1 ULP, deterministic),
so the batch/verify pair now pins the form per profile. Expect the same
class of pin work anywhere a recipe expression is re-compiled in a new
kernel body.

The MoE kernel's phases map as follows (current scratch:
idx[8], sco[8], logits[NROUT], probs[NROUT], ugstage[NEXP*2*KD], dstage):

- **Phase 0 (add+RMS) and A (router gemv)**: per-row by construction —
  task grids gain a ROWS_ factor, scratch regions gain a leading B.
- **Phase B (softmax/top-8/renorm)**: currently block-0-only; becomes
  one 64-thread sub-block per row (the pinned recipe is 64 threads), B
  rows across the first ceil(B/16) blocks, plus a NEW tail: build the
  DEDUP table — a 256-slot bitmask (uint32 per expert, bit r = row r
  subscribes) reduced with atomicOr in scratch, then a compacted list
  (unique expert id, subscriber mask) via a prefix scan on block 0.
- **Phase C/D (experts)**: iterate the COMPACTED unique list instead of
  the fixed 8: one weight read per unique expert serves every
  subscriber row through the free MMA A-rows (row r of the atom = the
  r-th subscriber's activation; the row-independence pin makes each
  subscriber's bits equal its solo run). Worst case 8B uniques degrades
  gracefully to per-row cost; the measured overlap (0.68-0.75 at L=4
  windows) does not apply to independent streams — expect near-zero
  sharing across unrelated requests and near-full sharing for
  same-prompt fan-out (n-best, self-consistency), which is exactly the
  serving shape that wants batch anyway.
- **Phase E (aggregate + residual + next norm)**: per-row task grids.

Order: land 0/A/B(+dedup) as one kernel with bit gates against B solo
runs; C/D as the second (the gather is the only genuinely new code);
E rides the existing recipe. Then the gate swap in `MapleModel.__call__`
(`h.shape[-2] == 1 and h.shape[0] <= 8`), per-B suites, and the curve.

## Status burndown (2026-08-13)

- [x] Batch attention M=B: **108/108** bit-proven (`maple_attn_batch_check.py`),
  RoPE contraction pinned per profile.
- [x] Batch exact-MoE M=B: **162/162** bit-proven kernel-vs-kernel
  (`maple_moe_batch_check.py`) — out, hout and the routing tail across
  layers 0/3/7 x B in {2,4,8} x 6 seeds. No contraction pins needed.
- [x] Per-layer perf probes (sm86, sync loop, graphs off — ratios are the
  signal): MoE batch beats B sequential megakernels x1.7–2.0 AND the stock
  batched MoE by −22% (B4) / −19% (B8); attention pair beats sequential by
  −5% (B2) → −36% (B8).
- [ ] E2E wiring: batch cache state (B-plane persistent buffers, shared
  counters), the `MapleModel.__call__` gate swap, batched lm_head tail.
- [ ] Per-B service suites (LRU isolation at B>1) before any default.
- [ ] The end-to-end curve — the judge is aggregate tok/s vs stock
  (289/472/642 at B=1/4/8 on sm89).
