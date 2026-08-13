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
