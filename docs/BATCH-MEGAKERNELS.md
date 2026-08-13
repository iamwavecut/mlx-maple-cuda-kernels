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

## E2E wiring round 1 (2026-08-13): lane landed opt-in, gate is RED

`MAPLE_BATCH_MEGAKERNELS=1` wires both proven kernels behind the stock
per-layer structure (`_decode_batch_fused`, `_attn_mega_call_batch`,
`_moe_batch_call`); default stays off. The E2E gate
(`maple_batch_e2e_check.py`) prefills each row SOLO, merges the caches on
the batch axis, then requires every batched-decode row to equal its solo
stream. Verdict: RED — and the control row is the discovery: **stock
batched decode fails the same gate** (2/4 at B4, 4/8 at B8), so the
batched tails outside our kernels (lm_head, embeddings) are not
row-independent at the bit level. Our lane is additionally worse (0/4 at
B4) — a wiring bug of its own on top. Aggregate tok/s (graphs off):
mega 262/341/294 vs stock 231/370/344 at B=2/4/8.

Debug order: (1) microscope one diverging row at B=2 layer by layer
(batch-vs-solo capture, the lru_microscope pattern) to find the first
diverging stage; (2) decide the tail strategy — per-row lm_head loop
(cheap at B<=8) vs a row-exact batched port; (3) re-gate, then the curve
with graphs on stock-side.

### Debug session 1 (2026-08-13, evening)

The kernels are exonerated on real state (three-way compare: stock ==
production == pair, 0 diffs). Two real culprits found so far: the
STOCK lm_head qmm is not row-invariant at B=8 (bit-level, all rows,
maxabs 0.0625) — so the solo-exact contract needs a per-row head tail
at B=8 regardless of our kernels; and `_attn_mega_call` needed a
`state.rows` guard (hygiene, landed). B=2/4 still diverge somewhere
past step 8 — the 8-step microscope on the same prompts is clean, so
the next tool is a 48-step per-layer capture to localize the first
diverging (step, layer, stage). Lane stays opt-in.

## E2E PASS (2026-08-13 night): solo-exact batching, all B

Three root causes later (per-row boundary fuse — the solo
`_exact_add_rms_norm` is neither pair-equal nor row-safe; per-row
lm_head past M=4 — the stock qmm head loses bit row-invariance at M=8;
the `state.rows` guard), `maple_batch_e2e_check.py` is green: **every
batched row reproduces its solo stream bit-for-bit** at B=2/4/8 over 48
steps. The stock batch path cannot make that promise on its own rows
(2/4 at B4, 4/8 at B8). Aggregate tok/s (graphs off): 266/300/364 vs
stock 252/386/322 — correctness landed, throughput is the open front
(graph capture for the AB/CD pair, then the curve vs stock-with-graphs
472 @ B4).

### Grid tunes + graphs (2026-08-13, late night)

The lane runs under CUDA graphs (the old AB/CD capture failure does not
reproduce end-to-end), and both kernels gain from bigger grids on sm86:
the pair 290→272µs (B4) / 490→438µs (B8) at grid 80, batch MoE 241→216 /
385→340µs at grid 128. Wired as `MAPLE_BATCH_ATTENTION_GRID` /
`MAPLE_BATCH_MOE_GRID` (defaults stay safe: the batch MoE kernel is
register-heavier and DEADLOCKS at the production-safe 192 on 82 SMs —
residency is per-kernel, not per-source-family). Tuned E2E aggregate:
260/331/373 tok/s vs stock 279/378/342 at B=2/4/8, bits still PASS —
B8 beats stock +9% on a noisy host. Remaining: per-B LRU suites, a
clean-host curve, then the default decision.

### LRU isolation gate (2026-08-13): PASS

`maple_batch_lru_check.py`: a stored solo history survives a full batch
intervention on the same modules and continues bit-equal to an
uninterrupted run — the materialize-on-detach discipline holds through
row-count rebinds. Remaining before any default: a clean-host curve
(GPU0 neighbor makes farm walls noisy) and multi-arch scale-out.

## sm120 scale-out (2026-08-13, rented RTX 5090, quiet host)

Bits: the full battery is green — attention pair all 6/6 (the predicted
`second_half_form=0` pin held on first try), MoE 162/162, E2E solo-exact
2/2 4/4 8/8, LRU PASS. The stock control collapses here: its own batched
rows match solo only 1/2, 3/4, **2/8**. Grids scale much further on 170
SMs — attention 302→168µs (B8, g64→g160), MoE 277→148µs (g64→g256) —
and with them the lane **beats or matches stock at every B**: curve
333/500/853/930 vs stock 308/495/697/918 (B4 **+22%**). Per-profile safe
grid defaults landed (sm100/sm120 → 80/160, clamped to the smallest
class member); big hosts opt higher via the env tunes. Ops scar for the
book: CUDA-12.4 pod images + the new MLX wheel hit the ≥12.8-headers
nvrtc trap — pip `nvidia-cuda-runtime-cu12==12.9.*` + a CUDA_HOME
symlink fixes it without a toolkit install.

### 2-pass in the pair (2026-08-13, late): kL <= 1024 scope lifted

The CD kernel carries the production 2-pass slab recipe behind a
per-row algorithm pick — full-attention layers in the batch lane now
grow 1024→8192 like B=1 instead of dropping to stock past 1024. Gate:
72/72 (pure 2-pass at kL≈3000 + 1-pass control, B=2/4/8). The port
surfaced a latent race: AB consumes all four barrier-counter pairs of
the shared scratch, so CD's old C→D barrier was a pass-through that had
been green by luck — both CD barriers moved to fresh slots. Full
battery re-run green.
