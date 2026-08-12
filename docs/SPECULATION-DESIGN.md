# Free-MMA-row speculation: design notes

Status: foundation proven, not yet built. This documents what is settled
and what remains, so the next cycle starts from evidence rather than
memory.

## The settled part

`benchmarks/maple_mma_row_independence.py` (sm89, 2026-08-12): filling
rows 1..15 of the `m16n8k16` bf16 atom does not move row 0's bits, and
every filled row is bit-equal to the same activation run alone in row 0
(16/16). Therefore a verify pass over L ≤ 16 draft tokens through the
expert MMA **is** the L sequential M=1 passes, bit for bit — speculation
does not have to trade away the project's bit-exactness invariant at the
expert phase, and the expert GEMM cost of verifying L tokens is one pass,
not L.

The wall is GPU time now, not host time (the chain negative result: the
stock async double-buffer already hides host build behind execution), so
tokens-per-pass is the remaining lever with real headroom.

## What verification needs per phase, for a pack of L drafts

- **Experts (the big cost)**: free — the same qmm_naive-recipe MMA with
  rows 1..L filled. Gathered experts differ per token though: each token
  routes its own top-8, so the gather is L×8 expert blocks; tokens
  sharing an expert can share the pass (the win shrinks toward L separate
  passes in the worst case — measure the overlap on real streams).
- **Router**: L RMS-normed hiddens through the fp32 gemv + softmax +
  top-8 + renorm chain. Cheap arithmetically; the exact recipes are all
  pinned and per-token independent, so an L-row port keeps bits by
  construction.
- **Attention qkv/o_proj**: the 2-bit `qmv` recipe is one-row; an L-row
  variant is L warpsets of the same recipe in one dispatch (dispatch cost
  ×1, GPU cost ×L on those phases — they are small).
- **SDPA**: draft token i attends to the cache plus drafts 0..i-1 —
  causal inside the pack. The 1-pass port handles it with per-row kL
  offsets; the pack's K/V live in registers/smem before being committed
  on acceptance.
- **Cache commit**: only accepted tokens' K/V are appended; the persistent
  buffers and the seed-kernel discipline already support exact rollback
  (write happens post-acceptance, or writes land and the counters simply
  do not advance past the accepted prefix — choose during implementation).
- **lm_head + acceptance**: L logits rows, argmax each; accept the
  longest prefix where draft[i] == argmax[i], emit argmax at the first
  mismatch (greedy correctness = the sequential stream by construction,
  given every phase above is bit-exact per row).

## Draft sources, measured so far

The earlier prototype measured n-gram acceptance at 1.02-1.34 tokens per
pass on this model's decode streams — near break-even but thin.
Prompt-lookup (copy spans from the prompt/context) is the cheapest
next candidate and shines exactly where Maple serves (structured/agent
traffic with heavy copying); a self-draft head would need training and is
out of scope.

## Break-even sketch (sm89 numbers)

The two-dispatch step runs ~2.2 ms/token at 455.7 tok/s. A verify pass
for L=4 adds roughly: experts ≈ free to +30% (gather overlap dependent),
attention/router phases ×L on ~15% of the step. Call it 1.2-1.5× a single
step; acceptance of A tokens/pass yields A/1.35 ≥ 1 ⇒ **A ≥ ~1.4 pays**.
n-gram sits below that on average; prompt-lookup on agent traffic is the
bet worth measuring first — instrument acceptance on real service logs
before building the kernels.

## First acceptance numbers (2026-08-12)

Five prod-generated agent streams through the offline simulator
(`benchmarks`-adjacent `acceptance_lab.py`; grid k∈{2,3,4} × L∈{4,8,16}):
best mean 1.678 tokens/pass at k=2, L=16 — carried by copy-heavy traffic
(code refactor: **4.11**, JSON extract 1.27) while quoting/freeform sit at
~1.0. Above the ~1.4 break-even exactly where Maple serves agent loops;
the corpus is small, so the next measurement round should be dozens of
real streams before kernel work starts. Side effect of collecting them:
the cross-request isolation incident (chronicle #17) — found, mitigated,
and gated behind the LRU repro.

## Order of work for the next cycle

1. Extend the acceptance corpus to dozens of real service streams (the
   collection+tokenize+simulate pipeline exists).
2. If A clears ~1.5 on real traffic: build the L-row router/qmv ports and
   pin their bits (the recipes are per-token independent — expect clean
   ports).
3. Pack-causal SDPA port; acceptance loop in `_decode_fused`; the cache
   commit protocol.
4. The megakernel M=L variants ride the existing templates (`ROWS_`
   parameter), grid unchanged; bit gates extend the existing suites.
