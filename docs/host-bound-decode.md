# Decode on CUDA is host-bound

Every optimization in the 0.4.0 release follows from one measurement, so it is
worth stating precisely how it was taken and what it rules out.

## The measurement

`mlx_lm`'s `generate_step` already double-buffers: it builds the graph for step
N+1 while step N runs on the GPU, using `mx.async_eval`. Reproducing that
pattern and timing the three parts separately gives:

| Host | graph build | submission | GPU wait | step | tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| RTX 3090, EPYC 7452 | 2.40 ms | 3.96 ms | **0.0025 ms** | 6.36 ms | 157 |
| RTX 3090, AI Farm | 2.02 ms | 2.76 ms | **0.0028 ms** | 4.78 ms | 209 |

The GPU wait is the time the host spends blocked on the previous step. At
0.003 ms it is not a rounding error in the measurement — it means the GPU has
been idle and waiting since well before the host asked for the result.

A second, independent confirmation: a dedicated RTX 3090 with a slower CPU ran
*slower* than a contended RTX 3090 with a faster one.

## What it rules out

A kernel that makes the GPU faster without removing operations cannot move the
wall clock. The clearest case in this repository: a hand-written 2-bit gathered
QMV reached 384-402 GB/s against 203 GB/s for stock `qmm_naive`, a 1.88x-1.98x
isolated speedup and 1.66x measured inside the model — and end-to-end
throughput moved by roughly nothing.

The corollary is more useful: because the GPU has milliseconds of slack per
step, it is worth *spending* GPU time to buy back host operations. The MoE
megakernel recomputes the residual add and RMSNorm redundantly in every block
purely to avoid needing an extra grid barrier before the router, and that
trade is free — free enough that raising the grid from 32 to 96 blocks, which
triples the redundant work, is worth up to 18% because it finishes the expert
pass sooner. The same logic later paid for a genuine fourth barrier: the tail
phase runs the *next* layer's add+RMSNorm on one block behind it, which costs
GPU microseconds nobody was using and removes a whole host dispatch per layer.

## The cost model

Host cost per operation, measured as median Python construction time plus
`mx.eval` time per op over a tape of 400 independent ops:

| Operation | build | submit | total |
| --- | ---: | ---: | ---: |
| `gather_qmm`, 2-bit | 2.72 | 25.31 | **28.03** |
| `take_along_axis` with `argpartition` | 5.19 | 15.52 | 20.71 |
| router matmul | 2.46 | 10.21 | 12.67 |
| `astype` bf16 to f32 | 1.70 | 9.44 | 11.14 |
| `argpartition`, 256 | 1.79 | 9.26 | 11.05 |
| `softmax` f32, 256 | 1.86 | 7.47 | 9.33 |
| bf16 add | 1.26 | 6.04 | 7.30 |
| `mx.fast.cuda_kernel`, arguments rebuilt | 6.82 | 9.44 | 16.26 |
| `mx.fast.cuda_kernel`, arguments hoisted | 3.86 | 9.35 | **13.21** |

A decode step issues roughly 760 operations, and the sum of these costs matches
the observed step time. Two practical rules follow.

**Hoist the call arguments.** A custom kernel that rebuilds its template list,
grid and output shapes on every call pays 3 us more than one that does not.
Every fusion in this release caches those per layer.

**Count the outputs.** Each additional kernel output costs 5-7 us of host time,
which is the same order as a whole extra dispatch:

| Outputs | host cost |
| ---: | ---: |
| 1 | 8.47 us |
| 2 | 16.50 us |
| 3 | 19.72 us |
| 6 | 35.73 us |

The MoE megakernel initially collapsed three dispatches into one and got *no
faster*, because it had six outputs. Moving the barrier counters, logits,
indices, weights and activations into offsets of a single scratch buffer is
what recovered the win.

## Where the remaining time goes

With the release fusions active, Python graph construction is the larger half
of the step. Exclusive Python time per decode step, measured by instrumenting
the sub-blocks (strict lane, before the megakernel became the default):

| Block | us/step | share |
| --- | ---: | ---: |
| attention | 1248.3 | 57.9% |
| MoE, excluding router | 330.4 | 15.3% |
| add + RMSNorm | 299.0 | 13.9% |
| router | 146.4 | 6.8% |
| rest | 130.2 | 6.0% |

With the megakernel on and its tail phase folding the inter-layer add+RMSNorm
(`benchmarks/maple_fast_lane_profile.py`), the same host's step keeps:

| Block | us/step |
| --- | ---: |
| attention, excluding cache updates | 681.1 |
| megakernel dispatches (24) | 410.0 |
| KV-cache updates (48 scatter assignments) | 333.7 |
| fuse (1 per step, was 25) | 20.0 |

Attention and its cache writes are now two thirds of the remaining host
budget, which is what the next round of work has to attack. The KV row is
pure `self.keys[..., i:i+1, :] = k` bookkeeping — two scatter assignments per
layer per step at ~7 us each.

Attention is the obvious next target, and it resists. Fusing SDPA with the
output projection *regressed*: `RotatingKVCache` returns non-contiguous views,
so MLX materializes a copy of K and V before any kernel declaring
`ensure_row_contiguous` — 10.81 us per sliding layer, about 390 us per step.
Fusing the qkv projection into the per-head kernel also regressed, because the
projection is the dominant read of the block and a per-head grid starves it:
24 blocks on 82 SMs reached 6.6 GB/s against MLX's 384-block `qmv`.

## Other consequences worth knowing

**Extra tokens per pass are cheap but not free.** A forward pass costs 5.08 ms
at one token and 26.35 ms at 32, so a marginal token costs 0.686 ms — 7.4x
leverage. Speculative decoding therefore breaks even at about 2.4 accepted
tokens per pass. An n-gram draft over the context reached 1.02-1.34, and a
layer-skipping self-draft accepted 0 of 573 proposals at every depth tried,
because the model has no early-exit head.

**Batching is a separate lever.** Host time barely grows with batch size, so
aggregate throughput rises to about 523 tok/s at batch 4 against 256 at batch 1.
Past that the MoE weight traffic dominates: eight tokens routed to eight
experts each can touch 64 distinct experts per layer.

**The generation loop is no longer free.** `stream_generate`'s detokenizer and
response objects cost 0.23 ms per token, 7.6% at this speed. A serving path
that does not detokenize every token gains about that much.

**`mx.compile` helps where the chain is elementwise.** A norm-and-projection
chain drops from 48.5 us to 15.1 us under `shapeless=True`, array-equal. The
router chain, dominated by a matmul, a sort and a gather, drops only from
96.5 us to 77.9 us. `shapeless=True` cannot wrap a custom kernel or a top-k
slice.

## Reproducing

The probes are in `benchmarks/`; each prints one JSON line per configuration.
Measure on a host with a quiet CPU — `/proc/loadavg` before, not after — and
report medians over fresh processes, because a shared CPU moves this workload
far more than a shared GPU does.
