# The full record: mechanics, protocol, chronicle, limitations

Moved out of the README to keep it short; nothing here is superseded.
Evidence lives under [`../results/`](../results/).

## Why: decode on CUDA is host-bound

The measurement that reframed this work: with a warm cache and MLX's
double-buffered `async_eval`, the **GPU wait per decode step is ~0.003 ms**.
The GPU finishes before Python has finished building the next step, on every
host tested. Wall clock is the sum of per-operation host costs, so the
currency of optimization is operation count, not arithmetic.

| Host | graph build | submission | GPU wait | step |
| --- | ---: | ---: | ---: | ---: |
| RTX 3090, EPYC 7452 | 2.40 ms | 3.96 ms | **0.0025 ms** | 6.36 ms |
| RTX 3090, AI Farm | 2.02 ms | 2.76 ms | **0.0028 ms** | 4.78 ms |
| RTX 4090 | 2.43 ms | 3.31 ms | **0.0029 ms** | 5.76 ms |
| H100 80GB HBM3 | — | — | **0.0040 ms** | — |
| B200 | — | — | **0.0030 ms** | — |
| RTX 5090 | — | — | **0.0022 ms** | — |

Kernels that make the GPU faster without removing operations do nothing (a
hand-written 2-bit expert GEMV at 1.88-1.98x in isolation moved end-to-end
throughput by roughly nothing), and it is worth spending GPU time to buy
back host operations — which is what both megakernels do. Related, all on
RTX 3090: streaming reads reach 743 GB/s while the stock expert projections
run at 203 GB/s; a forward pass costs 5.08 ms at one token and 26.35 ms at
32, so a marginal token costs 0.686 ms (7.4x leverage, speculative decoding
breaks even at ~2.4 accepted tokens); aggregate throughput peaks near batch
4 (523 tok/s vs 256 at batch 1); `stream_generate`'s detokenizer costs
0.23 ms per token. Full write-up:
[`docs/host-bound-decode.md`](host-bound-decode.md).

## The lanes

### The exact MoE megakernel

One dispatch per MoE layer: router logits, online softmax, top-8 with the
stock argpartition tie order, renorm, eight gathered 2-bit experts, clamped
SwiGLU, score-weighted aggregation, the surrounding residual add/RMSNorm and
the next layer's carrier — five phases behind four atomic-counter grid
barriers.

Every phase is a bit recipe pinned against the stock kernel it replaces
(`benchmarks/maple_qmm_naive_repro.py`,
`benchmarks/maple_exact_lane_semantics.py`): dequant as `bf16(bf16(q·s)+z)`
into the same `m16n8k16` bf16 tensor-core atom with k-tiles of 128 in order;
the fp32 logits gemv; an exact online-softmax port; argpartition's returned
order (argsort's ascending tail, ties included); the renorm reduce whose
**shape picks the kernel** — a `(1,1,8)` sum dispatches `row_reduce_simple`,
a flat `(8,)` an all-reduce with different bits; and the aggregation as
`col_reduce_small`'s linear loop with `__fmul_rn` separate from `__fadd_rn`,
where letting the compiler contract to fma is exactly what breaks equality.

Exact vs the ~1 ULP lane it replaced as default, medians over five fresh
interleaved processes per mode:

| GPU | CC | strict | ~1 ULP MK | exact MK | exact stream |
| --- | --- | ---: | ---: | ---: | --- |
| RTX 3090 | `sm86` | 176.3 | 341.1 | **345.2** | 8/8 identical |
| RTX 4090 | `sm89` | 175.8 | 318.9 | **320.3** | 8/8 identical |
| H100 80GB | `sm90` | 233.5 | 395.3 | 388.6 | 8/8 identical |
| B200 | `sm100` | 322.1 | 389.3 | 358.0 | 8/8 identical |
| RTX 5090 | `sm120` | 217.7 | 399.3 | 381.6 | 8/8 identical |

Parity on consumer Ampere/Ada, within 1.7% on H100, 4.4% on RTX 5090 and
8.0% on B200 — and +64% to +96% over the strict lane everywhere, with the
846-token quality suite reproducing the strict lane's corpus NLL to the
last digit on every row.

### The attention megakernel

The second dispatch: the whole decode attention block. Phase A computes the
fused 2-bit qkv projection with the stock `qmv` recipe (bf16 HFMA
accumulators, one packed word and one scale/bias pair per 128-tile); phase B
splits heads, applies Q/K RMSNorm and partial RoPE (only the first 64 of 128
dims rotate) line-for-line with the shipped exact split kernel and appends
K/V into persistent caller-owned buffers at the same physical slot the stock
rotating cache uses; phase C is the stock SDPA — the `kernel_sdpav_1pass`
port up to 1024 keys (12/12 bitwise at five context lengths), and past that
the `kernel_sdpav_2pass` port (32 slabs of 8 warps per head, fp32 partials
scaled to the slab max, one extra grid barrier, a 32×32 merge block per
head; standalone recipe 72/72 vs stock at kL 1025-8192); phase D is the
o_proj `qmv`.

Full-attention layers grow their buffers 1024 → 2048 → 4096 → 8192 (one
recompile per tier) and fall back past 8192. Sliding-window layers mirror
the stock ring exactly, including the concat-tail state a multi-token
prefill leaves on a rotated ring. The kernel advances its own on-device step
counters, and every (re)seed is written by a helper kernel through the input
pointers — the persistent buffers never move, so CUDA graphs capture once.
Leaving the lane (a new prefill, a fallback) flushes the fused buffers back
into the stock cache; re-entry re-seeds from it.

Measured on clean hosts (paired within one process, stream 4/4 prompts
identical in every configuration):

| GPU | CC | exact MoE only | + attention MK | gain |
| --- | --- | ---: | ---: | ---: |
| RTX 4090 | `sm89` | 318.8 | **455.7** | **+42.9%** (spread 454.1-455.8) |
| RTX 3090 | `sm86` | — | — | **+20.6%** graphs on / +13.3% off (shared host, paired) |

Bit evidence on `sm86`: 4/4 stream prompts over 256 tokens, ring rotation
at 700 tokens 2/2, the 1024-boundary suite (cross-in-flight, start-past,
write-back + regrow, 4096-tier, 2048→4096 growth over 260 tokens) all
identical, multi-turn chains bit-equal, per-layer live probes bit-equal on
all 24 layers; `sm89` reproduces the stream suites.

### The ~1 ULP fallback and what it costs

Geometries the exact plan declines run the earlier megakernel, within
~1 ULP of bf16. Token equality does not survive that (near-tie argmax flips
on ~9% of scored tokens), so the question is quality, measured rather than
assumed on 846 scored tokens across five architectures
(`benchmarks/maple_quality_suite.py` — twelve documents: prose, dialogue,
code, SQL, structured text, arithmetic, scored one token at a time against
a cache so the fused decode kernels are actually exercised):

| GPU | reference ppl | strict | megakernel | top-1 changed |
| --- | ---: | --- | ---: | ---: |
| RTX 3090 / 4090 | 33.1857 | identical | 32.8173 (−1.1%) | 78 / 846 |
| H100 80GB | 32.9462 | identical | 32.6704 (−0.8%) | 80 / 846 |
| RTX 5090 / B200 | 33.1647 | identical | 32.7340 (−1.3%) | 75 / 846 |

Perplexity *lower* by 0.8-1.3% is the signature of unbiased last-bit noise,
not a better or worse model; what it rules out is a quality regression.

### The strict fusions

`MAPLE_MOE_MEGAKERNEL_EXACT=0 MAPLE_MOE_MEGAKERNEL=0` steps down to the two
array-exact fusions alone — residual add + RMSNorm (+3.3% as a component
median; the `mx.fast.rms_norm` thread mapping is what makes it exact) and
the fused QKV split (+2.2%; bit-identical by construction), together +7.0%.
Each runs a live comparison at first use and switches itself off on
mismatch.

### Opt-in

`MAPLE_COMPILED_ROUTER=1` compiles the stock router chain (array-exact,
cuts the router's host cost 96.5→77.9 µs per layer, end-to-end 1.0062x with
a 95% interval of 0.9927-1.0198 — indistinguishable from zero, and the
megakernel absorbs the router anyway). `--cached-lhs` caches the flat
decode LHS; exact in campaign but process-global and keyed only by top-k,
so it stays opt-in. Every lane is a module attribute
(`maple._use_moe_megakernel` and friends); the environment only seeds them
at import. Because the megakernel rides the fused add/RMSNorm carrier,
`MAPLE_FUSED_ADD_RMS=0` switches it off too — that is how `off` is
measured in every table here.

## Exactness protocol

The stock path is **not always reproducible run to run**: on some hosts six
identical runs produced two different token streams, always diverging at the
same position, with or without CUDA graphs. Forty-repeat probes of the
prefill forward, a single decode step, the router, the MoE block, attention
and RMSNorm found zero differences, so it is not a race inside any one
operation and the source is not localized.

Every equivalence verdict here therefore screens first: each prompt is
generated three times with the fusions off, and a candidate counts as
divergent only inside the region where those three runs agree. Under that
protocol the reference was stable on all eight prompts, `strict` matched on
8/8 and the ~1 ULP megakernel on 1/8. Disabling cuDNN SDPA is required to
make the stock oracle bit-stable and is not credited as a speedup.

## Chronicle

Linear history of what landed, oldest first, with the numbers each stage
was accepted on. Backed by records under [`results/`](../results/).

### 1. 2026-08-06 — Q/K norm + partial RoPE/NoPE fusion (`sm86`)

The first array-exact fused path and the fail-closed probing scheme;
strict-exact kernel profile published for RTX 3090 (v0.3.0/v0.3.1 era).

### 2. 2026-08-07 — the strict multi-arch matrix

Five fresh NVIDIA targets under sealed campaign revisions; 12 fresh model
processes per device; exactness gated by `mx.array_equal` on live fused
outputs (all 24 Q/K layers, no silent fallback), 144 deterministic W2
fingerprints, exact 1024-token outputs, multi-seed matrix, 20/20 fixed
regressions at 512 and 1024 tokens:

| GPU | CC | Portable | Exact Q/K default | Paired gain (95% CI) | + cached-LHS opt-in |
| --- | --- | ---: | ---: | ---: | ---: |
| RTX 3090 | `sm86` | 145.89 | **154.95** | **+6.06%** (+2.73%–+9.50%) | 159.73, **+9.37%** |
| RTX 4090 | `sm89` | 182.36 | **209.84** | **+15.31%** (+11.29%–+19.49%) | 214.66, **+18.01%** |
| H100 80GB | `sm90` | 202.62 | **233.67** | **+15.24%** (+12.86%–+17.66%) | 246.34, **+21.54%** |
| B200 | `sm100` | 241.51 | **280.70** | **+16.28%** (+14.23%–+18.37%) | 297.47, **+23.37%** |
| RTX 5090 | `sm120` | 398.49 | **429.72** | **+7.84%** (+6.88%–+8.81%) | 438.01, **+9.92%** |

Index: [`results/summary.csv`](../results/summary.csv).

### 3. 2026-08-07 — the Blackwell RoPE rounding fix

On `sm100`/`sm120` the fused upper-half RoPE could contract the opposite
product from stock; one FP32 rounding bit could cross a BF16 midpoint. The
profiles now pin `__fmaf_rn(value, rope_cos[p], __fmul_rn(paired,
rope_sin[p]))`. B200 passed 2,048 isolation comparisons with zero fixed
mismatches (32 old-control), RTX 5090 an expanded 4,608 with zero (63
old-control); no statistically significant slowdown on either. Frozen
fixture:
[`tests/data/sm100_qk_rope_boundary.npz`](../tests/data/sm100_qk_rope_boundary.npz)
— re-run after any MLX or CUDA upgrade.

### 4. 2026-08-11 — the host-bound diagnosis

GPU wait per decode step is ~3 µs on every host from a 3090 to a B200 (the
table in [Why](#why-decode-on-cuda-is-host-bound)); wall clock is per-op
host cost. Reframed the effort toward dispatch-count reduction. Also from
this study: the graph profile (cache 400, 100 ops / 100 MB per buffer) was
supported on RTX 4090 (+25.07%, `p=0.0148`) and RTX 5090 (+16.31%,
`p=1.2e-5`); the B200 factorial ops effect was +10.90% (`p=6.8e-5`); H100
graph effects were inconclusive on its single instance.

### 5. 2026-08-11 — the exact fusions promoted

Residual add+RMSNorm and the fused QKV split made default after matching
the stock thread mappings bit for bit; together +7.0% as component medians,
stream identical on 8/8 screened prompts on every architecture:

| GPU | CC | off | strict | Paired gain (95% CI) | Wins |
| --- | --- | ---: | ---: | ---: | ---: |
| RTX 3090 | `sm86` | 152.29 | **164.94** | **+6.68%** (+4.33%–+9.09%) | 8/8 |
| RTX 4090 | `sm89` | 157.66 | **184.25** | **+16.91%** (+15.27%–+18.56%) | 6/6 |
| H100 80GB | `sm90` | 206.75 | **226.88** | **+9.32%** (+2.70%–+16.36%) | 5/6 |
| B200 | `sm100` | 141.87 | **165.19** | +11.77% (−4.35%–+30.61%) | 5/6 |
| RTX 5090 | `sm120` | 242.61 | **269.34** | **+10.63%** (+8.84%–+12.46%) | 6/6 |

### 6. 2026-08-11 — the ~1 ULP MoE megakernel

The whole MoE block in one dispatch behind grid barriers, measured at the
originally shipped fixed grid of 32 blocks:

| GPU | CC | megakernel | Paired gain (95% CI) | Wins |
| --- | --- | ---: | ---: | ---: |
| RTX 3090 | `sm86` | **272.80** | **+79.51%** (+76.18%–+82.91%) | 8/8 |
| RTX 4090 | `sm89` | **292.15** | **+87.04%** (+81.01%–+93.26%) | 6/6 |
| H100 80GB | `sm90` | **361.72** | **+76.11%** (+67.86%–+84.77%) | 6/6 |
| B200 | `sm100` | **254.37** | **+73.86%** (+47.35%–+105.14%) | 6/6 |
| RTX 5090 | `sm120` | **426.66** | **+75.35%** (+71.10%–+79.70%) | 6/6 |

At 2048 generated tokens the ordering holds everywhere and peak memory does
not grow (`sm86`: 154.75 / 163.51 / 274.72 tok/s at 6.478 GB). The quality
suite (table in [the fallback section](#the-1-ulp-fallback-and-what-it-costs))
shows no regression.

### 7. 2026-08-11 — the per-device grid retune

The grid barrier is correct only while every block is resident, and MLX
exposes neither SM count nor occupancy, so the block count is selected from
compute capability and memory. Medians for the fast lane, four fresh
processes per point:

| GPU | 32 | 64 | 96 | 128 | 192 | shipped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX 3090 | 357.5 | **375.0** | 365.3 | 370.2 | 333.9 | 64 |
| RTX 4090 | 429.1 | 469.1 | **507.7** | 478.5 | 463.3 | 96 |
| H100 80GB | 344.8 | 359.0 | **365.9** | 358.4 | 356.2 | 96 |
| RTX 5090 | 422.9 | 424.9 | **435.3** | 427.2 | 426.7 | 96 |
| B200 | 293.4 | 327.9 | 324.9 | 332.1 | **335.4** | 192 |

A fixed 32 costs 2.9% on RTX 5090 and 18.3% on RTX 4090.
`MAPLE_MOE_MEGAKERNEL_GRID` overrides, clamped at 240. A confirmation run
on a fresh RTX 4090 selected 96 unprompted and measured 260.4 off / 280.8
strict / **465.0 megakernel (+78.6%)**, strict identical on 8/8.

### 8. 2026-08-11 — the tail phase

The next layer's add/RMSNorm folded into the megakernel: the decode loop
issues **one fusion per step instead of one per layer** (measured 1 vs 25
on the shipped checkpoint). Bit-identical to the exact fuse (a CUDA test
asserts it); end-to-end +2.2% (RTX 3090) and +3.5% (RTX 4090) as paired
geomeans over ten interleaved fresh-process pairs.

### 9. 2026-08-11 — `qmm_naive` reproduced bit for bit

The fact that made an exact fast lane constructible: the stock kernel for a
decode step's eight gathered experts re-derived exactly — dequant
`bf16(bf16(q·s)+z)`, the same `m16n8k16` bf16 tensor-core atom with one row
of A populated, k-tiles of 128 in order, one epilogue rounding. Every
column of both expert projections matches on real weights. Two corollaries:
the CUDA dispatch sends `M*B < 8` to `gather_qmv`, a *different* kernel, so
single-expert comparisons measure the wrong reference; and each output
column's bits depend only on the k-order, so any grid layout over columns
preserves them. The rest of the block was pinned the same day (router
chain, the shape-picks-the-kernel renorm, the non-contracted aggregation).

### 10. 2026-08-11 — the exact MoE megakernel assembled, tuned, made default

Stock stream on 8/8 screened prompts at megakernel speed (the lane table in
[The exact MoE megakernel](#the-exact-moe-megakernel)); three rounds of
bit-neutral load and scheduling work (tile-wide `uint4` weight reads,
single-projection warp tasks with the activation folded into the next
phase's shared load, paired router reads) took it from 110 tok/s at
assembly to parity with the ~1 ULP lane; validated on all five
architectures. Two humbling details: the renorm's `sum(axis=-1)` over a
`(1,1,8)` array dispatches a different reduce kernel than a flat `(8,)` —
the shape picks the bits — and the one-element divergence that exposed it
survived 72 random layer tests before live data caught it.

### 11. 2026-08-11..12 — the attention bit map and megakernel

The decode attention block's stock kernels pinned
(`benchmarks/maple_attention_semantics.py`: 1-pass SDPA port 12/12 at five
context lengths, both bf16 gemv shapes 12/12 — then discarded for the real
2-bit `qmv` recipe, exact split+RoPE) and assembled into one dispatch; a
decode layer is now **two dispatches**. +20.6% on `sm86` (graphs on),
**+42.9% on `sm89`** — 318.8 → 455.7 tok/s with the spread collapsing to
±1.7. Made the default with live per-layer probes and a multi-turn flush
guard. Bugs the screen caught on the way: the projections are 2-bit
quantized (dense-gemv phases measured the wrong reference), partial RoPE
rotates only 64 of 128 dims, per-step re-sync erased the lane's own cache
appends, write-back must rebuild the stock buffers outright, and per-step
`mx.array` scalars thrashed CUDA-graph capture — the counters now live
on-device and the kernel advances them itself.

### 12. 2026-08-12 — past the 1-pass limit

The stock `kernel_sdpav_2pass` pair ported inside the same dispatch
(`benchmarks/maple_attention_2pass_semantics.py`: 72/72 bitwise vs stock at
kL 1025/1360/2048/3333/4096/8192); phase C branches on the live kL, one
extra grid barrier, full-attention buffers growing 1024→8192 with one
recompile per tier.

### 13. 2026-08-12 — the ring re-entry fixes

The boundary suite (`benchmarks/maple_attention_boundary_check.py`) caught
two real bugs: re-entry after a multi-token prefill on a rotated ring
mis-read the stock cache's temporal concat state (out-of-range slot, wrong
window — **this affected the shipped default with prompts longer than the
sliding window**), and python-side slice assignment into the persistent
buffers copies-on-write, orphaning the kernel's const_cast appends. All
seeding is now kernel-side through the input pointers. Boundary cases A-E
(cross-1024 in flight, start past 1024, write-back + regrow, the 4096
tier, 2048→4096 growth over 260 tokens) and every legacy regression
(multi-turn 2/2, rotation 700×2, stream 4×256, chained cache) are
bit-identical.

### 14. 2026-08-12 — the multi-arch verdicts and the per-arch default

Clean-pod campaigns: the boundary suite is bit-identical on `sm89` and
`sm90` (all five cases incl. growth tiers), and the attention lane's gain
holds past kL=1024 on `sm89` (+2.2…+6.5% in-run at every length up to
6000). But H100 measured a consistent **−12…−14%** in interleaved pairs —
132 SMs starve on the 64-block grid — so the lane's default became
data-driven: auto-on for `sm86`/`sm89`, auto-off elsewhere until profiled,
explicit env always winning. The same day the multi-token chain idea was
honestly killed: bits identical at every L, but against the stock async
double-buffer it loses (351.5 vs 296-348) — after the two-dispatch layer
the wall is GPU time, and host-side batching is exhausted.

## Known limitations

- Claims apply to the exact representative SKU, driver, MLX/CUDA version,
  model revision, and source provenance in the artifacts — not every GPU
  sharing a compute capability. Each throughput row is a single device
  instance measured once, not a fleet claim.
- The attention megakernel's throughput is profiled on `sm86`/`sm89`; on
  `sm90`/`sm100`/`sm120` it is bit-validated by its live probes but not yet
  perf-profiled. Contexts past 8192 keys fall back to the stock attention
  path per layer.
- Fresh Mac/Metal validation remains outstanding.
- Five `tests/test_generate.py` failures remain baseline-compatible on the
  tested checkout and are not claimed as fixed.
- Approximate router/add-RMS/ternary paths remain research-only.
- **MLX CUDA 0.32.0 needs a driver newer than 550.163.** An L40S offered
  with driver 550.163.01 could not start MLX at all
  (`cudaMallocManaged failed: unknown error`). Working hosts here ran
  575-610.
- **A CUDA 13 toolkit on the host breaks every custom kernel.**
  `mlx-cuda-12` compiles kernels with its bundled nvrtc 12.9 but takes
  headers from `$CUDA_HOME/include`, defaulting to `/usr/local/cuda`. On a
  host whose toolkit is 13.x that mixes nvrtc 12 with CUDA 13 headers and
  every `mx.fast.cuda_kernel` fails to compile inside `cuda_fp8.hpp`, which
  surfaces as *every* fast path silently falling back. Install the matching
  headers and point `CUDA_HOME` at them:

  ```bash
  pip install 'nvidia-cuda-runtime-cu12==12.9.*'
  mkdir -p ~/cuda12 && ln -s "$(python -c 'import nvidia.cuda_runtime, pathlib; print(pathlib.Path(nvidia.cuda_runtime.__file__).parent / "include")')" ~/cuda12/include
  export CUDA_HOME=~/cuda12
  ```
- The megakernels' grid barriers assume every block is resident, and MLX
  exposes neither the multiprocessor count nor occupancy, so grids are
  inferred from compute capability and memory. The rule is deliberately
  conservative and `MAPLE_MOE_MEGAKERNEL_GRID` overrides it, clamped at 240
  so a typo becomes a slowdown rather than a deadlock.
- **The megakernels want the GPU to themselves.** The barriers spin, so a
  GPU shared with another CUDA process pays for the spin. On a 3090 running
  two other workloads the fast lane swung between 183 and 340 tok/s across
  four fresh processes while `off` swung between 123 and 211 — usable, but
  not something to benchmark on.
- Community-cloud hosts share the CPU even when the GPU is dedicated, and
  this workload is host-bound. One host with an idle GPU and a load average
  of 4-9 returned between 104.7 and 325.8 tok/s for the same configuration.
  Check `/proc/loadavg` before trusting a measurement and report medians
  over many fresh processes.
- The accepted RTX 5090 W2 tile (`16x32x128`, +1.615%, 95% CI
  +1.322%-+1.909%, 12/12 wins) requires the separately built experimental
  MLX backend and is not enabled by this patch alone; RTX 4090 and B200
  candidates were array-exact but failed their performance gates, H100
  retained its stock tile.


### 15. 2026-08-12 — the grid-scan postscript on sm90

An H100 grid scan (64/96/128/192 blocks, `MAPLE_ATTENTION_MEGAKERNEL_GRID`)
landed on a severely CPU-starved host (loadavg 16 on 8 vcpu; the stock
baseline itself collapsed 260 → 73 tok/s) and inverted the picture: on a
starved CPU the attention lane WINS +13-14% at every grid — it removes
host work, which is exactly what a poor host lacks — while the healthy-CPU
H100 the day before measured it ~12% slower. Grids 64/96/128 tie within
noise, 128 already grazes the residency edge (the kernel's register weight
holds about one 1024-thread block per SM even on Hopper), and 192
deadlock-spins on the grid barrier. So the sm90 regression is CPU-class,
not grid starvation: the auto default stays off, `MAPLE_ATTENTION_MEGAKERNEL=1`
is the documented lever for CPU-poor deployments, and the grid override is
clamped to 112 above sm86.

### 16. 2026-08-12 — the MMA row-independence proof

The make-or-break probe for free-MMA-row speculation
(`benchmarks/maple_mma_row_independence.py`, sm89): filling rows 1..15 of
the `m16n8k16` bf16 atom does not move row 0's bits, and every filled row
is bit-equal to the same activation run alone (16/16). A verify pass over
L ≤ 16 draft tokens through the expert MMA IS the L sequential M=1 passes
bit for bit — speculation can keep the project's exactness invariant. The
design and break-even arithmetic live in `SPECULATION-DESIGN.md`; the
first step of the next cycle is measuring prompt-lookup acceptance on
real service streams, no kernels required.
