# Performance notes

## Current `sm86` result

The conservative strict default enables exact-probed Q/K norm + RoPE and keeps
all known approximate kernels disabled. In the eight-block component
factorial, Q/K alone averaged 195.27 tok/s versus 179.39 tok/s portable; its
direct paired geometric ratio was +8.51% (95% CI +0.90% to +16.70%). The
factorial Q/K main effect averaged across LHS states was +10.00% (`p=0.044`).

An explicit exact speed profile additionally enabled cached flat decode LHS.
Across six alternating pairs at graph settings 100 ops / 100 MB / cache 400:

- portable arithmetic mean: 177.56 tok/s;
- strict-profile arithmetic mean: 209.58 tok/s;
- paired geometric ratio: +18.28%;
- 95% CI: +5.18% to +33.00%;
- wins: 6/6, log-ratio t-test `p=0.0143`.

Cached LHS's isolated main effect was +2.10%, with a CI from -2.73% to +7.17%
and `p=0.345`; it is therefore opt-in rather than a default optimization.

## Graph tuning

At graph cache 2000, increasing the ops limit from 20 to 100 had a +11.85%
factorial main effect (95% CI +7.19% to +16.71%, 4/4 wins). The MB main effect
crossed zero. Focused 1000 MB / 100 MB comparison over eight blocks, combining
the initial and follow-up blocks, was +1.79% with a wide CI crossing zero.

A separate four-pair cache comparison at 100 ops / 1000 MB estimated 2000 /
400 at -4.71%, also with a wide CI crossing zero. Use cache 400, 100 ops, and
100 MB. The larger MB
limit increased peak MLX memory without an established marginal speed benefit.

## Superseded initial result

The initial public table reported 136.86 -> 189.30 tok/s (+38.3%) on RTX 3090.
Its absolute accelerated speed was real for that process, but it is no longer a
strict-exact claim: the campaign used a 256-token oracle and tolerant live
probes, and enabled router/add-RMS paths that later changed long generation.
The raw data remains under `results/legacy-initial-port/` for historical
transparency.

The revised exact speed profile reaches 209.58 tok/s using deterministic SDPA
and excluding those approximate paths. The historical artifact observed
189.30 tok/s, but generation length, baseline, graph configuration, oracle, and
active paths differ; the two absolute observations are not a controlled
incremental-speed comparison.

## Dominant remaining bottleneck

An Nsight Systems audit of the exact-head legacy RTX 3090 workload attributed about 34.6% of GPU
kernel time to MLX CUDA's generic affine 2-bit expert `qmm_naive`. Other notable
shares were non-gather 2-bit QMV (10.4%), router work (9.6%), exact 4-bit
LM-head QMV (7.9%), Q/K (6.4%), SDPA (4.8%), and add/RMS (3.9%). These shares
are workload-specific.

Native experiments included B=8 QMV, several tile shapes, rows-per-block,
elements-per-thread, and a top-8 -> two-top-4 split. None produced a repeatable
strict end-to-end win; the split preserved the short token hash but was 30.4%
slower. They are not shipped as defaults.

The highest-value next kernel is a generic affine W2 top-8 multi-row
projection. It must preserve stock FP32 accumulation order, BF16 rounding after
each projection, expert-slot order, and ordered FP32 aggregation. A mathematically
equivalent but reordered reduction is not acceptable in the strict lane.

## Experimental semantic work

- Router microbenchmarks can be materially faster, but normalized FP32 scores
  are not array-exact. An exact future hybrid should retain stock
  matmul/softmax/renormalization and fuse only stable top-8 selection/copy.
- The checkpoint's structured ternary up/gate weights enabled roughly 1.47x
  projection speed in a prototype, but full-layer BF16 values differed.
- Residual add/RMS passed local probes yet changed the deterministic long
  token stream at generated token 217.

All three remain opt-in experiments and are excluded from strict throughput.
