# Performance notes

## Fresh strict matrix

The current multi-architecture table uses 12 fresh model processes per mode on
one physical instance of each SKU. Within every process, warm `B=1`, `L=1`,
128/512 BF16 decode is measured after correctness and warmup. Arithmetic means
are displayed; effects are geometric means of paired ratios.

| GPU | Portable | Q/K default | Gain (95% CI) | Q/K + cached-LHS | Gain (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: |
| RTX 4090 / `sm89` | 182.36 | 209.84 | +15.31% (+11.29%–+19.49%) | 214.66 | +18.01% (+13.68%–+22.50%) |
| H100 80GB HBM3 / `sm90` | 202.62 | 233.67 | +15.24% (+12.86%–+17.66%) | 246.34 | +21.54% (+19.10%–+24.03%) |
| B200 / `sm100` | 241.51 | 280.70 | +16.28% (+14.23%–+18.37%) | 297.47 | +23.37% (+19.55%–+27.31%) |
| RTX 5090 / `sm120` | 398.49 | 429.72 | +7.84% (+6.88%–+8.81%) | 438.01 | +9.92% (+8.89%–+10.96%) |

All eight comparisons won 12/12 pairs. Cached LHS remains off by default due to
its cache lifecycle. Absolute rates are not cross-GPU rankings: hosts, clocks,
and GPUs differ, and each claim is a single-instance observation.

The historical RTX 3090 / `sm86` experiment used a different paired design and
shared host. Its conservative Q/K estimate was +8.51%; Q/K plus cached LHS was
+18.28%. It remains auditable but should not be pooled with the fresh matrix.

## Graph screen

Five-block screens used the same device instance as each strict baseline. The
common campaign profile was cache 400 / 100 ops / 100 MB. It is reproducible,
not asserted globally optimal.

| Profile | 100 vs 20 ops factorial effect | Direct recommended B/A |
| --- | ---: | ---: |
| `sm89` | +19.73%, `p=0.00230` | +25.07%, `p=0.0148` |
| `sm90` | +8.45%, CI crosses 1 | +6.58%, CI crosses 1 |
| `sm100` | +10.90%, `p=6.82e-5` | +6.60%, CI crosses 1 |
| `sm120` | +14.22%, `p=1.65e-5` | +16.31%, `p=1.20e-5` |

Cache and MB effects were mostly inconclusive. On `sm120`, 1000 MB was slightly
slower than 100 MB in this screen; cache 2000 had no supported benefit over 400.

## Experimental W2 tile follow-up

The W2 study used an architecture-specific MLX 0.32.0 wheel built from a sealed
runtime tile-override patch. It is not part of the normal QuickStart.

- `sm120`: `16x32x128` strict-accepted, +1.615% (95% CI +1.322%–+1.909%),
  12/12 fresh-process wins, `p=9.63e-8`.
- `sm89`: same tile array-exact but +0.066% (CI -1.259%–+1.409%); demoted.
- `sm100`: same tile array-exact but +0.715% (CI -0.003%–+1.438%); demoted.
- `sm90`: stock default ranked first end-to-end in screening; no candidate.

Do not add the `sm120` tile percentage to the strict Q/K result: it is a
separate candidate-vs-default comparison under a custom backend.

## Blackwell fix performance gate

The rounding fix was compared with the old arithmetic in 16 balanced fresh
processes using distinct source/kernel identities. On B200, fixed/original was
+0.91% with a 95% CI of -4.35% to +6.46% (9/16 wins). On RTX 5090 it was +8.58%
with a CI of -0.58% to +18.60% (11/16 wins). The predeclared statistical test
did not detect a slowdown; the intervals do not prove zero cost or universal
non-inferiority.

## Scope and excluded work

Throughput excludes prefill, batched decode, scaled-RoPE policies, JIT/live
probe, cold cache setup, selected-logprob/top-1 instrumentation, FlashHead, KV
quantization, and all approximate router/add-RMS/ternary paths. Intervals are
small-n exploratory evidence after tuning, with no multiple-testing correction.
The exact per-trial data and hashes are under [`../results/cuda/`](../results/cuda/).

## Superseded initial result

The original 136.86 → 189.30 tok/s (+38.3%) RTX 3090 table is historical only.
Its 256-token oracle and tolerant probes admitted router/add-RMS paths that
later diverged. Raw sanitized data remains under `results/legacy-initial-port/`
for transparency, but it is not a current strict claim.
