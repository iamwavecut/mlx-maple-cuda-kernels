# Changelog

## 0.5.0 — 2026-08-11

- Folded the next layer's residual add + RMSNorm into the megakernel as a
  tail phase. The Python decode loop issued one standalone fuse dispatch per
  MoE layer per step purely to prepare the next attention input; the tail
  moves that to the GPU behind a fourth grid barrier, and the loop now issues
  **one** fuse per step (the first layer) instead of twenty-five. The tail
  mirrors `_EXACT_ADD_RMS_SOURCE` line for line, and a CUDA test asserts its
  output is bit-identical to the standalone fuse — the lane's ~1 ULP story
  stays confined to the MoE math. The 846-token quality suite reproduces the
  previous build's corpus NLL to the last digit in all three lanes.
- Measured the tail end-to-end on rented hosts, paired and interleaved:
  +2.2% paired geomean on an RTX 3090 (8/10 wins) and +3.5% on an RTX 4090
  (8/10 wins), token stream bit-identical in every process. Conservative --
  both hosts carried external CPU load.
- Proved the stock MoE matmul is reproducible bit for bit in a custom kernel
  (`benchmarks/maple_qmm_naive_repro.py`): dequant as bf16(bf16(q*s)+z), the
  same m16n8k16 bf16 tensor-core atom, k-tiles in order, one epilogue
  rounding — every column of both projections matches `qmm_naive` exactly on
  real weights. The decode dispatch picks qmm_naive whenever a step routes 8
  experts (M*B >= 8); gather_qmv is the B=1 kernel and was the wrong earlier
  reference. This makes an array-exact megakernel expert phase — a fast lane
  that keeps the strict lane's reproducible stream — constructively possible.
- Added `benchmarks/maple_fast_lane_profile.py`: exclusive host time per
  sub-block with the megakernel on. On the shared-GPU dev host the step's
  remaining host budget is attention 681 us, KV-cache updates 334 us,
  megakernel dispatches 410 us, fuse 20 us — the fuse column used to be ~330,
  which is the tail's savings measured directly.

## 0.4.0 — 2026-08-11

- Established that Maple decode on CUDA is host-bound: with a warm cache the
  GPU wait per step is ~0.003 ms on every host measured, so wall clock is the
  sum of per-operation host costs and the lever is operation count.
- Promoted the fused residual add + RMSNorm to a strict default. The kernel now
  reproduces `mx.fast.rms_norm`'s thread mapping (512 threads, four contiguous
  elements each) instead of the elementwise thread count with two chunks; that
  partition was the whole reason the path was array-inexact and demoted.
- Added a strict-default fused QKV split: the Q/K norm + RoPE kernel widened to
  consume the fused qkv projection and emit queries, keys and values in their
  final shapes, removing the slice-and-reshape chain. Bit-identical by
  construction.
- **Made the MoE megakernel the default lane.** It runs the router, experts,
  activation, score-weighted aggregation and the preceding add/RMSNorm in one
  dispatch, using atomic-counter grid barriers, and is worth 73-88%. It is
  within ~1 ULP of bf16 rather than array-exact, so **the default token stream
  is no longer reproducible against stock** — about 9% of top-1 predictions on
  near-ties differ. The quality suite below finds no cost to that on any
  supported architecture. `MAPLE_MOE_MEGAKERNEL=0` restores the array-exact
  lane, which is what a reproducibility claim or a bisect needs.
- Added an opt-in compiled router: the stock chain under `mx.compile`,
  array-exact, but its end-to-end effect measured 1.0062x with a 95% interval
  of 0.9927-1.0198, so it ships off.
- Validated on all five targets. Array-exact lane: RTX 3090 +6.68%, RTX 4090
  +16.91%, H100 80GB +9.32%, B200 +11.77%, RTX 5090 +10.63%, and an identical
  token stream on 8/8 screened prompts on every one. Megakernel: +79.51%,
  +87.04%, +76.11%, +73.86%, +75.35%.
- The GPU wait per step measures 0.002-0.004 ms from a 3090 to a B200, so the
  host-bound finding is not specific to a small GPU.
- Made the megakernel grid depend on the device instead of a fixed 32 blocks,
  which was tuned on a 3090 and left 18.3% on an RTX 4090 and 14.3% on a B200.
  Selected from compute capability and memory, overridable and clamped.
- Documented that a CUDA 13 toolkit on the host makes every custom kernel fail
  to compile, because `mlx-cuda-12` pairs its bundled nvrtc 12.9 with headers
  from `$CUDA_HOME`. It presents as every fast path silently falling back.
- Made every lane selectable from the environment (`MAPLE_MOE_MEGAKERNEL`,
  `MAPLE_COMPILED_ROUTER`, `MAPLE_FUSED_ADD_RMS`, `MAPLE_FUSED_QKV`). The fast
  lane previously required importing the module and setting an attribute before
  the model loaded, which is the wrong shape for a server.
- Added `benchmarks/maple_quality_suite.py`: twelve documents scored one token
  at a time against a cache. The strict lane matches the reference mean NLL to
  the last digit with zero top-1 changes on all five architectures; the
  megakernel moves corpus perplexity by -0.8% to -1.3%, which rules out a
  quality regression without being an improvement.
- Adopted a screened equivalence protocol after finding that the stock path is
  not always reproducible run to run. Verdicts are now taken only inside the
  region where three reference runs agree.
- Strengthened the add/RMSNorm probe to inject outliers: on Gaussian vectors a
  wrong reduction passes 300/300 trials, and only a realistic dynamic range
  separates the candidates.

## 0.3.1 — 2026-08-07

- Re-ran RTX 3090 (`sm86`) with the current release source and the same
  12-fresh-process strict baseline used by the multiarchitecture matrix.
- Published the allowlisted `sm86` bundle, raw-manifest commitment, campaign
  provenance, generated-kernel identity, and updated primary table.
- Measured exact Q/K at +6.06% (95% CI +2.73%–+9.50%) and exact Q/K plus
  cached LHS at +9.37% (+5.73%–+13.12%); both passed the complete finite
  correctness gate with all 24 Q/K layers active.
- Retained the earlier shared-host `sm86` records as historical evidence rather
  than pooling them with the fresh current-source result.

## 0.3.0 — 2026-08-06

- Completed fresh fail-closed strict campaigns on RTX 4090 (`sm89`), H100
  80GB HBM3 (`sm90`), B200 (`sm100`), and RTX 5090 (`sm120`).
- Added the independently gated `sm100`/`sm120` stock-RoPE rounding pin and a
  frozen BF16-midpoint regression fixture.
- Published 12-fresh-process Q/K and cached-LHS results, per-SKU sanitized
  bundles, graph screens, W2 decisions, source mapping, and recursive checksums.
- Accepted the experimental `16x32x128` W2 tile only on RTX 5090; demoted the
  RTX 4090/B200 follow-ups and retained the H100 stock default.
- Updated the NVIDIA QuickStart to disable TF32 and remote code and to scope all
  claims to exact SKU/driver/MLX/CUDA/checkpoint/source provenance.

## 0.2.0 — 2026-08-06

- Replaced tolerant strict probes with shape/dtype/`mx.array_equal` gates.
- Added deterministic `sm86` oracle requirements and actual-loaded-module
  provenance checks in the common-slice harness.
- Demoted non-array-exact router, residual add/RMS, and ternary projection to
  explicit opt-in experimental paths; all are disabled by default.
- Added exact Q/K CUDA profiles, cached decode-LHS support, router graph
  dependency repair, and fail-closed fallbacks.
- Added fixed 20-case 512/1024 regression gates, separated correctness/timing,
  component and graph factorials, answer/provenance metadata, and source
  snapshots.
- Published the current warm `sm86` measurements: 195.27 tok/s conservative
  Q/K default and 209.58 tok/s Q/K + cached-LHS profile.
- Added a pinned canonical NVIDIA inference QuickStart and exact-head entry
  point that asserts the patched model source and reports active/fallback state.
- Archived the initial 256-token multi-architecture results as superseded
  historical evidence pending fresh `sm89`-`sm120` validation.

## 0.1.0 — 2026-08-06

- Initial CUDA port and short-oracle multi-architecture benchmark snapshot.
