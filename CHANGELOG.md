# Changelog

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
