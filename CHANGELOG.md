# Changelog

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
- Archived the initial 256-token multi-architecture results as superseded
  historical evidence pending fresh `sm89`-`sm120` validation.

## 0.1.0 — 2026-08-06

- Initial CUDA port and short-oracle multi-architecture benchmark snapshot.
