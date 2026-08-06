# Published results

## Current strict evidence

`summary.csv` contains historical `sm86` rows and two fresh rows for each of
`sm89`, `sm90`, `sm100`, and `sm120`:

- `strict_qk_default_fresh_process`: conservative exact-probed Q/K source default;
- `strict_qk_cached_lhs_opt_in_fresh_process`: exact Q/K plus the lifecycle-limited cached
  decode LHS option.

For the fresh matrix, throughput columns are arithmetic means across 12 fresh
model processes on one device instance. Gains and confidence intervals are
geometric statistics from paired within-process ratios. Do not compare
absolute rates across hosts as a GPU ranking or pool the historical `sm86`
design with the fresh matrix.

[`PUBLIC-INDEX.json`](PUBLIC-INDEX.json) binds every canonical baseline, graph, W2, Blackwell, campaign, source, analysis, sanitized-bundle, and private raw-manifest commitment used by the release.

## Allowlisted per-SKU bundles

Each directory under `cuda/multiarch/` contains a 15-file sanitized baseline
bundle plus its leaf `SHA256SUMS`:

- `sm89`: RTX 4090, driver 580.159.04;
- `sm90`: H100 80GB HBM3, driver 580.126.09;
- `sm100`: B200, driver 580.126.20;
- `sm120`: RTX 5090, driver 580.126.20.

`summary.json` is the canonical baseline analysis. Other files retain
allowlisted strict-path, fresh-process, component, common-slice, package,
model, harness, and source provenance. Every bundle excludes UUID, PCI ID,
IP/port, local/model paths, raw service logs, generated text, and profiler data.

Compact cross-architecture evidence:

- `cuda/multiarch-strict-summary.jsonl`: correctness and primary Q/Q+LHS
  fresh-process results;
- `cuda/multiarch-graph-summary.jsonl`: five-block graph screen;
- `cuda/multiarch-w2-summary.jsonl`: W2 screen/follow-up and custom-wheel
  provenance; only the `sm120` tile was accepted;
- `cuda/blackwell-qk-rounding.jsonl`: B200/RTX 5090 isolation and fixed/original
  gates;
- `cuda/release-source-equivalence.json`: generated Q/K kernel hash bridge from
  executed source files to the release source.

The root `SHA256SUMS` covers every published file under `results/` except
itself, including leaf manifests. The compact files are derived from the bound
canonical analyses; raw campaign artifacts remain private because their schemas
contain sensitive provider/device/path fields.

## Exactness envelope

Every fresh target had 24/24 Q/K layers active, an exact random 1024-token gate,
three exact multi-seed cases, 20/20 exact cases at both 512 and 1024 tokens, and
144 deterministic stock W2 projection fingerprints. W2 candidates separately
had to match all 144 reference arrays before timing. Common-slice equality
covers token IDs, decoded text, selected-token-logprob hash, and top-1 hash. It is finite
regression evidence, not exhaustive full-logit equality or a quality score.

Executed Maple source hashes differ by campaign stage: `7785da2a…` on
`sm89/sm90`, `b34cd977…` on `sm100`, and release `28ceabac…` on `sm120`.
Generated release Q/K source matches the executed profile-specific source for
the earlier three targets; this narrow bridge is disclosed rather than calling
the whole files identical.

## Historical evidence

The top-level `cuda/sm86-*.jsonl` files are the revised historical RTX 3090
campaign. `legacy-initial-port/` preserves the original short-oracle data only
for transparency; tolerant probes admitted paths now classified as semantic,
so those records are not current strict claims. Mac/Metal remains unvalidated
with the current source.
