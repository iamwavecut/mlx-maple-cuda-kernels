# Benchmark methodology

## Frozen inputs

Fresh `sm86`–`sm120` campaign inputs:

- checkpoint: `deepgrove/maple-preview-2bit-mlx` revision
  `361db5da5e74ff6fcdd852d478e1f266ce11013a`, with exact recursive model
  manifest verified before every run;
- DeepGrove base: `eba96c16158f032821b0bf374ea1421cfddef0a9`;
- Python 3.12.3, `mlx==0.32.0`, `mlx-cuda-12==0.32.0`;
- CUDA runtime 12.9.79, NVRTC 12.9.86; driver recorded per SKU;
- exact LM head, `model_file=None`, `use_flash_head=False`, and
  `trust_remote_code=False`;
- actual loaded module/class source and SHA-256 recorded in every relevant lane.

Executed full-file Maple hashes were:

| Profile | Executed source SHA-256 |
| --- | --- |
| `sm86`, `sm120` | `28ceabac2b7570ff3712473c88eb7698b5a1904cd1b9cd55c698794fd457ccb8` |
| `sm89`, `sm90` | `7785da2a85b97b9fd7759d8756b1daf2231ec8b912d42b4b7bc9c04637b371ae` |
| `sm100` | `b34cd9777cf5a8775ed4e814fe7e14c987a9021627224168595c92ddf21edae4` |

The release source is `28ceabac…`. A local capture test hashes the exact CUDA
source passed to `mx.fast.cuda_kernel`; release RoPE/NoPE generated source
matches the validated profile source for `sm89`, `sm90`, and `sm100`. This is
an architecture code-generation bridge, not a claim that the whole release
file was freshly rerun on those three devices. The fresh `sm86` and `sm120`
runs directly used the release file. Earlier `sm86` evidence remains historical
and is not pooled with the fresh matrix.

## Deterministic process environment

Set before the first CUDA operation:

```sh
MLX_CUDA_USE_CUDNN_SDPA=0
MLX_ENABLE_TF32=0
MLX_USE_CUDA_GRAPHS=1
MLX_CUDA_GRAPH_CACHE_SIZE=400
MLX_MAX_OPS_PER_BUFFER=100
MLX_MAX_MB_PER_BUFFER=100
TOKENIZERS_PARALLELISM=false
```

Portable long generation could diverge while cuDNN SDPA was eligible. Its
disablement is an oracle-stability requirement, not credited acceleration.
TF32 is disabled to preserve the tested arithmetic contract.

## Correctness pass

Correctness and timing are separate. Every baseline required:

1. focused array tests, including shape/dtype/value equality with
   `mx.array_equal` and the frozen Blackwell boundary where applicable;
2. three multi-seed cases (generation caps 1024, 2048, and 1024);
3. 144 deterministic stock W2 projection fingerprints
   (`24 layers × 2 projections × 3`); every tile candidate had to match the
   corresponding 144 reference arrays before timing;
4. direct random 1024-token reference/strict equality;
5. 20 fixed cases at both 512 and 1024 tokens, comparing token IDs, decoded
   text, selected-token-logprob hash, and top-1 hash;
6. recorded strict path state, with all 24 Q/K layers active for acceleration.

Selected-logprob/top-1 equality is stronger than token equality alone but is
not exhaustive full-logit equality. The fixed slice is pinned to antirez/ds4
commit `b0309611041655f4e45671cfd9c9886aff161406`; it is a regression harness,
not a representative quality benchmark.

## Fresh-process timing

The primary statistical unit is one fresh model process on one physical device
instance. Twelve position-balanced blocks compare:

- `R`: portable reference;
- `Q`: exact fused Q/K only;
- `QL`: Q/K plus cached decode LHS.

Each mode warms before timing `B=1`, `L=1`, 128 deterministic prompt tokens and
512 generated tokens with EOS disabled. JIT/live-probe, cold cache, prefill,
batched decode, and validation instrumentation are excluded. Displayed tok/s
are arithmetic means; primary effects are geometric means of within-block
ratios with two-sided 95% t-intervals on log ratios.

The direct profile separately uses six alternating pairs. Component factorials
cross `R/Q/L/QL` within a loaded model for attribution; they are supporting,
not the primary fresh-process claim.

## Graph and W2 screens

Graph screening uses five blocks over cache/ops/MB configurations. It is
screening-only and device-instance-specific. Runtime W2 tile alternation sets
CUDA graphs off and uses distinct tile-bearing JIT names to prevent silent
cubin reuse. A tile can advance only after projection exactness; final
acceptance requires 144 comparisons, direct/multi-seed/common-slice equality,
12 fresh paired processes, and the performance gate.

The W2 wheel is a separately built experimental MLX backend. Build provenance
pins the MLX commit, image digest, cuDNN 9.25.0.15-1, Python headers, build
patch, wheel, and installed library hashes. It is not the stock QuickStart
backend.

## Blackwell fixed/original comparison

B200 and RTX 5090 use a frozen FP32-weight boundary artifact derived from seed
6, offset 613, head 8, dimension 45. Original and fixed variants run in 16
balanced fresh-process blocks with distinct source/kernel identities. The
acceptance rule requires exact fixed output and no statistically significant
slowdown; confidence intervals are disclosed and are not interpreted as proof
of zero cost.

The full diagnostic artifact SHA-256 is `ecb4ff1…`. The checked-in regression
fixture is an input-only compressed subset/repack with SHA-256 `837638a…`; the
two hashes intentionally identify different files.

## Published and excluded data

Only allowlisted records are published. Sanitization removes GPU UUIDs, PCI
IDs, IPs/ports, filesystem/model paths, generated answer text, raw service logs,
and profiler databases. Each per-SKU bundle has its own `SHA256SUMS`; the root
results manifest covers every published result file except itself.

Cloud failures, raw logs, provider identifiers, approximate paths, and dataset
content are excluded. Intervals are finite single-instance evidence after
substantial tuning, with no multiple-testing correction; they are not
population-level hardware guarantees or cross-host performance rankings.
