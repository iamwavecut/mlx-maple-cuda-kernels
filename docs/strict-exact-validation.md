# Maple strict-exact validation

## Contract

`strict-auto` is the deployment lane that must reproduce portable MLX arrays and
greedy output. A custom path may be enabled only after an array-equality live
probe; known non-exact paths are hard-disabled rather than admitted by a
one-input tolerance probe. The semantic/token-gated lane is measured and
reported separately.

For CUDA validation, set these variables **before the first CUDA use**:

```sh
MLX_CUDA_USE_CUDNN_SDPA=0
MLX_USE_CUDA_GRAPHS=1
MLX_CUDA_GRAPH_CACHE_SIZE=400
MLX_MAX_OPS_PER_BUFFER=100
MLX_MAX_MB_PER_BUFFER=100
```

The first setting is part of the deterministic oracle, not a claimed Maple
kernel optimization. The 400/100/100 graph profile is the recommended measured
configuration. The initial A-D factorial used cache 2000 to attribute ops/MB;
a separate cache comparison at 100 ops / 1000 MB found no supported benefit
over 400. Repeated
portable generations diverged after long decode
with cuDNN SDPA enabled, both with CUDA graphs on and off. With cuDNN SDPA
disabled, repeated 512-token portable generations had identical token, text,
selected-logprob, and top-1 hashes.

## Current lane assignment

Strict-auto candidates:

- portable router (`_use_approximate_router = False`);
- portable residual add/RMS chain (`_use_approximate_add_rms = False`);
- stock GatherQMM for every W2 expert projection;
- exact fused Q/K norm + RoPE, admitted by `mx.array_equal`;
- cached flat decode `lhs_indices` is array-exact but remains off by default:
  its isolated marginal effect was not statistically supported;
- exact LM head (`use_flash_head=False`);
- ternary expert GEMV disabled.

Semantic/experimental candidates:

- fused router GEMV/softmax/top-8;
- residual-carrier add+RMS chain;
- ternary up/gate GEMV;
- FlashHead and KV quantization.

## Why two formerly “strict” paths were demoted

A tolerance probe is insufficient for autoregressive exactness.

- The router preserves top-8 ids/order on sm86, but its normalized fp32 scores
  are not array-exact. Small routing-weight differences can compound.
- With deterministic SDPA, the residual-carrier add+RMS path first changed the
  fixed cognitive regression at generated token 217. Q/K-only and cached-LHS-
  only runs stayed identical for all 512 generated tokens in the same ablation.

Both paths remain available for explicit semantic research but default to the
portable implementation.

## Correctness harness

`benchmarks/maple_common_slice_benchmark.py` uses a pinned 20-case regression
slice from GPQA Diamond, SuperGPQA, and AIME 2025. It:

- forces the worktree model (`model_file=None`) and exact head;
- hashes the actual loaded model source and manifest;
- alternates reference/strict-auto order;
- records prompt hashes, token/text hashes, selected-token logprob hashes,
  top-1 hashes, first mismatch, finish reason, and strict final-answer grading;
- records every accepted/fallback path;
- fails if any required decode artifact differs.

This is a small deterministic regression slice, not a statistically complete
quality evaluation. Timing claims come from separate warm paired runs so the
per-token validation instrumentation is not included in throughput.

## Next exact router boundary

The largest plausible strict router fusion keeps every arithmetic operation in
portable MLX:

1. stock fp32 matmul;
2. stock full softmax;
3. a specialized CUDA stable top-8 compare/copy over the 256 rounded scores;
4. the same stock eight-value sum/add/div helper as the reference.

The CUDA selector must reproduce MLX 0.32's stable sorted argpartition suffix,
including tie order, and only copy score bits. It must not recompute GEMV,
`expf`, the softmax denominator, or selected-score renormalization. Gate this
prototype to the exact MLX/version/shape and retain an array-equality canary.

## Remaining strict work

- Determine why MLX's decode-time cuDNN eligibility changes across repeated
  rotating-cache generations (evaluated/sliced cache status is part of the
  current heuristic). A future strict-fast lane can re-enable cuDNN only after
  repeated reference hashes are stable across fresh processes and graph states.
- Prototype the portable-arithmetic exact top-8 selector above.
- Continue the dominant exact W2 work as a generic affine top-8 multi-row
  GatherQMM replacement, retaining bf16 projection boundaries and slot order.
- Validate each accepted path independently on sm89, sm90, sm100, and sm120.
