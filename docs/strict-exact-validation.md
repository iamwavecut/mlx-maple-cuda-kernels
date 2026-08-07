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
MLX_ENABLE_TF32=0
MLX_USE_CUDA_GRAPHS=1
MLX_CUDA_GRAPH_CACHE_SIZE=400
MLX_MAX_OPS_PER_BUFFER=100
MLX_MAX_MB_PER_BUFFER=100
```

The first setting is part of the deterministic oracle, not a claimed Maple
kernel optimization. Repeated portable generations diverged after long decode
with cuDNN SDPA enabled, both with CUDA graphs on and off. With cuDNN SDPA
disabled, repeated 512-token portable generations had identical token, text,
selected-logprob, and top-1 hashes.

## Current lane assignment

Strict-auto candidates:

- portable router (`_use_approximate_router = False`);
- portable residual add/RMS chain (`_use_approximate_add_rms = False`);
- stock GatherQMM for every W2 expert projection;
- exact fused Q/K norm + RoPE, admitted by `mx.array_equal`;
- cached flat decode `lhs_indices` is array-exact in the tested matrix but
  remains off by default because its process-global cache identity omits model
  and device invalidation;
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

## sm100 and sm120 upper-half RoPE rounding boundary

MLX 0.32's stock CUDA RoPE and the fused Q/K kernel were algebraically equal
but did not always have the same FP32 rounding graph on the tested B200
(`sm100`) and RTX 5090 (`sm120`). For the second rotary half, stock RoPE
rounds the sine product and contracts the cosine product into the sum:

```cuda
__fmaf_rn(value, rope_cos[p], __fmul_rn(paired, rope_sin[p]))
```

NVRTC was otherwise free to contract the opposite product. A single FP32-bit
difference could then cross a BF16 midpoint; the frozen regression case is at
offset 613, head 8, dimension 45. The `sm100` and `sm120` sources now pin
stock MLX's association explicitly while retaining the same dispatch,
shared-memory layout, and one-multiply/one-FMA instruction count. The fixture
in `tests/data/sm100_qk_rope_boundary.npz` makes that boundary reproducible
without depending on an RNG implementation.

The two architecture claims were gated independently. On B200, the fixed path
was exact in 2,048 projection comparisons while the old-arithmetic control
mismatched 32 times. On RTX 5090, an expanded 4,608-comparison matrix had zero
fixed-path mismatches while the old control had 63 mismatching elements. Both
architectures also passed full-output, 20-case 512/1,024-token, fresh-process,
and no-statistically-significant-slowdown gates. Fixed/original 95% intervals
were -4.35% to +6.46% on B200 and -0.58% to +18.60% on RTX 5090: the test did
not detect slowdown, but does not prove zero cost. The generated `sm86`,
`sm89`, and `sm90` kernels remain unchanged.

This rounding claim is scoped to MLX 0.32.0 and CUDA 12.9. Re-run the frozen
boundary and complete architecture gate after either toolchain changes.

## Remaining strict work

- Determine why MLX's decode-time cuDNN eligibility changes across repeated
  rotating-cache generations (evaluated/sliced cache status is part of the
  current heuristic). A future strict-fast lane can re-enable cuDNN only after
  repeated reference hashes are stable across fresh processes and graph states.
- Prototype the portable-arithmetic exact top-8 selector above.
- Continue the dominant exact W2 work as a generic affine top-8 multi-row
  GatherQMM replacement, retaining bf16 projection boundaries and slot order.
- Repeat the full architecture gate for every new path and after MLX, CUDA,
  compiler, driver, or checkpoint changes; the current Q/K and cached-LHS
  gates are complete on `sm86`, `sm89`, `sm90`, `sm100`, and `sm120`.
