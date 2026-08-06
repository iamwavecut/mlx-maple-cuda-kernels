# Kernel coverage and design

## Strict boundary

The implementation has one current CUDA fast path admitted to the conservative
strict default: per-head Q/K RMSNorm fused with partial RoPE or NoPE. A live
probe compares shape, dtype, and every output value with portable MLX using
`mx.array_equal`. Compile failures, unsupported policies, future MLX changes,
or any mismatch latch the path to portable MLX.

Cached flat decode `lhs_indices` is also array-exact in the validated `sm86`
workload. It is retained as an explicit speed-profile option, not enabled by
default, because its isolated marginal throughput effect was inconclusive.

The exact LM head is required. FlashHead, KV quantization, and other reduction
order changes are outside this strict boundary.

## CUDA paths

### Q/K norm + RoPE/NoPE

Decode Q and K normalization and positional transformation are performed in a
CUDA kernel after live comparison against the original MLX operation. The
kernel preserves BF16 output boundaries and independently probes every layer.
This is the only custom arithmetic path enabled automatically in the current
strict source.

### Residual add + RMSNorm

CUDA and Metal implementations remain available for controlled semantic
studies. The local array probe was not a sufficient end-to-end oracle: under
deterministic long decode, enabling this path first changed the token stream at
generated token 217. `_use_approximate_add_rms` is therefore `False` by
default, and strict state configuration refuses it.

### Router

The experimental CUDA router supports fused GEMV, FP32 softmax, stable top-8,
and an MLX-GEMV hybrid profile. Its graph counter is an explicit output rather
than hidden mutable input state, and its expert ordering matches MLX
`argpartition` order.

Despite matching selected indices/order, normalized FP32 routing scores are not
array-exact. `_use_approximate_router` is `False` by default. The strict lane
uses the portable MLX matmul, softmax, argpartition, gather, and renormalization.
A future exact hybrid should retain stock GEMV/softmax/renormalization and fuse
only stable top-8 compare/copy over already-rounded scores.

### Ternary expert prototype

The checkpoint's 2-bit up/gate tensors use codes `{0,1,2}` with scale `alpha`
and bias `-alpha`; code 3 is absent. A direct ternary GEMV prototype accelerated
the projection by roughly 1.47x, but full-layer validation found BF16 output
differences. `_use_cuda_ternary_up_gate` remains `False` and the stock
`QuantizedSwitchLinear` path is the default.

### Decode LHS cache

`SwitchGLU` can reuse a flat cached sequence of decode row indices instead of
rebuilding it every token. The current process-global cache is keyed only by
`top_k`; it has no device/model invalidation. Evidence is limited to a
single-device, top-k=8, warm steady-state workload. The option defaults off both
because of that lifecycle limitation and because its independent speed benefit
was inconclusive.

## Architecture profiles

Profile selection and live fallback exist for NVIDIA compute capabilities 8.6
through 12.0:

| Profile | Representative GPU | Elementwise threads | Experimental router strategy |
| --- | --- | ---: | --- |
| `sm86` | RTX 3090 | 256 | fused GEMV, 128 threads, 2 rows/warp |
| `sm89` | RTX 4090 | 512 | fused GEMV, 128 threads, 1 row/warp |
| `sm90` | H100 | 512 | MLX GEMV + selection, 512 threads |
| `sm100` | B200 | 512 | MLX GEMV + selection, 512 threads |
| `sm120` | RTX 5090 | 512 | MLX GEMV + selection, 256 threads |

These definitions are not current performance claims. Only `sm86` has passed
the revised array-exact, deterministic long-decode campaign. Unknown future
capabilities use a conservative `future` profile, attempt Q/K only through the
same array-exact live probe, and otherwise fall back; unsupported devices stay
portable.

## Deterministic oracle

On the tested MLX 0.32.0 CUDA stack, repeated portable long generation could
diverge while cuDNN SDPA was eligible. The strict oracle therefore requires
`MLX_CUDA_USE_CUDNN_SDPA=0` before the first CUDA use. CUDA graph settings are
also process-level and must be fixed before model loading.

Correctness and timing are separate passes. Selected-token logprob and top-1
hashes strengthen the token/text gate, but they are not full-logit equality and
are not included in reported generation throughput.

## Dominant remaining cost

The generic affine 2-bit expert `qmm_naive` path accounted for about 34.6% of
profiled RTX 3090 GPU time. The next strict kernel must preserve FP32
accumulation order, BF16 rounding after each projection, original expert-slot
order, and the ordered FP32 down-projection aggregation. Native QMV/tile
controls tested so far did not provide a repeatable end-to-end win.
