# Kernel coverage and design

## Strict boundary

The conservative CUDA fast path is per-head Q/K RMSNorm fused with partial
RoPE or NoPE. Every layer must pass a shape/dtype/value live comparison through
`mx.array_equal`; compile errors, unsupported policies, and mismatches latch to
portable MLX. The exact LM head is required.

Cached flat decode `lhs_indices` was exact in the current campaign but remains
an explicit opt-in because its cache is process-global and keyed only by top-k.
FlashHead, KV quantization, and operation/reduction-order changes are outside
the strict boundary.

## CUDA paths

### Q/K norm + RoPE/NoPE

The fused decode kernel preserves FP32 norm accumulation and BF16 output
boundaries. `sm100` and `sm120` explicitly pin the stock MLX upper-half RoPE
rounding graph; see [`strict-exact-validation.md`](strict-exact-validation.md).

### Residual add + RMSNorm

CUDA and Metal implementations remain available only for semantic studies. A
local probe was insufficient: deterministic long decode first changed at token
217. `_use_approximate_add_rms` is therefore `False` by default.

### Router

The experimental router can fuse GEMV, FP32 softmax, and stable top-8 selection,
but its normalized FP32 scores are not array-exact. Strict mode retains stock
matmul, softmax, argpartition, gather, and renormalization.

### Ternary expert prototype

The checkpoint's structured 2-bit up/gate tensors permit a faster ternary
prototype, but full-layer BF16 outputs differ. `_use_cuda_ternary_up_gate`
remains `False`; stock `QuantizedSwitchLinear` is the strict default.

### Decode LHS cache

The cached decode-row index sequence is exact for the tested top-k=8,
single-model/single-device warm workload. It defaults off because device/model
invalidation is not encoded in its identity.

## Architecture profiles and evidence

| Profile | Tested GPU | Driver | Elementwise threads | Router fallback strategy | Fresh strict Q/K gate |
| --- | --- | --- | ---: | --- | --- |
| `sm86` | RTX 3090 | 580.159.03 | 256 | portable in strict mode | pass |
| `sm89` | RTX 4090 | 580.159.04 | 512 | portable in strict mode | pass |
| `sm90` | H100 80GB HBM3 | 580.126.09 | 512 | portable in strict mode | pass |
| `sm100` | B200 | 580.126.20 | 512 | portable in strict mode | pass after RoPE fix |
| `sm120` | RTX 5090 | 580.126.20 | 512 | portable in strict mode | pass after independent RoPE fix |

The router launch strategy stored in each profile is experimental and is not
enabled in strict mode. These are exact representative-SKU claims, not blanket
support for every GPU sharing a compute capability. Unknown capabilities use a
conservative future profile and must pass their own live probes.

The release source was directly run on `sm86` and `sm120`. The earlier runs used
recorded full-file source hashes; captured generated RoPE/NoPE kernel hashes
match the release for `sm89`, `sm90`, and `sm100`. That is a code-generation
isolation check, not whole-module rerun equivalence.

## Deterministic oracle

On MLX 0.32.0, portable long generation was not bit-stable while cuDNN SDPA was
eligible. Before the first CUDA use, strict runs set cuDNN SDPA off, TF32 off,
and freeze graph policy. Correctness and timing run separately; selected-logprob
and top-1 instrumentation does not enter throughput.

## Dominant remaining cost

Generic affine 2-bit expert `qmm_naive` remains a major cost. An experimental
`16x32x128` tile passed the RTX 5090 exact/performance follow-up, but it requires
a separately built MLX backend and is not shipped here. Future W2 work must
preserve FP32 accumulation order, BF16 projection boundaries, expert-slot
order, and ordered down-projection aggregation.
