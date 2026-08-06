# Kernel coverage and design

The public Maple implementation contains three hand-written Metal fast paths.
This port implements CUDA equivalents for all three:

1. Residual add plus RMSNorm.
2. Per-head Q/K RMSNorm plus partial RoPE or NoPE.
3. Router GEMV plus FP32 softmax and stable top-8 selection.

The kernels are created lazily through `mx.fast.cuda_kernel`; no separate
PyTorch extension or precompiled shared object is required. CUDA Graphs remain
owned by MLX and are not replaced by this patch.

## Architecture profiles

The current profiles cover NVIDIA compute capabilities 8.6 through 12.0:

| Profile | Representative GPU | Elementwise threads | Router strategy |
| --- | --- | ---: | --- |
| `sm86` | RTX 3090 | 256 | fused GEMV, 128 threads, 2 rows/warp |
| `sm89` | RTX 4090 | 512 | fused GEMV, 128 threads, 1 row/warp |
| `sm90` | H100 | 512 | MLX GEMV plus fused softmax/top-8, 512 threads |
| `sm100` | B200 | 512 | MLX GEMV plus fused softmax/top-8, 512 threads |
| `sm120` | RTX 5090 | 512 | MLX GEMV plus fused softmax/top-8, 256 threads |

Unknown future capabilities use the portable modern-router profile. Devices
older than `sm86` stay on stock MLX operators.

## Correctness and fallback

Each fast path has a live probe against the portable MLX implementation. A
compile error, unsupported shape or dtype, failed numerical check, or scaled
RoPE policy that the JIT kernel does not implement latches that path back to
portable MLX instead of returning unchecked output.

Router logits and accumulation remain FP32. Selection order is checked because
equal-score experts can change floating-point aggregation order even when the
selected set is the same. The focused suite also stresses repeated router
dispatches to catch counter-reset races.

The published quality gate is exact equality of 256 greedy output token IDs
between portable and accelerated paths on each representative architecture.
This is stronger than comparing decoded text, but it is not a claim that every
possible prompt, sampling policy, or checkpoint has been exhaustively proven.
