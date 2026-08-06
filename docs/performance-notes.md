# Performance notes

Porting all three public Metal kernels does not remove every CUDA bottleneck.
On RTX 3090 (`sm86`), an Nsight Systems audit of exact decode attributed about
34.6% of GPU kernel time to MLX's generic 2-bit expert `qmm_naive` path. The
next notable shares were non-gather 2-bit QMV (10.4%), router work (9.6%), the
exact 4-bit LM-head QMV (7.9%), Q/K fusion (6.4%), SDPA (4.8%), and residual
add/RMSNorm (3.9%). Percentages are workload-specific and can overlap with CPU
and dispatch effects; the raw profiler database is intentionally not shipped.

This explains why matching Metal kernel coverage produces a substantial but
not Mac-like multiple on CUDA: the remaining expert matmul dominates. It is a
generic MLX CUDA quantized-matmul path, not a missing Maple Metal kernel.

One exact experiment split top-8 expert execution into two top-4 calls to force
a QMV path. It preserved the 256-token output but reduced end-to-end throughput
from 219.7 to 153.0 tok/s (-30.4%), so it is documented rather than shipped.

CUDA Graphs were already active in MLX. Replacing or duplicating them was not a
missing optimization in the Maple port.
