# MLX Maple CUDA kernels

An exact-output CUDA port of all three hand-written Metal fast paths in
[DeepGrove's Maple preview](https://github.com/deepgrove-ai/mlx-lm-deepgrove):
residual add plus RMSNorm, Q/K norm plus RoPE/NoPE, and router GEMV plus FP32
softmax/top-8.

The implementation uses `mx.fast.cuda_kernel`, selects tuned profiles from
`sm86` through `sm120`, and falls back to portable MLX when a live correctness
probe fails. It keeps the exact LM head; approximate FlashHead results are not
mixed into the speedup claims below.

> **Status:** experimental, reproducible research code. The representative
> architecture suite and 256-token exact-generation gates pass, but this is not
> an upstream MLX or DeepGrove release.

## Results

Paired end-to-end decode, 128-token prompt, 256 generated tokens, MLX 0.32.0.
Values are arithmetic means from the published JSONL trials.

| GPU | CC | Portable MLX | CUDA fast paths | Gain | Exact 256 tokens |
| --- | --- | ---: | ---: | ---: | --- |
| RTX 3090 (GPU2) | `sm86` | 136.86 tok/s | 189.30 tok/s | **+38.3%** | yes |
| RTX 4090 | `sm89` | 120.56 tok/s | 167.79 tok/s | **+39.2%** | yes |
| H100 NVL | `sm90` | 216.25 tok/s | 283.42 tok/s | **+31.1%** | yes |
| B200 | `sm100` | 149.16 tok/s | 167.98 tok/s | **+12.6%** | yes |
| RTX 5090 | `sm120` | 228.76 tok/s | 262.08 tok/s | **+14.6%** | yes |

The fresh GPU2 run shared the host with the required embedder. The other rows
are dedicated cloud-pod snapshots. Absolute speed is therefore less comparable
across rows than each paired within-host gain. B200 was especially variable;
the raw trial distribution is retained rather than smoothed away.

### Fresh M2 Max exact baseline

Apple M2 Max, 38 GPU cores, 64 GB unified memory, original Metal kernels,
exact LM head, five trials on 2026-08-06:

| Prompt / generation | Mean | Median | Range |
| --- | ---: | ---: | ---: |
| 128 / 256 | **172.90 tok/s** | 173.06 | 170.55–174.52 |
| 128 / 1024 | **170.41 tok/s** | 170.84 | 168.86–171.74 |

This Mac was in normal interactive use and is tagged
`interactive_non_quiescent`; it is a local baseline, not a peak claim.
DeepGrove currently lists 169 tok/s exact on M4 and 359 tok/s exact on M5 Pro,
but their host conditions are not specified, so those numbers are context, not
a controlled cross-machine benchmark. Their faster FlashHead rows are
approximate and intentionally excluded here.

## What is included

- [`src/maple.py`](src/maple.py): readable snapshot of the patched Maple model.
- [`patches/mlx-lm-deepgrove-maple-cuda.patch`](patches/mlx-lm-deepgrove-maple-cuda.patch):
  patch against DeepGrove commit `eba96c16158f032821b0bf374ea1421cfddef0a9`.
- [`tests/test_maple_kernels.py`](tests/test_maple_kernels.py): focused numerical,
  fallback, architecture-profile, race, and CLI checks.
- [`benchmarks/`](benchmarks): kernel, end-to-end, ablation, profile-sweep, and
  M2 Max baseline harnesses.
- [`results/`](results): sanitized per-trial data and summaries. No GPU UUIDs,
  pod IDs, local paths, raw consoles, model weights, or profiler databases.

## Quick start

See [the integration guide](docs/integration.md) for the pinned checkout,
checkpoint preparation, MLX CUDA installation, exact-generation gate, and
benchmark command. The short form is:

```bash
git checkout eba96c16158f032821b0bf374ea1421cfddef0a9
git apply /path/to/mlx-maple-cuda-kernels/patches/mlx-lm-deepgrove-maple-cuda.patch
cp /path/to/mlx-maple-cuda-kernels/benchmarks/maple_*.py benchmarks/
uv venv --python 3.12
uv pip install -e '.[cuda12]' rich
PYTHONPATH="$PWD" .venv/bin/python tests/test_maple_kernels.py -v
```

Supported devices use the fast paths automatically after their live probes;
there is no serving-time flag.

## Why the CUDA gain is not a Metal-sized multiple

Nothing from the three public Metal kernels is missing: all three are ported.
The largest remaining GPU2 decode cost is MLX CUDA's generic 2-bit expert
matmul (`qmm_naive`), measured at about 34.6% of GPU kernel time. An exact
two-by-top-4 workaround forced a QMV path but made end-to-end decode 30.4%
slower, so it was rejected.

See [performance notes](docs/performance-notes.md),
[kernel design](docs/architecture.md), and
[benchmark methodology](docs/benchmark-methodology.md) for the boundaries and
reproducibility details.

## License

MIT. Original Apple and DeepGrove notices are preserved; see
[`NOTICE.md`](NOTICE.md).
