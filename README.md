# MLX Maple CUDA kernels

Fail-closed CUDA research kernels for
[DeepGrove's Maple preview](https://github.com/deepgrove-ai/mlx-lm-deepgrove),
with a strict exact-output lane validated on `sm86`.

The source contains CUDA implementations and experimental prototypes for
residual add + RMSNorm, Q/K norm + RoPE/NoPE, router selection, and ternary
expert projections. Custom arithmetic paths need array-exact live probes to
enter the strict lane; non-arithmetic cleanup must preserve exact indices and
ordering. Known approximate paths are opt-in and disabled by default.

> **Status:** experimental research code, not an upstream MLX or DeepGrove
> release. The current strict evidence is for RTX 3090 / `sm86`; `sm89`,
> `sm90`, `sm100`, and `sm120` must be revalidated before release claims are
> extended to them.

## Current `sm86` results

MLX/MLX-CUDA 0.32.0, 128 prompt tokens, 512 generated tokens, deterministic
SDPA, CUDA graph cache 400, 100 ops/buffer, 100 MB/buffer. This is warm,
single-stream `B=1`, `L=1`, BF16 decode; JIT/live-probe and cold-cache costs
are excluded, while prefill, batched decode, and scaled-RoPE policies use
portable paths. Ratios are paired geometric means; displayed throughput values
are arithmetic means.

| Strict configuration | Portable MLX | Strict | Paired gain | 95% CI | Pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Conservative default: exact Q/K norm + RoPE | 179.39 tok/s | **195.27 tok/s** | **+8.51%** | +0.90%–+16.70% | 8 |
| Measured profile: Q/K + cached decode LHS | 177.56 tok/s | **209.58 tok/s** | **+18.28%** | +5.18%–+33.00% | 6 |

These small-n paired intervals are exploratory after extensive tuning, with a
host co-tenant active and no multiple-testing correction; they are not a
population-level hardware guarantee.

The old `+38.3%` strict claim is superseded: its oracle was too short, used
tolerant probes, and admitted router/add-RMS kernels that later diverged under
a deterministic long-decode oracle. Its 189.30 tok/s absolute observation is
retained historically, but is not directly comparable to 209.58 tok/s because
the generation length, baseline, graph configuration, oracle, and active paths
all changed.

Cached LHS is array-exact, but its isolated factorial main effect was only
+2.10% with a CI crossing zero (`p=0.345`), so it remains off by default. The
Q/K main effect was +10.00% (`p=0.044`). These are warm steady-state
decode measurements after live probes/warmup; JIT and cold cache construction
are excluded. See
[`results/cuda/sm86-component-factorial.jsonl`](results/cuda/sm86-component-factorial.jsonl).

### Graph settings

The supported tuning result is:

```sh
MLX_CUDA_USE_CUDNN_SDPA=0
MLX_USE_CUDA_GRAPHS=1
MLX_CUDA_GRAPH_CACHE_SIZE=400
MLX_MAX_OPS_PER_BUFFER=100
MLX_MAX_MB_PER_BUFFER=100
```

Set these before the first CUDA use. Raising ops/buffer from 20 to 100 had a
+11.85% factorial main effect in the four A-D blocks run at cache 2000. A
separate cache-size factorial at 100 ops / 1000 MB found no supported benefit
from 2000 over 400, and focused 100 MB vs 1000 MB pairs at cache 400 found no
supported MB benefit.
The recommended 100/100/cache-400 profile is the one used for the 209.58 tok/s
result.

## Exactness evidence

With `MLX_CUDA_USE_CUDNN_SDPA=0` and the exact LM head:

- the Q/K-only default matched the portable 512-token hash in every balanced
  component-factorial block, in addition to its per-layer array-exact probes;
- the Q/K + cached-LHS speed profile matched a random-prompt portable gate for
  1024/1024 token IDs;
- that same speed profile matched a fixed audited 20-case slice for every
  emitted token, decoded text, selected-token logprob hash, and top-1 hash with
  generation caps of 512 and 1024 tokens;
- the final CUDA focused suite passed: **20 passed, 2 skipped**;
- unsupported shapes, dtypes, devices, or numerical changes fail closed to
  portable MLX.

This does not claim equality of every full-logit tensor or exhaustive quality
coverage. The 20-case slice is a regression harness; most cases hit the token
limit, so its 3/20 and 4/20 scores are not representative quality estimates.

Repeated portable long decode was itself unstable while MLX could select the
cuDNN SDPA path. Disabling cuDNN SDPA made the oracle deterministic, so that
environment setting is part of the current strict contract rather than a Maple
speedup.

## Default and experimental paths

| Path | Default | Strict status |
| --- | --- | --- |
| Q/K norm + partial RoPE/NoPE | auto-probed | array-exact on validated `sm86` |
| Cached flat decode LHS | off | array-exact, opt-in; marginal speed not established |
| Router GEMV/softmax/top-8 | off | normalized scores are not array-exact; semantic only |
| Residual add + RMSNorm | off | diverged at generated token 217; semantic only |
| Ternary up/gate GEMV | off | faster projection, but BF16 values differed; experimental |
| FlashHead / KV quantization | off | approximate; excluded |

## Repository contents

- [`src/maple.py`](src/maple.py) and [`src/switch_layers.py`](src/switch_layers.py):
  readable snapshots of the patched implementation.
- [`patches/mlx-lm-deepgrove-maple-cuda.patch`](patches/mlx-lm-deepgrove-maple-cuda.patch):
  patch against DeepGrove commit `eba96c16158f032821b0bf374ea1421cfddef0a9`.
- [`tests/test_maple_kernels.py`](tests/test_maple_kernels.py): exact probes,
  fallback tests, architecture profiles, dependency/race checks, and defaults.
- [`benchmarks/`](benchmarks): correctness, common-slice, factorial, tuning,
  router, and ternary harnesses, plus a pinned fixed-slice input generator.
- [`results/`](results): sanitized trials, paired statistics, and superseded
  initial-port results retained as historical evidence.

The frozen laboratory implementation is commit
`b3d03fb19b522f307d0df7ba2ea347711a2ee337`; published `src/maple.py` has
SHA-256 `7785da2a85b97b9fd7759d8756b1daf2231ec8b912d42b4b7bc9c04637b371ae`.

## Quick start

See [`docs/integration.md`](docs/integration.md) for the complete pinned
workflow. In short:

```bash
git clone https://github.com/deepgrove-ai/mlx-lm-deepgrove.git
cd mlx-lm-deepgrove
git checkout eba96c16158f032821b0bf374ea1421cfddef0a9
git apply --check /path/to/mlx-maple-cuda-kernels/patches/mlx-lm-deepgrove-maple-cuda.patch
git apply /path/to/mlx-maple-cuda-kernels/patches/mlx-lm-deepgrove-maple-cuda.patch
cp -R /path/to/mlx-maple-cuda-kernels/benchmarks/. benchmarks/
uv venv --python 3.12
uv pip install -e '.[cuda12]' rich pytest
python benchmarks/prepare_maple_common_slice.py
```

Load the package implementation rather than the checkpoint-local model:

```python
from mlx_lm import load

model, tokenizer = load(
    model_path,
    model_config={"model_file": None, "use_flash_head": False},
)
```

## Known limitations

- Fresh current-source GPU validation is limited to `sm86`; Metal and
  `sm89`-`sm120` are pending.
- Five `tests/test_generate.py` fixture failures remain baseline-compatible on
  the tested checkout and are not presented as resolved by this patch.
- Cached LHS has process-global, top-k-only cache identity and is therefore
  opt-in for single-device steady-state use.
- The fixed 20-case slice is a regression harness, not a quality estimate.

## Remaining bottleneck

On the RTX 3090 exact-head legacy profile, MLX CUDA's generic affine 2-bit
expert `qmm_naive` accounted
for about 34.6% of profiled GPU kernel time. B=8 QMV and several native tile
variants did not produce a repeatable strict win. The next major strict target
is an affine W2 top-8 multi-row kernel that preserves FP32 accumulation order,
BF16 projection boundaries, expert-slot order, and ordered aggregation.

## License

Project code is MIT. Original Apple and DeepGrove notices are preserved; see
[`NOTICE.md`](NOTICE.md). The optional generated regression manifest obtains
third-party question content under its source licenses; see
[`DATASET-NOTICE.md`](DATASET-NOTICE.md).
