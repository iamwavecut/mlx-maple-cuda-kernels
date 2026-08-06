# Integration

## Requirements

- Linux with an NVIDIA GPU. Current strict evidence is for RTX 3090 / `sm86`.
- Python 3.12.
- MLX and MLX CUDA 0.32.0.
- The pinned Maple preview checkpoint.
- A trusted DeepGrove checkout at the base revision below.

Other CUDA profiles are fail-closed in source but have not passed the revised
long-decode campaign. Do not treat profile presence as current `sm89`-`sm120`
validation.

## Apply the patch

```bash
git clone https://github.com/deepgrove-ai/mlx-lm-deepgrove.git
cd mlx-lm-deepgrove
git checkout eba96c16158f032821b0bf374ea1421cfddef0a9

git apply --check /path/to/mlx-maple-cuda-kernels/patches/mlx-lm-deepgrove-maple-cuda.patch
git apply /path/to/mlx-maple-cuda-kernels/patches/mlx-lm-deepgrove-maple-cuda.patch
cp -R /path/to/mlx-maple-cuda-kernels/benchmarks/. benchmarks/

uv venv --python 3.12
uv pip install -e '.[cuda12]' rich pytest
source .venv/bin/activate

# Fetch and verify the separately licensed fixed-slice questions.
python benchmarks/prepare_maple_common_slice.py
```

Download the frozen checkpoint:

```bash
hf download deepgrove/maple-preview-2bit-mlx \
  --revision 361db5da5e74ff6fcdd852d478e1f266ce11013a \
  --local-dir maple-preview-2bit-mlx
```

The checkpoint config may name its bundled `maple.py`. Every strict harness
must override that and disable FlashHead at load time:

```python
from mlx_lm import load

model, tokenizer = load(
    "maple-preview-2bit-mlx",
    model_config={"model_file": None, "use_flash_head": False},
    tokenizer_config={"trust_remote_code": True},
    trust_remote_code=True,
)
```

Do not rely on importing `mlx_lm.models.maple` alone. Assert
`inspect.getfile(type(model))` against the intended worktree module when
collecting evidence.

## Set the deterministic environment

These values must be set in a fresh process before any CUDA use:

```bash
export MLX_CUDA_USE_CUDNN_SDPA=0
export MLX_USE_CUDA_GRAPHS=1
export MLX_CUDA_GRAPH_CACHE_SIZE=400
export MLX_MAX_OPS_PER_BUFFER=100
export MLX_MAX_MB_PER_BUFFER=100
```

The cuDNN setting is required for a stable portable reference oracle on the
tested stack. The graph settings are the supported `sm86` throughput profile.

## Verify before serving

Run the focused suite through the project environment:

```bash
PYTHONPATH="$PWD" python -m pytest tests/test_maple_kernels.py -q
```

Then run correctness and timing separately. The model harness performs a
1024-token exact gate before six paired 512-token timing trials:

```bash
PYTHONPATH="$PWD" python benchmarks/maple_model_benchmark.py \
  --model maple-preview-2bit-mlx \
  --output results/reference-vs-strict.jsonl \
  --prompt-tokens 128 \
  --generation-tokens 512 \
  --equivalence-tokens 1024 \
  --trials 6
```

Run the fixed regression slice in separate processes for each length:

```bash
PYTHONPATH="$PWD" python benchmarks/maple_common_slice_benchmark.py \
  --model maple-preview-2bit-mlx \
  --manifest benchmarks/data/maple_common_slice_20.json \
  --output results/common-512.jsonl \
  --max-tokens 512

PYTHONPATH="$PWD" python benchmarks/maple_common_slice_benchmark.py \
  --model maple-preview-2bit-mlx \
  --manifest benchmarks/data/maple_common_slice_20.json \
  --output results/common-1024.jsonl \
  --max-tokens 1024
```

Both scripts record source provenance; the common-slice harness also refuses a
loaded-class/worktree-module mismatch. They abort on approximate-path
activation or reference/strict token divergence and record live-path
acceptance/fallbacks;
a strict run may legitimately contain portable fallbacks.

## Runtime modes

The source default is conservative:

- exact-probed Q/K can enable itself;
- cached decode LHS is off;
- approximate router and add/RMS are off;
- ternary up/gate is off;
- FlashHead is off through model configuration.

The published 209.58 tok/s speed profile explicitly enables cached LHS. To
reproduce it, use `maple_model_benchmark.py`, which sets and records all state.
For an application-specific opt-in after its own equality gate:

```python
from mlx_lm.models import maple
maple._use_cached_decode_lhs = True
```

Set the global on the actual module used by the loaded class. Do not enable
`_use_approximate_router`, `_use_approximate_add_rms`, or
`_use_cuda_ternary_up_gate` in a strict deployment.

## Interpreting results

Token and decoded-text equality are the release gate. Selected-logprob and
top-1 hashes are additional diagnostics, not full-logit equality. Throughput
must come from the timing pass without per-token correctness instrumentation.

The 20-case slice is intentionally small and often length-limited. It detects
regressions; it is not a model-quality leaderboard.
