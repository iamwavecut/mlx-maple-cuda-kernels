# Integration

## Requirements

- Linux with an NVIDIA GPU of compute capability 8.6 or newer.
- Python 3.12.
- MLX and MLX CUDA 0.32.0 (the tested release).
- The Maple preview checkpoint.
- A trusted checkout of `deepgrove-ai/mlx-lm-deepgrove` at the pinned base
  commit below.

The Maple checkpoint executes its bundled `maple.py` when
`trust_remote_code=True`. Review and pin the checkpoint revision before using
that flag.

## Apply the port

```bash
git clone https://github.com/deepgrove-ai/mlx-lm-deepgrove.git
cd mlx-lm-deepgrove
git checkout eba96c16158f032821b0bf374ea1421cfddef0a9

git apply --check /path/to/mlx-maple-cuda-kernels/patches/mlx-lm-deepgrove-maple-cuda.patch
git apply /path/to/mlx-maple-cuda-kernels/patches/mlx-lm-deepgrove-maple-cuda.patch
cp /path/to/mlx-maple-cuda-kernels/benchmarks/maple_*.py benchmarks/

uv venv --python 3.12
uv pip install -e '.[cuda12]' rich
source .venv/bin/activate
```

Use a locally downloaded and pinned checkpoint:

```bash
hf download deepgrove/maple-preview-2bit-mlx \
  --revision 361db5da5e74ff6fcdd852d478e1f266ce11013a \
  --local-dir maple-preview-2bit-mlx
```

For CUDA inference, clear the checkpoint-local model override so the patched
package model is used:

```bash
cp maple-preview-2bit-mlx/config.json maple-preview-2bit-mlx/config.json.original
jq '.model_file = null' maple-preview-2bit-mlx/config.json.original \
  > maple-preview-2bit-mlx/config.json.tmp
mv maple-preview-2bit-mlx/config.json.tmp maple-preview-2bit-mlx/config.json
```

Restore `config.json.original` to return to the checkpoint-bundled model.

## Verify before serving

```bash
PYTHONPATH="$PWD" python tests/test_maple_kernels.py -v

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
  python benchmarks/maple_model_benchmark.py \
  --model maple-preview-2bit-mlx \
  --output results/reference-vs-auto.jsonl \
  --prompt-tokens 128 \
  --generation-tokens 256 1024 \
  --equivalence-tokens 256 \
  --trials 5
```

The benchmark aborts if any live fast path is disabled or if the exact token
sequence diverges. It alternates reference/accelerated order between trials to
reduce ordering bias.

## Use

No inference flag is required. Supported fast paths are selected lazily after
their first live correctness probe. Existing `mlx_lm.generate`, chat, and
server entry points continue to work.

To force the portable implementation for a comparison, use
`benchmarks/maple_model_benchmark.py`; its `reference` mode disables all three
fast paths in the same loaded process.
