# Integration

## Validated scope

- Linux, Python 3.12.3, MLX/MLX-CUDA 0.32.0, CUDA 12.9.
- Fresh strict evidence on one RTX 4090 (`sm89`), H100 80GB HBM3 (`sm90`),
  B200 (`sm100`), and RTX 5090 (`sm120`) instance.
- Checkpoint `deepgrove/maple-preview-2bit-mlx` revision
  `361db5da5e74ff6fcdd852d478e1f266ce11013a`.
- DeepGrove base `eba96c16158f032821b0bf374ea1421cfddef0a9`.

Other SKUs, drivers, MLX versions, CUDA 13, future architectures, and execution
policies require fresh validation. Profile presence alone is not a support
claim.

## Apply the patch

```bash
git clone https://github.com/deepgrove-ai/mlx-lm-deepgrove.git
cd mlx-lm-deepgrove
git checkout eba96c16158f032821b0bf374ea1421cfddef0a9

git apply --check /path/to/mlx-maple-cuda-kernels/patches/mlx-lm-deepgrove-maple-cuda.patch
git apply /path/to/mlx-maple-cuda-kernels/patches/mlx-lm-deepgrove-maple-cuda.patch
cp -R /path/to/mlx-maple-cuda-kernels/benchmarks/. benchmarks/
mkdir -p tests/data
cp /path/to/mlx-maple-cuda-kernels/tests/data/sm100_qk_rope_boundary.npz tests/data/

uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[cuda12]' 'mlx==0.32.0' 'mlx-cuda-12==0.32.0' rich pytest huggingface_hub
```

For the published version claim, verify `mlx==0.32.0` and
`mlx-cuda-12==0.32.0`; do not substitute a CUDA 13 wheel and inherit the claim.
Download the frozen checkpoint:

```bash
hf download deepgrove/maple-preview-2bit-mlx \
  --revision 361db5da5e74ff6fcdd852d478e1f266ce11013a \
  --local-dir maple-preview-2bit-mlx
```

## Force package implementation

The checkpoint config may name bundled Python. Strict loading overrides it,
disables FlashHead, rejects remote code, and verifies the actual class source:

```python
import inspect
from pathlib import Path
from mlx_lm import load
from mlx_lm.models import maple

model, tokenizer = load(
    "maple-preview-2bit-mlx",
    model_config={"model_file": None, "use_flash_head": False},
    tokenizer_config={"trust_remote_code": False},
    trust_remote_code=False,
)
assert Path(inspect.getfile(type(model))).resolve() == Path(maple.__file__).resolve()
```

Importing `mlx_lm.models.maple` alone is not a provenance check.

## Deterministic process environment

Set all values in a fresh process before any CUDA use:

```bash
export MLX_CUDA_USE_CUDNN_SDPA=0
export MLX_ENABLE_TF32=0
export MLX_USE_CUDA_GRAPHS=1
export MLX_CUDA_GRAPH_CACHE_SIZE=400
export MLX_MAX_OPS_PER_BUFFER=100
export MLX_MAX_MB_PER_BUFFER=100
export TOKENIZERS_PARALLELISM=false
```

cuDNN SDPA is disabled to stabilize the portable oracle, not to claim a Maple
speedup. Graph settings are the reproducible campaign profile, not a universal
per-SKU optimum. Runtime tile alternation must instead use
`MLX_USE_CUDA_GRAPHS=0` to prevent graph/JIT identity contamination.

## Verify before serving

Run the focused tests through the project environment:

```bash
PYTHONPATH="$PWD" python -m pytest tests/test_maple_kernels.py -q
```

Then separate correctness from timing. The model harness performs an exact
1024-token gate before paired 512-token timing:

```bash
PYTHONPATH="$PWD" python benchmarks/maple_model_benchmark.py \
  --model maple-preview-2bit-mlx \
  --output results/reference-vs-strict.jsonl \
  --prompt-tokens 128 \
  --generation-tokens 512 \
  --equivalence-tokens 1024 \
  --trials 6
```

Generate the separately licensed, pinned fixed-slice manifest, then run the two
lengths in separate processes:

```bash
PYTHONPATH="$PWD" python benchmarks/prepare_maple_common_slice.py

PYTHONPATH="$PWD" python benchmarks/maple_common_slice_benchmark.py \
  --model maple-preview-2bit-mlx \
  --manifest benchmarks/data/maple_common_slice_20.json \
  --output results/common-512.jsonl --max-tokens 512

PYTHONPATH="$PWD" python benchmarks/maple_common_slice_benchmark.py \
  --model maple-preview-2bit-mlx \
  --manifest benchmarks/data/maple_common_slice_20.json \
  --output results/common-1024.jsonl --max-tokens 1024
```

A strict run may legitimately report portable fallback. Acceleration is claimed
only when all 24 Q/K layers are active and all other path-state/correctness gates
pass. Token/text/selected-logprob/top-1 equality is finite evidence, not
exhaustive full-logit equality or a quality benchmark.

## Runtime modes

The source default enables only exact-probed Q/K. Cached decode LHS is off;
approximate router, add/RMS, ternary up/gate, FlashHead, and KV quantization are
off. For the constrained warm single-model/single-device top-k=8 workload:

```python
from mlx_lm.models import maple
maple._use_cached_decode_lhs = True
```

Its process-global cache lacks model/device invalidation, so keep it off when a
process changes model or device. The accepted RTX 5090 W2 tile requires a
separate experimental MLX wheel and is not enabled by this repository patch.
