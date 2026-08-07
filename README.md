# MLX Maple CUDA kernels

Fail-closed CUDA research kernels for
[DeepGrove's Maple preview](https://github.com/deepgrove-ai/mlx-lm-deepgrove).
The strict lane preserves the stock token stream and array boundaries; known
approximate router, add/RMS, ternary, FlashHead, and KV-quantized paths remain
off by default.

> **Status:** independent community research, not an MLX or DeepGrove release
> and not a claim of official model-author support. Evidence is scoped to the
> exact GPUs, drivers, MLX 0.32.0, CUDA 12.9, checkpoint revision, and source
> hashes recorded below.

## Strict multi-architecture result

All five fresh NVIDIA targets passed the deterministic strict methodology under
architecture-bound sealed campaign revisions. Values are warm `B=1`, `L=1`,
128-token prompt / 512-token decode throughput. Ratios are paired geometric
means over 12 fresh model processes on one device instance; displayed tok/s
values are arithmetic means.

| GPU | CC | Portable | Exact Q/K default | Paired gain (95% CI) | Q/K + cached-LHS opt-in |
| --- | --- | ---: | ---: | ---: | ---: |
| RTX 3090 | `sm86` | 145.89 | **154.95** | **+6.06%** (+2.73%–+9.50%) | 159.73, **+9.37%** |
| RTX 4090 | `sm89` | 182.36 | **209.84** | **+15.31%** (+11.29%–+19.49%) | 214.66, **+18.01%** |
| H100 80GB HBM3 | `sm90` | 202.62 | **233.67** | **+15.24%** (+12.86%–+17.66%) | 246.34, **+21.54%** |
| B200 | `sm100` | 241.51 | **280.70** | **+16.28%** (+14.23%–+18.37%) | 297.47, **+23.37%** |
| RTX 5090 | `sm120` | 398.49 | **429.72** | **+7.84%** (+6.88%–+8.81%) | 438.01, **+9.92%** |

The cached-LHS mode is exact in these runs but remains opt-in: its cache is
process-global and keyed only by top-k. The conservative source default is the
exact-probed fused Q/K path alone. Full retained records are indexed by
[`results/summary.csv`](results/summary.csv).

### Exactness gate

For each of the five fresh multiarchitecture targets, the release gate required:

- shape, dtype, and value equality through `mx.array_equal` for live fused
  outputs, with all 24 Q/K layers active and no silent fallback;
- 144 deterministic stock W2 projection fingerprints; any tile candidate had
  to match all 144 arrays before timing;
- exact direct/random 1024-token output and a three-case multi-seed matrix;
- 20/20 fixed regression cases at both 512 and 1024 generated tokens, including
  token IDs, decoded text, selected-token logprob hash, and top-1 hash;
- timing in separate fresh processes without correctness instrumentation.

The 20-case slice is a regression harness, not a quality leaderboard. Disabling
cuDNN SDPA is required to make the stock oracle bit-stable and is not credited
as a Maple speedup.

## Blackwell RoPE rounding fix

On both B200 and RTX 5090, the original fused upper-half RoPE expression could
contract the opposite product from stock MLX. One FP32 rounding bit could cross
a BF16 midpoint. The `sm100` and `sm120` profiles now pin stock association:

```cuda
__fmaf_rn(value, rope_cos[p], __fmul_rn(paired, rope_sin[p]))
```

The fix was accepted independently on each SKU. B200 passed 2,048 isolation
comparisons with zero fixed mismatches (32 old-control mismatches); RTX 5090
passed an expanded 4,608 comparisons with zero fixed mismatches (63 old-control
mismatching elements). Balanced 16-process fixed/original tests found no
statistically significant slowdown on either device. The frozen boundary
fixture is [`tests/data/sm100_qk_rope_boundary.npz`](tests/data/sm100_qk_rope_boundary.npz).
Re-run it after any MLX or CUDA upgrade.

## Graph and W2 tuning

The reproducible campaign graph profile is cache 400, 100 ops/buffer, and
100 MB/buffer; it is not asserted per-SKU optimal. In the five-block graph
screen, that profile over the 20-op control was supported on RTX
4090 (+25.07%, `p=0.0148`) and RTX 5090 (+16.31%, `p=1.20e-5`). The B200
factorial ops effect was +10.90% (`p=6.82e-5`); H100 graph effects were
inconclusive on its single tested instance.

The experimental MLX `qmm_naive` tile screen is separate from this package.
`16x32x128` passed the complete RTX 5090 follow-up and improved fresh-process
throughput by +1.615% (95% CI +1.322%–+1.909%, 12/12 wins,
`p=9.63e-8`). It is an accepted tuning result but is **not bundled as the stock
MLX backend**. RTX 4090 and B200 candidates were array-exact but failed their
performance gates; H100 retained its stock tile.

## Strict defaults

| Path | Default | Strict status |
| --- | --- | --- |
| Q/K norm + partial RoPE/NoPE | auto-probed | array-exact on the listed SKUs/toolchains |
| Cached flat decode LHS | off | exact in campaign; lifecycle-limited opt-in |
| Router GEMV/softmax/top-8 | off | normalized scores not array-exact |
| Residual add + RMSNorm | off | changed deterministic long decode |
| Ternary up/gate GEMV | off | projection values not array-exact |
| FlashHead / KV quantization | off | approximate; excluded |

Unsupported shapes, policies, devices, compile failures, or failed live probes
fall back to portable MLX. `False` in the reported Q/K path state means safe
fallback, not accelerated success.

## NVIDIA QuickStart

```bash
git clone https://github.com/iamwavecut/mlx-maple-cuda-kernels.git
git clone https://github.com/deepgrove-ai/mlx-lm-deepgrove.git
cd mlx-lm-deepgrove
git checkout eba96c16158f032821b0bf374ea1421cfddef0a9
git apply --check ../mlx-maple-cuda-kernels/patches/mlx-lm-deepgrove-maple-cuda.patch
git apply ../mlx-maple-cuda-kernels/patches/mlx-lm-deepgrove-maple-cuda.patch

uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[cuda12]' 'mlx==0.32.0' 'mlx-cuda-12==0.32.0' rich huggingface_hub

hf download deepgrove/maple-preview-2bit-mlx \
  --revision 361db5da5e74ff6fcdd852d478e1f266ce11013a \
  --local-dir ./maple-preview-2bit-mlx

MLX_CUDA_USE_CUDNN_SDPA=0 \
MLX_ENABLE_TF32=0 \
MLX_USE_CUDA_GRAPHS=1 \
MLX_CUDA_GRAPH_CACHE_SIZE=400 \
MLX_MAX_OPS_PER_BUFFER=100 \
MLX_MAX_MB_PER_BUFFER=100 \
python ../mlx-maple-cuda-kernels/examples/nvidia_generate.py \
  --model ./maple-preview-2bit-mlx \
  --prompt "Write a haiku about a maple grove." \
  --max-tokens 256
```

Pin `mlx==0.32.0` and the matching `mlx-cuda-12==0.32.0` wheel to reproduce the
published version claim. The example passes
`model_config={"model_file": None, "use_flash_head": False}`, uses
`trust_remote_code=False`, verifies the loaded package source, uses the exact LM
head, and prints strict path state. Add `--cached-lhs` only for the constrained
single-model/single-device warm workload described above.

## Evidence and provenance

- [`src/maple.py`](src/maple.py), SHA-256
  `28ceabac2b7570ff3712473c88eb7698b5a1904cd1b9cd55c698794fd457ccb8`;
- integration patch against DeepGrove `eba96c1`, SHA-256
  `eb9c36eb5aec3c93e52ddcc35d735f816a18ab5330460a05b7a641ba0f5174f0`;
- frozen fixture SHA-256
  `837638a799bef1b8ea7e7a23c77791964ca88f2bfc698f50910655c5f9bddb64`;
- [`results/PUBLIC-INDEX.json`](results/PUBLIC-INDEX.json), binding canonical
  analyses, manifests, source maps, and private raw-manifest commitments;
- detailed allowlisted artifacts under
  [`results/cuda/multiarch/`](results/cuda/multiarch/), plus compact strict,
  graph, W2, and Blackwell summaries in [`results/cuda/`](results/cuda/).

The baseline full-file source hashes were the release `28ceabac…` for
`sm86/sm120`, `7785da2a…` for `sm89/sm90`, and `b34cd977…` for
`sm100`. The release
changes are architecture-isolated; captured generated RoPE/NoPE kernel hashes
match the validated source for every profile. See
[`release-source-equivalence.json`](results/cuda/release-source-equivalence.json).
This is deliberately narrower than claiming whole-module equivalence.

## Known limitations

- Claims apply to the exact representative SKU, driver, MLX/CUDA version, model
  revision, and source provenance in the artifacts—not every GPU sharing a
  compute capability.
- Fresh Mac/Metal validation remains outstanding.
- Five `tests/test_generate.py` failures remain baseline-compatible on the
  tested checkout and are not claimed as fixed.
- Approximate router/add-RMS/ternary paths remain research-only.
- The accepted RTX 5090 W2 tile requires the separately built experimental MLX
  backend and is not enabled by this patch alone.

## License

Project code is MIT. Original Apple and DeepGrove notices are preserved; see
[`NOTICE.md`](NOTICE.md). The optional regression manifest obtains third-party
question content under its source licenses; see [`DATASET-NOTICE.md`](DATASET-NOTICE.md).
