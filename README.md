# MLX Maple CUDA kernels

CUDA kernels that make
[DeepGrove's Maple preview](https://github.com/deepgrove-ai/mlx-lm-deepgrove)
(ternary MoE, 2-bit, 256 experts / top-8) decode **2-3x faster on NVIDIA
GPUs under MLX — while reproducing the stock token stream bit for bit.**

> Independent community research, not an MLX or DeepGrove release. Claims
> are scoped to the exact GPUs, drivers, MLX 0.32.0 / CUDA 12.9, checkpoint
> and source hashes in [`results/`](results/). Latest release:
> [v0.8.1](https://github.com/iamwavecut/mlx-maple-cuda-kernels/releases).

## Why it exists

Decode on MLX/CUDA is **host-bound**: with a warm cache the GPU finishes a
step in microseconds and waits ~3 µs while Python builds the next one —
wall clock is host operation count, not arithmetic. So the fix is not
faster math, it is *fewer operations*: this patch fuses an entire decode
layer into **two dispatches** — one attention megakernel (qkv projection,
Q/K norm + RoPE, KV-cache append, SDPA, o_proj) and one MoE megakernel
(router, 8 experts, SwiGLU, aggregation, both residual norms). Every phase
re-derives the stock kernel's arithmetic bit for bit, so the speedup costs
nothing in reproducibility. Details:
[`docs/host-bound-decode.md`](docs/host-bound-decode.md).

## Results

Warm B=1 decode, 128-token prompt / 512 generated, medians over fresh
processes; each row is one rented host, so compare within rows:

| GPU | portable MLX | this patch | speedup | token stream |
| --- | ---: | ---: | ---: | --- |
| RTX 4090 | 157.7 | **455.7** | **×2.9** | bit-identical, 8/8 prompts |
| RTX 3090 | 152.3 | ~415 (345.2 + 20.6%) | ×2.7 | bit-identical, 8/8 |
| H100 80GB | 206.8 | 388.6 | ×1.9 | bit-identical, 8/8 |
| RTX 5090 | 242.6 | 381.6 | ×1.6 | bit-identical, 8/8 |
| B200 | 141.9 | 358.0 | ×2.5 | bit-identical, 8/8 |

Long contexts hold: the attention lane runs the stock 2-pass SDPA in-kernel
past 1024 keys (buffers grow to 8192), bit-validated through every boundary
including in-flight growth. Bit-exactness is not sampled, it is gated: live
per-layer probes at load, screened-prompt stream equality, and an 846-token
quality suite that reproduces the reference NLL to the last digit.

Defaults are data-driven per architecture: the attention lane is auto-on
where measured faster (sm86/sm89) and auto-off elsewhere (H100/5090
measured slower on healthy-CPU hosts). Cross-request isolation against
serving LRU caches is gated by `benchmarks/maple_lru_service_repro.py` —
green 3/3 (chronicles #17-18). Every lane falls back to portable MLX on
any mismatch; `MAPLE_ATTENTION_MEGAKERNEL=0/1` forces the lane,
`MAPLE_MOE_MEGAKERNEL_EXACT=0 MAPLE_MOE_MEGAKERNEL=0` steps down to the
strict two-fusion lane (+7% over portable, also bit-identical).

## Use it

```bash
git clone https://github.com/iamwavecut/mlx-maple-cuda-kernels.git
git clone https://github.com/deepgrove-ai/mlx-lm-deepgrove.git
cd mlx-lm-deepgrove
git checkout eba96c16158f032821b0bf374ea1421cfddef0a9
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

Sharp edges: the driver must be newer than 550.163; a CUDA 13 toolkit in
`$CUDA_HOME` silently breaks every custom kernel (point it at 12.9
headers); the megakernels' grid barriers want the GPU to themselves — do
not benchmark on a shared device. Full list in
[`docs/DETAILS.md`](docs/DETAILS.md#known-limitations).

## More

- [`docs/DETAILS.md`](docs/DETAILS.md) — per-lane mechanics, the
  exactness protocol, the dated chronicle of how each stage landed, full
  campaign tables, known limitations.
- [`docs/host-bound-decode.md`](docs/host-bound-decode.md) — the
  measurement study behind the design.
- [`results/`](results/) — evidence: campaign records, provenance
  manifests, per-architecture artifacts.
- Provenance: [`src/maple.py`](src/maple.py) SHA-256
  `2ec5f36f1ff367fdd4046f3617522d079de9a95ad0cdfdcaf8edb01ec9161542`,
  patch SHA-256
  `ce985ed3246657f1e5293808f935c28e9d0cf853308b4407d7d80107c634c9f2`.

## License

MIT for project code; Apple and DeepGrove notices preserved
([`NOTICE.md`](NOTICE.md)); dataset notices in
[`DATASET-NOTICE.md`](DATASET-NOTICE.md).
