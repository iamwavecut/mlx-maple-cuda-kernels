# MLX Maple CUDA kernels

Fail-closed CUDA research kernels for
[DeepGrove's Maple preview](https://github.com/deepgrove-ai/mlx-lm-deepgrove)
(ternary MoE, 2-bit affine, 256 experts / top-8) on MLX 0.32.0.

**Current state, in one sentence:** a decode layer is **two dispatches** —
one array-exact attention megakernel and one array-exact MoE megakernel —
and the default configuration **reproduces the stock token stream bit for
bit** while decoding 2-3x faster than portable MLX.

> **Status:** independent community research, not an MLX or DeepGrove release
> and not a claim of official model-author support. There are no GitHub
> releases: `main` plus the pinned hashes below **is** the artifact. Evidence
> is scoped to the exact GPUs, drivers, MLX 0.32.0, CUDA 12.9, checkpoint
> revision, and source hashes recorded in `results/`.

## Current state

| Path | Default | Status |
| --- | --- | --- |
| Attention megakernel (`MAPLE_ATTENTION_MEGAKERNEL`) | **on** | array-exact; the whole decode attention block in one dispatch — fused 2-bit qkv projection, Q/K RMSNorm + partial RoPE, KV-cache append, SDPA, o_proj. kL ≤ 1024 runs the 1-pass SDPA port; longer contexts the 2-pass port, full-attention buffers growing 1024→8192. All (re)seeding is kernel-side so the persistent buffers never move |
| Exact MoE megakernel (`MAPLE_MOE_MEGAKERNEL_EXACT`) | **on** | array-exact; router, 8 gathered experts, SwiGLU, aggregation, both surrounding add/RMSNorm and the next layer's carrier in one dispatch; stock stream on 8/8 screened prompts on all five architectures |
| ~1 ULP MoE megakernel (`MAPLE_MOE_MEGAKERNEL`) | **on** (fallback) | within ~1 ULP of bf16; runs only where the exact plan declines (non-standard geometry) |
| Fused QKV split (`_use_fused_qkv`) | **on** | array-exact by construction; probed live |
| Residual add + RMSNorm (`_use_fused_add_rms`) | **on** | array-exact once the thread mapping matches `mx.fast.rms_norm` |
| Q/K norm + partial RoPE/NoPE | auto-probed | array-exact on the listed SKUs/toolchains |
| Compiled router (`MAPLE_COMPILED_ROUTER`) | off | array-exact; end-to-end effect not distinguishable from zero |
| Cached flat decode LHS | off | exact in campaign; lifecycle-limited opt-in |
| Router GEMV/softmax/top-8, original add/RMS kernel, ternary GEMV | off | not array-exact; research lanes |
| FlashHead / KV quantization | off | approximate; excluded |

Unsupported shapes, policies, devices, compile failures, or failed live
probes fall back to portable MLX. `False` in the reported path state means
safe fallback, not accelerated success.

Throughput of the default lane (warm `B=1`, `L=1`, 128-token prompt /
512-token decode, medians over fresh interleaved processes; each row is one
device instance on one host — for a host-bound workload the CPU moves the
number more than the GPU does):

| GPU | CC | strict fusions | exact MoE megakernel | + attention megakernel | stream vs stock |
| --- | --- | ---: | ---: | ---: | --- |
| RTX 3090 | `sm86` | 176.3 | 345.2 | **+20.6%** on top | 8/8 identical |
| RTX 4090 | `sm89` | 175.8 | 318.8 | **455.7** (+42.9%, spread ±1.7) | 8/8 identical |
| H100 80GB | `sm90` | 233.5 | 388.6 | not yet profiled | 8/8 identical |
| B200 | `sm100` | 322.1 | 358.0 | not yet profiled | 8/8 identical |
| RTX 5090 | `sm120` | 217.7 | 381.6 | not yet profiled | 8/8 identical |

The attention lane is bit-validated everywhere it engages (live per-layer
probes plus the stream/rotation/boundary suites below); its *performance*
profile on `sm90`/`sm100`/`sm120` is the outstanding follow-up. The RTX 4090
spread collapsing to ±1.7 tok/s is the host-bound step disappearing: two
dispatches per layer leave almost nothing for the host to do.

## Why: decode on CUDA is host-bound

The measurement that reframed this work: with a warm cache and MLX's
double-buffered `async_eval`, the **GPU wait per decode step is ~0.003 ms**.
The GPU finishes before Python has finished building the next step, on every
host tested. Wall clock is the sum of per-operation host costs, so the
currency of optimization is operation count, not arithmetic.

| Host | graph build | submission | GPU wait | step |
| --- | ---: | ---: | ---: | ---: |
| RTX 3090, EPYC 7452 | 2.40 ms | 3.96 ms | **0.0025 ms** | 6.36 ms |
| RTX 3090, AI Farm | 2.02 ms | 2.76 ms | **0.0028 ms** | 4.78 ms |
| RTX 4090 | 2.43 ms | 3.31 ms | **0.0029 ms** | 5.76 ms |
| H100 80GB HBM3 | — | — | **0.0040 ms** | — |
| B200 | — | — | **0.0030 ms** | — |
| RTX 5090 | — | — | **0.0022 ms** | — |

Kernels that make the GPU faster without removing operations do nothing (a
hand-written 2-bit expert GEMV at 1.88-1.98x in isolation moved end-to-end
throughput by roughly nothing), and it is worth spending GPU time to buy back
host operations — which is what both megakernels do. Full write-up:
[`docs/host-bound-decode.md`](docs/host-bound-decode.md).

## The lanes

### The exact MoE megakernel

One dispatch per MoE layer: router logits, online softmax, top-8 with the
stock argpartition tie order, renorm, eight gathered 2-bit experts, clamped
SwiGLU, score-weighted aggregation, the surrounding residual add/RMSNorm and
the next layer's carrier — five phases behind four atomic-counter grid
barriers.

Every phase is a bit recipe pinned against the stock kernel it replaces
(`benchmarks/maple_qmm_naive_repro.py`,
`benchmarks/maple_exact_lane_semantics.py`): dequant as `bf16(bf16(q·s)+z)`
into the same `m16n8k16` bf16 tensor-core atom with k-tiles of 128 in order;
the fp32 logits gemv; an exact online-softmax port; argpartition's returned
order (argsort's ascending tail, ties included); the renorm reduce whose
**shape picks the kernel** — a `(1,1,8)` sum dispatches `row_reduce_simple`,
a flat `(8,)` an all-reduce with different bits; and the aggregation as
`col_reduce_small`'s linear loop with `__fmul_rn` separate from `__fadd_rn`,
where letting the compiler contract to fma is exactly what breaks equality.

On the real checkpoint it is array-equal to the stock chain on every MoE
layer, its decode stream is identical to the stock reference on **8/8**
screened prompts on all five architectures, and the 846-token quality suite
reproduces the strict lane's corpus NLL to the last digit.

### The attention megakernel

The second dispatch: the whole decode attention block. Phase A computes the
fused 2-bit qkv projection with the stock `qmv` recipe (bf16 HFMA
accumulators, one packed word and one scale/bias pair per 128-tile); phase B
splits heads, applies Q/K RMSNorm and partial RoPE (only the first 64 of 128
dims rotate) line-for-line with the shipped exact split kernel and appends
K/V into persistent caller-owned buffers at the same physical slot the stock
rotating cache uses; phase C is the stock SDPA — the `kernel_sdpav_1pass`
port up to 1024 keys, and past that the `kernel_sdpav_2pass` port (32 slabs
of 8 warps per head, fp32 partials scaled to the slab max, one extra grid
barrier, a 32×32 merge block per head; standalone recipe 72/72 vs stock at
kL 1025-8192); phase D is the o_proj `qmv`.

Full-attention layers grow their buffers 1024 → 2048 → 4096 → 8192 (one
recompile per tier) and fall back past 8192. Sliding-window layers mirror
the stock ring exactly, including the concat-tail state a multi-token
prefill leaves on a rotated ring. The kernel advances its own on-device step
counters, and every (re)seed is written by a helper kernel through the input
pointers — the persistent buffers never move, so CUDA graphs capture once.
Leaving the lane (a new prefill, a fallback) flushes the fused buffers back
into the stock cache; re-entry re-seeds from it.

Bit evidence on `sm86`: 4/4 stream prompts over 256 tokens, ring rotation at
700 tokens 2/2, the 1024-boundary suite (cross-in-flight, start-past,
write-back + regrow, 4096-tier, 2048→4096 growth over 260 tokens) all
identical, multi-turn chains bit-equal, per-layer live probes bit-equal on
all 24 layers. `sm89` reproduces the stream suites and adds the +42.9%
end-to-end figure above.

### The ~1 ULP fallback and what it costs

Geometries the exact plan declines run the earlier megakernel, within
~1 ULP of bf16. Token equality does not survive that (near-tie argmax flips
on ~9% of scored tokens), so the question is quality, measured rather than
assumed on 846 scored tokens across five architectures:

| GPU | reference ppl | strict | megakernel | top-1 changed |
| --- | ---: | --- | ---: | ---: |
| RTX 3090 / 4090 | 33.1857 | identical | 32.8173 (−1.1%) | 78 / 846 |
| H100 80GB | 32.9462 | identical | 32.6704 (−0.8%) | 80 / 846 |
| RTX 5090 / B200 | 33.1647 | identical | 32.7340 (−1.3%) | 75 / 846 |

Perplexity *lower* by 0.8-1.3% is the signature of unbiased last-bit noise,
not a better or worse model; what it rules out is a quality regression.

### The strict fusions

`MAPLE_MOE_MEGAKERNEL_EXACT=0 MAPLE_MOE_MEGAKERNEL=0` steps down to the two
array-exact fusions alone — residual add + RMSNorm (the `mx.fast.rms_norm`
thread mapping is what makes it exact) and the fused QKV split. Together
+7.0% over portable MLX, stream-identical, and each runs a live comparison
at first use and switches itself off on mismatch.

### Opt-in

`MAPLE_COMPILED_ROUTER=1` compiles the stock router chain (array-exact,
cuts the router's host cost 96.5→77.9 µs per layer, end-to-end
indistinguishable from zero — the megakernel absorbs the router anyway).
`--cached-lhs` caches the flat decode LHS; exact in campaign but
process-global and keyed only by top-k, so it stays opt-in. Every lane is a
module attribute (`maple._use_moe_megakernel` and friends); the environment
only seeds them at import.

## Exactness protocol

The stock path is **not always reproducible run to run**: on some hosts six
identical runs produced two different token streams, always diverging at the
same position, with or without CUDA graphs. Forty-repeat probes of the
prefill forward, a single decode step, the router, the MoE block, attention
and RMSNorm found zero differences, so it is not a race inside any one
operation and the source is not localized.

Every equivalence verdict here therefore screens first: each prompt is
generated three times with the fusions off, and a candidate counts as
divergent only inside the region where those three runs agree. Disabling
cuDNN SDPA is required to make the stock oracle bit-stable and is not
credited as a speedup.

## Chronicle

Linear history of what landed, oldest first. Each entry is backed by
records under [`results/`](results/).

1. **2026-08-06 — Q/K norm + partial RoPE/NoPE fusion, `sm86`.** The first
   array-exact fused path and the fail-closed probing scheme; strict-exact
   kernel profile published for RTX 3090.
2. **2026-08-07 — the strict multi-arch matrix.** Five fresh NVIDIA targets
   (`sm86`-`sm120`), +6.1% to +16.3% paired over portable, exactness gated
   by `mx.array_equal` on live outputs, 144 W2 fingerprints and 20/20 fixed
   regressions ([`results/summary.csv`](results/summary.csv)).
3. **2026-08-07 — the Blackwell RoPE rounding fix.** On `sm100`/`sm120` the
   fused upper-half RoPE could contract the opposite product from stock; the
   profiles now pin `__fmaf_rn(v, cos, __fmul_rn(p, sin))`. Frozen boundary
   fixture: [`tests/data/sm100_qk_rope_boundary.npz`](tests/data/sm100_qk_rope_boundary.npz).
4. **2026-08-11 — the host-bound diagnosis.** GPU wait per decode step is
   ~3 µs on every host from a 3090 to a B200; wall clock is per-op host
   cost. Reframed the whole effort toward dispatch-count reduction
   ([`docs/host-bound-decode.md`](docs/host-bound-decode.md)).
5. **2026-08-11 — the exact fusions promoted.** Residual add+RMSNorm
   (+3.3%) and the fused QKV split (+2.2%) made default after matching the
   stock thread mappings bit for bit.
6. **2026-08-11 — the ~1 ULP MoE megakernel.** The whole MoE block in one
   dispatch behind grid barriers: +73-88% over portable on all five
   architectures; per-device grid retune (64/96/192 blocks); the 846-token
   quality suite shows no regression.
7. **2026-08-11 — the tail phase.** The next layer's add/RMSNorm folded in;
   the decode loop issues one fusion per step instead of one per layer
   (+2.2-3.5%).
8. **2026-08-11 — `qmm_naive` reproduced bit for bit.** Dequant
   `bf16(bf16(q·s)+z)` into the same `m16n8k16` atom, k-tiles in order —
   the fact that made an exact fast lane constructible. The rest of the
   block's bit semantics pinned the same day (router chain, renorm's
   shape-picks-the-kernel, aggregation's fma-must-not-contract).
9. **2026-08-11 — the exact MoE megakernel assembled and made default.**
   Stock stream on 8/8 screened prompts at megakernel speed; three rounds
   of bit-neutral scheduling closed the gap to the ~1 ULP lane (345.2 vs
   341.1 on `sm86`); validated on all five architectures.
10. **2026-08-11..12 — the attention bit map and megakernel.** The decode
    attention block's stock kernels pinned (1-pass SDPA port 12/12 at five
    context lengths, 2-bit `qmv` recipe, exact split+RoPE) and assembled
    into one dispatch; a decode layer is now **two dispatches**. +20.6% on
    `sm86`, +42.9% on `sm89` (455.7 tok/s, spread ±1.7). Default on, with
    live per-layer probes and a multi-turn flush guard.
11. **2026-08-12 — past the 1-pass limit.** The `kernel_sdpav_2pass` pair
    ported inside the same dispatch (72/72 standalone at kL 1025-8192);
    full-attention buffers grow 1024→8192 with one recompile per tier.
12. **2026-08-12 — the ring re-entry fixes.** The boundary suite caught two
    real bugs: re-entry after a multi-token prefill on a rotated ring
    mis-read the stock cache's temporal concat state (out-of-range slot,
    wrong window — this affected the shipped default with prompts longer
    than the sliding window), and python-side slice assignment into the
    persistent buffers copies-on-write, orphaning the kernel's appends. All
    seeding is now kernel-side through the input pointers; the boundary
    suite (A-E) and every legacy regression run bit-identical.

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

That is the default lane: two dispatches per decode layer, the stock token
stream bit for bit. The example passes
`model_config={"model_file": None, "use_flash_head": False}`, uses
`trust_remote_code=False`, verifies the loaded package source, uses the
exact LM head, and prints strict path state. Pin `mlx==0.32.0` and the
matching `mlx-cuda-12==0.32.0` wheel to reproduce the published claims. Add
`--cached-lhs` only for the constrained single-model/single-device warm
workload described above.

## Evidence and provenance

- [`src/maple.py`](src/maple.py), SHA-256
  `88c9e7965130ffbf833e770841092eb3a72f15cf62f6d16ca4f63437d057444e`;
- integration patch against DeepGrove `eba96c1`
  ([`patches/mlx-lm-deepgrove-maple-cuda.patch`](patches/mlx-lm-deepgrove-maple-cuda.patch)),
  SHA-256
  `26767c564d17452c16a18019c265b3a0a42c24c31e8187f8b7d14e3b519fe8e8`
  (touches `mlx_lm/models/maple.py`, `mlx_lm/models/switch_layers.py`,
  the kernel tests and the frozen `sm100` fixture);
- frozen fixture SHA-256
  `837638a799bef1b8ea7e7a23c77791964ca88f2bfc698f50910655c5f9bddb64`;
- [`results/PUBLIC-INDEX.json`](results/PUBLIC-INDEX.json), binding
  canonical analyses, manifests, source maps, and private raw-manifest
  commitments;
- detailed allowlisted artifacts under
  [`results/cuda/multiarch/`](results/cuda/multiarch/); compact strict,
  graph, W2, and Blackwell summaries in [`results/cuda/`](results/cuda/);
  the megakernel and attention evidence log is
  [`megakernel-grid-and-quality.jsonl`](results/cuda/megakernel-grid-and-quality.jsonl).

Historical per-architecture source hashes and the campaign-era equivalence
map are retained in
[`release-source-equivalence.json`](results/cuda/release-source-equivalence.json);
current claims are made against the hashes above.

## Known limitations

- Claims apply to the exact representative SKU, driver, MLX/CUDA version,
  model revision, and source provenance in the artifacts — not every GPU
  sharing a compute capability. Each throughput row is a single device
  instance measured once, not a fleet claim.
- The attention megakernel's throughput is profiled on `sm86`/`sm89`; on
  `sm90`/`sm100`/`sm120` it is bit-validated by its live probes but not yet
  perf-profiled. Contexts past 8192 keys fall back to the stock attention
  path per layer.
- Fresh Mac/Metal validation remains outstanding.
- Five `tests/test_generate.py` failures remain baseline-compatible on the
  tested checkout and are not claimed as fixed.
- Approximate router/add-RMS/ternary paths remain research-only.
- **MLX CUDA 0.32.0 needs a driver newer than 550.163.** An L40S offered
  with driver 550.163.01 could not start MLX at all
  (`cudaMallocManaged failed: unknown error`). Working hosts here ran
  575-610.
- **A CUDA 13 toolkit on the host breaks every custom kernel.**
  `mlx-cuda-12` compiles kernels with its bundled nvrtc 12.9 but takes
  headers from `$CUDA_HOME/include`, defaulting to `/usr/local/cuda`. On a
  host whose toolkit is 13.x that mixes nvrtc 12 with CUDA 13 headers and
  every `mx.fast.cuda_kernel` fails to compile inside `cuda_fp8.hpp`, which
  surfaces as *every* fast path silently falling back. Install the matching
  headers and point `CUDA_HOME` at them:

  ```bash
  pip install 'nvidia-cuda-runtime-cu12==12.9.*'
  mkdir -p ~/cuda12 && ln -s "$(python -c 'import nvidia.cuda_runtime, pathlib; print(pathlib.Path(nvidia.cuda_runtime.__file__).parent / "include")')" ~/cuda12/include
  export CUDA_HOME=~/cuda12
  ```
- The megakernels' grid barriers assume every block is resident, and MLX
  exposes neither the multiprocessor count nor occupancy, so grids are
  inferred from compute capability and memory. The rule is deliberately
  conservative and `MAPLE_MOE_MEGAKERNEL_GRID` overrides it, clamped at 240
  so a typo becomes a slowdown rather than a deadlock.
- **The megakernels want the GPU to themselves.** The barriers spin, so a
  GPU shared with another CUDA process pays for the spin. On a 3090 running
  two other workloads the fast lane swung between 183 and 340 tok/s across
  four fresh processes while `off` swung between 123 and 211 — usable, but
  not something to benchmark on.
- Community-cloud hosts share the CPU even when the GPU is dedicated, and
  this workload is host-bound. One host with an idle GPU and a load average
  of 4-9 returned between 104.7 and 325.8 tok/s for the same configuration.
  Check `/proc/loadavg` before trusting a measurement and report medians
  over many fresh processes.
- The accepted RTX 5090 W2 tile (`16x32x128`, +1.6%) requires the
  separately built experimental MLX backend and is not enabled by this
  patch alone.

## License

Project code is MIT. Original Apple and DeepGrove notices are preserved; see
[`NOTICE.md`](NOTICE.md). The optional regression manifest obtains
third-party question content under its source licenses; see
[`DATASET-NOTICE.md`](DATASET-NOTICE.md).
