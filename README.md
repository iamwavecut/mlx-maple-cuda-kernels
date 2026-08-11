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

## Decode on CUDA is host-bound

The measurement that reframed this work: with a warm cache and MLX's
double-buffered `async_eval`, the **GPU wait per decode step is ~0.003 ms**.
The GPU finishes before Python has finished building the next step, on every
host tested. Wall clock is the sum of per-operation host costs, so the currency
of optimization is operation count, not arithmetic.

| Host | graph build | submission | GPU wait | step |
| --- | ---: | ---: | ---: | ---: |
| RTX 3090, EPYC 7452 | 2.40 ms | 3.96 ms | **0.0025 ms** | 6.36 ms |
| RTX 3090, AI Farm | 2.02 ms | 2.76 ms | **0.0028 ms** | 4.78 ms |
| RTX 4090 | 2.43 ms | 3.31 ms | **0.0029 ms** | 5.76 ms |
| H100 80GB HBM3 | — | — | **0.0040 ms** | — |
| B200 | — | — | **0.0030 ms** | — |
| RTX 5090 | — | — | **0.0022 ms** | — |

The GPU wait stays at the same few microseconds from a 3090 to a B200, which
is why the fusion gains do not track GPU class.

Two consequences shaped the release. Kernels that make the GPU faster without
removing operations do nothing: a hand-written 2-bit expert GEMV measured
1.88x-1.98x against stock in isolation and moved end-to-end throughput by
roughly nothing. And it is worth spending GPU time to buy back host operations,
which is what the megakernel does.

Related measurements, all on RTX 3090:

- Streaming-read bandwidth reaches 743 GB/s; the stock expert projections run
  at 203 GB/s, and the exact 4-bit lm_head at 589 GB/s.
- A forward pass costs 5.08 ms at one token and 26.35 ms at 32, so a marginal
  token costs 0.686 ms — a 7.4x leverage that makes speculative decoding
  break even at ~2.4 accepted tokens per pass.
- Aggregate throughput peaks near batch 4 (523 tok/s against 256 at batch 1);
  past that the MoE weight traffic dominates.
- `stream_generate`'s detokenizer and response objects cost 0.23 ms per token,
  7.6% at this speed.

Full write-up: [`docs/host-bound-decode.md`](docs/host-bound-decode.md).

## Fusion release

Fresh processes per mode, interleaved order, warm `B=1`, `L=1` BF16 decode,
128-token prompt / 512-token generation. `off` disables every new fusion and
reproduces the previous release path. Ratios are paired geometric means over
the processes of one device instance.

| GPU | CC | off | strict (default) | Paired gain (95% CI) | Wins |
| --- | --- | ---: | ---: | ---: | ---: |
| RTX 3090 | `sm86` | 152.29 | **164.94** | **+6.68%** (+4.33%–+9.09%) | 8/8 |
| RTX 4090 | `sm89` | 157.66 | **184.25** | **+16.91%** (+15.27%–+18.56%) | 6/6 |
| H100 80GB HBM3 | `sm90` | 206.75 | **226.88** | **+9.32%** (+2.70%–+16.36%) | 5/6 |
| B200 | `sm100` | 141.87 | **165.19** | +11.77% (−4.35%–+30.61%) | 5/6 |
| RTX 5090 | `sm120` | 242.61 | **269.34** | **+10.63%** (+8.84%–+12.46%) | 6/6 |

The opt-in megakernel on the same runs:

| GPU | CC | megakernel | Paired gain (95% CI) | Wins | Token stream |
| --- | --- | ---: | ---: | ---: | --- |
| RTX 3090 | `sm86` | **272.80** | **+79.51%** (+76.18%–+82.91%) | 8/8 | 1/8 prompts |
| RTX 4090 | `sm89` | **292.15** | **+87.04%** (+81.01%–+93.26%) | 6/6 | 0/8 |
| H100 80GB HBM3 | `sm90` | **361.72** | **+76.11%** (+67.86%–+84.77%) | 6/6 | 1/8 |
| B200 | `sm100` | **254.37** | **+73.86%** (+47.35%–+105.14%) | 6/6 | 3/8 |
| RTX 5090 | `sm120` | **426.66** | **+75.35%** (+71.10%–+79.70%) | 6/6 | 0/8 |

**The strict lane reproduced the stock token stream on 8/8 screened prompts on
every architecture.** The megakernel does not, as designed — see the quality
section below for what that costs.

The `sm100` strict interval crosses zero and its megakernel interval is wide:
that host had the lowest absolute throughput of the five despite the largest
GPU, which is what a host-bound workload on a contended CPU looks like. The
B200 numbers are a single-instance observation and should be read as such.

At 2048 generated tokens the ordering holds on every target and peak memory
does not grow; on `sm86` that is 154.75 / 163.51 / 274.72 tok/s at 6.478 GB.

Absolute rates are not a cross-GPU ranking: each row is one device instance on
one host, and for this workload the host CPU moves the number more than the
GPU does.

That megakernel column was measured at the fixed grid of 32 blocks this release
originally shipped, so it understates the lane on everything but a 3090; the
next section is the retune, and the shipped default is faster than the table.

### Megakernel grid

The grid barrier is correct only while every block is resident, and MLX exposes
neither the multiprocessor count nor occupancy, so the block count cannot be
read off the device. It is selected from compute capability and memory instead.
Medians for the fast lane, four fresh processes per point:

| GPU | 32 | 64 | 96 | 128 | 192 | shipped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX 3090 | 357.5 | **375.0** | 365.3 | 370.2 | 333.9 | 64 |
| RTX 4090 | 429.1 | 469.1 | **507.7** | 478.5 | 463.3 | 96 |
| H100 80GB | 344.8 | 359.0 | **365.9** | 358.4 | 356.2 | 96 |
| RTX 5090 | 422.9 | 424.9 | **435.3** | 427.2 | 426.7 | 96 |
| B200 | 293.4 | 327.9 | 324.9 | 332.1 | **335.4** | 192 |

Leaving this at a fixed 32 costs 2.9% on RTX 5090 and 18.3% on RTX 4090.
`MAPLE_MOE_MEGAKERNEL_GRID` overrides the choice; the value is clamped so a
typo becomes a slowdown rather than a deadlock.

Compare rows, not columns, with the release table above: that one is a paired
measurement inside single processes, this one is a separate sweep on separate
instances, and the absolute tok/s of a host-bound workload tracks the CPU it
was rented with. A confirmation run of the shipped rule on a fresh RTX 4090
selected 96 without being told to and measured 260.4 off, 280.8 strict
(+7.8%), **465.0 megakernel (+78.6%)**, with strict identical on 8/8 screened
prompts.

### Quality

Token equality answers whether the greedy path changed. It does not answer
whether the model is worse, which is the question that matters for a lane that
is within ~1 ULP rather than array-exact. `benchmarks/maple_quality_suite.py`
scores twelve documents — prose, dialogue, code, SQL, structured text and
arithmetic — one token at a time against a cache, so the fused decode kernels
are actually exercised.

| GPU | reference ppl | strict | megakernel | top-1 changed |
| --- | ---: | --- | ---: | ---: |
| RTX 3090 | 33.1857 | identical | 32.8173 (−1.1%) | 78 / 846 |
| RTX 4090 | 33.1857 | identical | 32.8173 (−1.1%) | 78 / 846 |
| H100 80GB | 32.9462 | identical | 32.6704 (−0.8%) | 80 / 846 |
| RTX 5090 | 33.1647 | identical | 32.7340 (−1.3%) | 75 / 846 |
| B200 | 33.1647 | identical | 32.7340 (−1.3%) | 75 / 846 |

The strict lane matches the reference to the last digit of the mean NLL on
every architecture, with zero top-1 changes — bit-exactness confirmed at the
level of the distribution, not just the sampled token.

The megakernel's perplexity is **lower** than the reference on all five, by
0.8-1.3%. That is the signature of unbiased last-bit noise, not of a better or
worse model: the same perturbation that flips 9% of top-1 predictions on
near-ties moves the likelihood slightly, and it happened to move down here. It
is not a quality improvement and should not be read as one. What it does rule
out is a quality *regression*.

### What is on by default

- `_use_fused_add_rms` — residual add + RMSNorm in one dispatch. The shipped
  kernel before this release used the elementwise thread count and two chunks
  per thread; `mx.fast.rms_norm` uses 512 threads with four contiguous
  elements each. Reproducing that partition is what makes the fusion
  array-exact, and it is why the path could be promoted out of the
  experimental lane.
- `_use_fused_qkv` — the Q/K norm + RoPE kernel widened to consume the fused
  qkv projection and emit queries, keys and values in their final shapes,
  removing the slice-and-reshape chain around it. Bit-identical by
  construction: the per-head arithmetic is unchanged.

Component medians on the same host: add+RMSNorm +3.3%, QKV split +2.2%,
together +7.0%.

### What is opt-in

- `MAPLE_MOE_MEGAKERNEL=1` — router, experts, activation, score-weighted
  aggregation and the preceding add/RMSNorm in **one dispatch**, with three
  atomic-counter grid barriers. Worth 73-88%. It is within ~1 ULP of bf16
  rather than array-exact: `qmm_naive` gets its accuracy from a tensor-core
  MMA, which a software fp32 reduction cannot reproduce.

  It is off by default because this repository's contract is a reproducible
  token stream, **not** because it is known to cost quality — the suite above
  finds no regression on any supported architecture. If you are serving rather
  than reproducing, this is the switch you want.
- `MAPLE_COMPILED_ROUTER=1` — the stock router chain under `mx.compile`.
  Array-exact, and in isolation it cuts the router's host cost from 96.5 us to
  77.9 us per layer, but paired over ten fresh processes the end-to-end effect
  was 1.0062x with a 95% interval of 0.9927-1.0198 and 6/10 wins. Exact but
  not distinguishable from zero, so it ships off.

Every lane is also a module attribute (`maple._use_moe_megakernel` and
friends), which is what the tests flip; the environment only seeds them at
import, so a server does not have to reach into the module before the model
loads. `MAPLE_FUSED_ADD_RMS=0` and `MAPLE_FUSED_QKV=0` turn the defaults back
off, which is how `off` is measured in every table here.

### Exactness protocol

The stock path is **not always reproducible run to run**: on some hosts six
identical runs produced two different token streams, always diverging at the
same position, with or without CUDA graphs. Forty-repeat probes of the prefill
forward, a single decode step, the router, the MoE block, attention and
RMSNorm found zero differences, so it is not a race inside any one operation
and the source is not localized.

Every equivalence verdict here therefore screens first: each prompt is
generated three times with the fusions off, and a candidate counts as
divergent only inside the region where those three runs agree. Under that
protocol the reference was stable on all eight prompts, `strict` matched on
8/8 and the megakernel on 1/8.

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
| Fused QKV split (`_use_fused_qkv`) | **on** | array-exact by construction; probed live |
| Residual add + RMSNorm (`_use_fused_add_rms`) | **on** | array-exact once the thread mapping matches `mx.fast.rms_norm` |
| MoE megakernel (`_use_moe_megakernel`) | off | within ~1 ULP of bf16; not array-exact |
| Compiled router (`_use_compiled_router`) | off | array-exact; end-to-end effect not distinguishable from zero |
| Cached flat decode LHS | off | exact in campaign; lifecycle-limited opt-in |
| Router GEMV/softmax/top-8 | off | normalized scores not array-exact |
| Residual add + RMSNorm, original kernel | off | thread mapping differs from `mx.fast.rms_norm` |
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

That is the strict default: a token stream identical to stock, 7-17% faster.
For the fastest lane, add `MAPLE_MOE_MEGAKERNEL=1` to the environment above —
73-88% instead, at the cost of a reproducible token stream and nothing else
that has been measurable. See [Quality](#quality).

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
- The fusion release, the megakernel grid sweep and the quality suite are in
  [`sm86-fusion-release.jsonl`](results/cuda/sm86-fusion-release.jsonl),
  [`fusion-multiarch.jsonl`](results/cuda/fusion-multiarch.jsonl) and
  [`megakernel-grid-and-quality.jsonl`](results/cuda/megakernel-grid-and-quality.jsonl).

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
- The fusion release is validated on all five targets, but each row is a single
  device instance measured once, not a fleet claim.
- **MLX CUDA 0.32.0 needs a driver newer than 550.163.** An L40S offered with
  driver 550.163.01 could not start MLX at all
  (`cudaMallocManaged failed: unknown error`). Working hosts here ran 575-580.
- The megakernel's grid barrier assumes every block is resident. MLX does not
  expose the multiprocessor count, so the grid is fixed at 32 blocks, which
  every CUDA GPU of this era holds at once. Throughput is flat from 32 to 160
  blocks on RTX 3090 and only falls off past 192, so the conservative choice
  costs nothing; raising it on a smaller device would risk a deadlock.
- Community-cloud hosts share the CPU even when the GPU is dedicated, and this
  workload is host-bound. One host with an idle GPU and a load average of 4-9
  returned between 104.7 and 325.8 tok/s for the same configuration. Check
  `/proc/loadavg` before trusting a measurement and report medians over many
  fresh processes.
- The accepted RTX 5090 W2 tile requires the separately built experimental MLX
  backend and is not enabled by this patch alone.

## License

Project code is MIT. Original Apple and DeepGrove notices are preserved; see
[`NOTICE.md`](NOTICE.md). The optional regression manifest obtains third-party
question content under its source licenses; see [`DATASET-NOTICE.md`](DATASET-NOTICE.md).
