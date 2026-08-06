# Benchmark methodology

## Frozen inputs

- Checkpoint: `deepgrove/maple-preview-2bit-mlx`, revision
  `361db5da5e74ff6fcdd852d478e1f266ce11013a`.
- DeepGrove base: `eba96c16158f032821b0bf374ea1421cfddef0a9`.
- Laboratory implementation: `b3d03fb19b522f307d0df7ba2ea347711a2ee337`.
- Published `src/maple.py` SHA-256:
  `7785da2a85b97b9fd7759d8756b1daf2231ec8b912d42b4b7bc9c04637b371ae`.
- MLX and MLX CUDA: 0.32.0; Python 3.12.3.
- Current strict hardware evidence: RTX 3090, compute capability `sm86`.

The setup pinned the checkpoint revision above, but the original run records did
not hash every weight/config/tokenizer artifact. Published JSONL marks this as
an asserted revision rather than a cryptographic binding. Driver, CUDA runtime,
cuDNN version, power/clock state, and host co-tenant scheduling were also not
frozen, so absolute-rate reproduction should expect system variance.

Benchmarks load with
`model_config={"model_file": None, "use_flash_head": False}` and record the
actual module/file/source hash. The common-slice harness additionally asserts
that the loaded class file is the imported worktree module. Expected hashes are
verified from the retained artifacts rather than by every CLI.

The main performance/correctness campaign used source `6c9fc558…`. The final
published source differs only by defaulting cached LHS and uint32 experimental
router indices off; every benchmark lane explicitly set both values. The final
CUDA focused suite was rerun on `7785da2…` and passed 20 tests with 2 skips.

## Deterministic process environment

Set before the first CUDA operation:

```sh
MLX_CUDA_USE_CUDNN_SDPA=0
MLX_USE_CUDA_GRAPHS=1
MLX_CUDA_GRAPH_CACHE_SIZE=400
MLX_MAX_OPS_PER_BUFFER=100
MLX_MAX_MB_PER_BUFFER=100
```

Repeated portable long generations could diverge while cuDNN SDPA was
eligible, even with graphs disabled. The cuDNN setting is therefore a
correctness-oracle requirement. It is not credited as a Maple speedup.

## Correctness pass

Correctness is run separately from timing. The gates include:

1. shape, dtype, and `mx.array_equal` live probes for each strict candidate;
2. 1024-token random-prompt reference/strict equality;
3. the audited fixed 20-case slice at up to 512 and 1024 generated tokens;
4. token, decoded-text, selected-token-logprob, and top-1 hashes;
5. active/fallback state recording for every layer and path.

Selected-token logprob and top-1 hashes do not establish full-logit equality.
They are deliberately not computed inside throughput trials. The 512-cap slice
used cache 2000 / 100 ops / 1000 MB; the follow-up 1024-cap slice used the
recommended cache 400 / 100 ops / 100 MB. JSONL records this per slice.

The 20-case manifest is pinned to antirez/ds4 commit
`b0309611041655f4e45671cfd9c9886aff161406`, upstream file SHA-256
`19545bf6…`; the local manifest SHA-256 is `d581a0a8…`. It is a regression
slice, not a statistically representative quality benchmark. Official grading
requires a final `Answer:` line; loose extraction is diagnostic only.

## Timing pass

### Direct strict profile

The measured path is warm, single-stream `B=1`, `L=1` BF16 decode. Prefill,
batched decode, scaled-RoPE policies, JIT/live-probe cost, and cold-cache setup
are not accelerated-rate claims.

- 128 deterministic pseudo-random prompt token IDs, seed `20260806`.
- 512 generated tokens, EOS disabled.
- Six alternating paired reference/strict trials.
- Strict profile: exact Q/K + cached flat decode LHS; portable router,
  add/RMS, W2, and exact LM head.
- Graph profile: 100 ops, 100 MB, cache 400.
- Reported throughput columns are arithmetic means.
- The primary effect is the geometric mean of within-pair ratios with a
  two-sided 95% t-interval on log ratios.

These small-n intervals and p-values are exploratory. They follow substantial
kernel/profile tuning, use no multiple-testing correction, and were collected
with a host co-tenant active; they are paired evidence for this run, not a
population-level hardware guarantee.

### Component factorial

Eight position-balanced blocks run `R`, `Q`, `L`, and `QL` in one loaded model:
portable reference, exact Q/K, cached LHS, and both. Main effects and the
multiplicative interaction are computed within block on log throughput. This
separates Q/K's supported contribution from cached LHS's noisy marginal effect.
Strict timing follows equivalence and warmup, so it excludes JIT/live-probe cost
and cold cached-LHS construction; it is a warm steady-state decode result.

### CUDA graph factorial

Four A-D blocks cross ops `{20,100}` and MB `{100,1000}` at graph cache 2000.
Four additional B/D blocks test 100 MB vs 1000 MB at cache 400. A separate
four-pair comparison tests cache 400 vs 2000 at 100 ops / 1000 MB. Result
records encode the cache
used by each block rather than implying one global setting.

## Published and excluded data

Published JSONL removes GPU UUIDs, PCI bus IDs, local paths, model paths, raw
consoles, generated answer text, and profiler databases. Case indices, aggregate grades, prompt hashes, token/text hashes, per-trial
throughput, and paired statistics are retained for auditability.

Initial multi-architecture results are preserved under
`results/legacy-initial-port/` but are superseded as strict claims. Their
short oracle and tolerant probes admitted paths now classified as semantic.
Fresh `sm89`, `sm90`, `sm100`, and `sm120` validation is required.

FlashHead, KV quantization, approximate router/add-RMS, ternary projection,
failed/preempted cloud attempts, and cross-host performance rankings are
excluded from the current strict table.
