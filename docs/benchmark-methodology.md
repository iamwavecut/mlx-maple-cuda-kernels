# Benchmark methodology

## End-to-end CUDA comparison

- Checkpoint: `deepgrove/maple-preview-2bit-mlx`, revision
  `361db5da5e74ff6fcdd852d478e1f266ce11013a`.
- MLX and MLX CUDA: 0.32.0.
- Prompt: 128 deterministic pseudo-random token IDs, seed `20260806`.
- Decode lengths: 256 tokens on all GPUs; 1024 tokens additionally on GPU2.
- EOS termination disabled, so every trial emits the requested token count.
- Reference and accelerated paths run in the same loaded process.
- The order is alternated per trial.
- Reported table values are arithmetic means; raw trial distributions are kept
  in JSONL.
- Quality gate: exact equality of 256 greedy token IDs and their SHA-256.

These are end-to-end decode measurements, not isolated kernel timings. They
include the rest of the model and MLX dispatch overhead.

## M2 Max baseline

The M2 Max run uses the checkpoint's original Metal implementation with exact
LM head (`use_flash_head=False`). It uses the same seed, 128-token prompt,
decode lengths, warmup, and five-trial summary as the CUDA harness.

The laptop remained in normal interactive use. Several browser renderers were
active, so the record is deliberately tagged `interactive_non_quiescent` and
should not be treated as a peak hardware result. The close min/max range makes
it useful as a current local baseline, not as a universal M2 Max claim.

## What is excluded

- FlashHead is approximate and excluded from exact speedup tables.
- Failed or preempted cloud attempts, pod metadata, console logs, local paths,
  GPU UUIDs, and raw profiler captures are not published.
- Cross-host comparisons are descriptive. Different CPUs, power limits,
  runtimes, and cloud tenancy make GPU-to-GPU ranking less controlled than the
  paired reference/accelerated comparison within a host.
