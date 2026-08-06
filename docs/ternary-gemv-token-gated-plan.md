# Maple Ternary GEMV: token-gated / semantic follow-up

Status: **experimental only; disabled by default; excluded from the strict-exact lane**.

## Evidence captured on sm86

The Maple-specific row-alpha ternary up/gate prototype reached about **1.47x**
relative to stock `mx.gather_qmm`.  The full 24-layer, six-input validation found:

- up/gate: 248 differing BF16 values out of 1,179,648 (about 0.021%);
- 88/144 comparisons were not array-identical;
- maximum observed absolute difference: 0.25 on deliberately scaled inputs;
- checkpoint schema checks passed for every layer (row alpha, bias = -alpha,
  and no packed code 3).

The likely cause is a different FP32 accumulation/reduction order.  Small local
errors can compound across 24 MoE layers or flip rare near-tie decisions, so the
prototype does not qualify as a strict-exact optimization.

A sanitized aggregate is published in
[`results/cuda/sm86-ternary-validation-summary.jsonl`](../results/cuda/sm86-ternary-validation-summary.jsonl).
Raw per-comparison laboratory logs are intentionally not published.

## Required policy

- Stock GatherQMM remains the default.
- `_use_cuda_ternary_up_gate` remains `False` unless an explicitly experimental
  token-gated/semantic run enables it.
- Unsupported schema, architecture, shape, dtype, or a failed live probe must
  fail closed to stock GatherQMM.
- Ternary results must never be added to the advertised strict-exact speedup.

## Follow-up validation matrix

After graph sizing, router dependency/order, and decode-cleanup validation is
complete, test the specialization separately with:

1. Greedy runs of 256, 1024, 4096 and longer tokens.
2. Multiple fixed prompts/seeds and prompt lengths, including long-context and
   rotating-cache boundaries.
3. Router/logit near-tie cases and adversarial scaled activations.
4. sm86, sm89, sm90, sm100, and sm120 independently tuned profiles.
5. For every paired run: full token hashes, per-step logit max/mean error,
   top-k margins, first differing token/logit position, and nondeterminism
   repeats.
6. Semantic/cognitive evaluation of continuations (instruction adherence,
   factual consistency, reasoning and long-range coherence), reported apart
   from exact-token results.
7. Compute-sanitizer and long graph-update/concurrency stress before any opt-in
   release.

Promotion is allowed only to an opt-in token-gated/semantic branch.  Strict
exact remains stock GatherQMM unless a future kernel reproduces MLX's
accumulation order and passes full BF16 array equality.
