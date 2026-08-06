# Published results

## Current strict evidence

`summary.csv` contains the two current `sm86` rows:

- `strict_qk_default`: conservative source default, measured in the balanced
  component factorial;
- `strict_qk_cached_lhs_opt_in`: exact speed profile used for the direct paired
  result and long common-slice gates.

Throughput columns are arithmetic means. `paired_geomean_gain_percent` and its
confidence interval are computed from within-block/pair log ratios, not from
the ratio of displayed means.

Current sanitized records:

- `cuda/sm86-strict-profile.jsonl`: six direct timing pairs and 1024-token gate;
- `cuda/sm86-component-factorial.jsonl`: eight balanced R/Q/L/QL blocks;
- `cuda/sm86-common-slice.jsonl`: 512/1024 fixed-slice artifact hashes and
  grading, without generated answer text;
- `cuda/sm86-graph-tuning.jsonl`: ops/MB/cache attribution with cache and MB
  provenance per block/pair;
- `cuda/sm86-ternary-validation-summary.jsonl`: semantic prototype evidence and
  strict rejection decision.

The benchmarked model/switch sources, executed harness variants, and complete
default-only diffs to the published source are retained under `../provenance/`.
The checkpoint revision is an asserted pinned setup input; the original run did
not record hashes for every weight/config/tokenizer artifact. Published JSONL
is sanitized: no GPU UUID, PCI bus ID, local/model path, raw service log, or
generated answer text.

`exact_gate` is deliberately narrow. Q/K live arrays were exact, and the listed
token/text/selected-logprob/top-1 artifacts matched in the stated runs. It does
not mean every full-logit tensor was compared.

## Legacy evidence

`legacy-initial-port/` preserves the original multi-architecture and M2 data.
Those records are historical/context-only and are not current strict claims.
The initial CUDA campaign used a short oracle and tolerant probes that admitted
router/add-RMS paths now classified as semantic. Mac/Metal and `sm89`-`sm120`
must be revalidated with the current source and deterministic contract.

`SHA256SUMS` covers every published result data file except itself.
