# Benchmark harnesses

## Strict/current

The current copies implement the sealed multi-architecture campaign harnesses:

- `maple_model_benchmark.py`: direct portable/strict 1024-token gate and timing;
- `maple_fresh_process_block.py`: one position-balanced fresh R/Q/QL block;
- `maple_equivalence_matrix.py`: three-case multi-seed exact-output gate;
- `maple_common_slice_benchmark.py`: fixed-slice token/text/logprob/top-1 gate;
- `maple_component_factorial_benchmark.py`: R/Q/L/QL attribution;
- `maple_auto_benchmark.py`: warm single-mode graph screen;
- `maple_qmm_fingerprint.py`: 144-projection W2 array fingerprint/reference;
- `maple_qmm_microbenchmark.py`: W2 per-layer screening only;
- `maple_qmm_tile_factorial.py`: fresh candidate/default tile comparison with
  graphs disabled.

Exact executed harness hashes remain in each sanitized bundle's
`harness.sha256`; the public component-factorial copy has formatting-only
changes from its sealed executed bytes.

`prepare_maple_common_slice.py` fetches and hash-checks the separately licensed
input. Dataset content is not committed here. All current strict loaders force
`model_file=None`, disable FlashHead and remote code, and record the actual
package source.

The QMM tile tools require the separately patched MLX tuning backend described
in `docs/benchmark-methodology.md`; environment tile variables do nothing on a
stock wheel and are not part of the normal QuickStart.

## Historical/experimental

`maple_mode_benchmark.py` retains isolated LHS experiments. Ablation, router,
elementwise, stack, and ternary tools can enable semantic or tolerant screening
lanes and must not be used to promote a strict claim.

A short token hash or tolerant microbenchmark is never sufficient. Follow the
full deterministic, array-exact, common-slice, fresh-process, and provenance
gates in `docs/benchmark-methodology.md`.
