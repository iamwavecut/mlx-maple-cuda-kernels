# Benchmark harnesses

## Strict/current

- `maple_model_benchmark.py`: portable vs exact Q/K + cached-LHS profile;
- `maple_common_slice_benchmark.py`: fixed-slice correctness artifacts;
- `maple_component_factorial_benchmark.py`: R/Q/L/QL attribution;
- `maple_auto_benchmark.py`: warm single-mode strict profile;
- `maple_mode_benchmark.py`: isolated LHS/index experiments;
- `prepare_maple_common_slice.py`: pinned, hash-checked input generator.

## Experimental/semantic

- `maple_ablation_benchmark.py` and `maple_stack_candidate_benchmark.py` may
  explicitly enable approximate router/add-RMS paths;
- router sweep/dependency tools exercise the experimental router;
- ternary prototype/validation tools are token-gated research only;
- kernel/elementwise sweeps use tolerant candidate screening and do not weaken
  strict live probes in `src/maple.py`.

Do not promote a result into the strict table merely because a short token hash
or tolerant microbenchmark check passes. Follow `docs/benchmark-methodology.md`
and run deterministic long-decode gates in fresh processes.
