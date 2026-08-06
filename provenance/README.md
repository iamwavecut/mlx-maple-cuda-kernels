# Source provenance

The main `sm86` performance and long-decode correctness campaign executed
`maple-benchmarked-6c9fc558.py`:

- SHA-256: `6c9fc558eeac8faa69eaa53d01e0c30828d7d976722c22924a9b646d848718b4`
- DeepGrove base: `eba96c16158f032821b0bf374ea1421cfddef0a9`

The published conservative source is [`../src/maple.py`](../src/maple.py):

- SHA-256: `7785da2a85b97b9fd7759d8756b1daf2231ec8b912d42b4b7bc9c04637b371ae`
- laboratory commit: `b3d03fb19b522f307d0df7ba2ea347711a2ee337`

[`benchmark-to-published.patch`](benchmark-to-published.patch) is the complete
diff. It only changes cached-LHS and uint32-router-index defaults from on to
off and adds the rationale comment. Benchmark lanes set both flags explicitly,
so their behavior is unchanged. The final published source separately passed
the CUDA focused gate (20 passed, 2 skipped).

## Additional executed sources

The benchmark-time `switch_layers.py` is retained as
`switch_layers-benchmarked-af207c5.py` (SHA-256
`af207c5cfad07594a3ce0d2a92cebe016f18273e604876b30cc5561341831887`).
The published `src/switch_layers.py` is `3b51288a…`; its complete diff only
removes an unused `functools.partial` import.

Executed harness SHA-256 values:

- direct profile, `benchmarks/maple_model_benchmark.py`: `b525c282058688300578154a944af069c757a21cf340c11fdb5644778975dd4e`;
- graph/cache, `benchmarks/maple_auto_benchmark.py`: `791c89c583b79f117e051b38bd8589b997bc7d6fc5626a4411a36f47dfc97822`;
- LHS mode, `benchmarks/maple_mode_benchmark.py`: `1f1eb31235cbaa86300f9371c5fe95eb7ebbb97b76fd8b8fc861d70bb7a00dfe`;
- common slice: `4ed255c904a603d10f791058759172cd8a4d61cacb224311d7ac0969a64ebdbf`;
- component factorial: `7dbfeae5392a9a854db881ba6757dbd96b98c58b6c83f297c0186b5f8f21d923`.

The last two exact executed files are retained here because the public benchmark
copies received import-only formatting after execution. The model/auto/mode
public copies remain byte-identical to the executed versions.
