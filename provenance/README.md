# Source provenance

## Fresh multi-architecture campaign

The release source is [`../src/maple.py`](../src/maple.py), SHA-256
`28ceabac2b7570ff3712473c88eb7698b5a1904cd1b9cd55c698794fd457ccb8`.
The integration patch against DeepGrove base
`eba96c16158f032821b0bf374ea1421cfddef0a9` is SHA-256
`eb9c36eb5aec3c93e52ddcc35d735f816a18ab5330460a05b7a641ba0f5174f0`.

Fresh baseline execution used:

| Profile | Full-file source SHA-256 | Retained source |
| --- | --- | --- |
| `sm86`, `sm120` | `28ceabac2b7570ff3712473c88eb7698b5a1904cd1b9cd55c698794fd457ccb8` | release source |
| `sm89`, `sm90` | `7785da2a85b97b9fd7759d8756b1daf2231ec8b912d42b4b7bc9c04637b371ae` | `maple-multiarch-validated-7785da2a.py` |
| `sm100` | `b34cd9777cf5a8775ed4e814fe7e14c987a9021627224168595c92ddf21edae4` | `maple-sm100-validated-b34cd977.py` |

The intervening changes are profile-conditional Blackwell rounding pins.
`results/cuda/release-source-equivalence.json` records SHA-256 values for the
exact RoPE/NoPE strings passed to `mx.fast.cuda_kernel`: release-generated
source matches each executed profile. This demonstrates Q/K code-generation
isolation only; it does not assert whole-module rerun equivalence.

The fresh current-source RTX 3090 extension is bound by
`sm86-fresh-release.json`: campaign manifest `b191ab1e…`, canonical raw manifest
`7452e59f…`, canonical analysis `d47337fa…`, and sanitized bundle manifest
`25d4872b…`. It used the release source directly and passed pre/post health and
the full finite strict baseline. Graph and custom-W2 screens were not repeated.

The frozen checked-in Blackwell fixture is SHA-256
`837638a799bef1b8ea7e7a23c77791964ca88f2bfc698f50910655c5f9bddb64`.
The private full diagnostic input artifact `ecb4ff1…` is a different file; the
checked-in fixture is its compressed input-only subset/repack.

## Earlier historical `sm86` campaign

The main historical `sm86` performance and long-decode campaign executed
`maple-benchmarked-6c9fc558.py` (SHA-256
`6c9fc558eeac8faa69eaa53d01e0c30828d7d976722c22924a9b646d848718b4`).
`benchmark-to-published.patch` is the complete diff to its then-published
source; it changed cached-LHS and uint32-router-index defaults from on to off.
The benchmark lanes set both explicitly.

The benchmark-time `switch_layers.py` is retained as
`switch_layers-benchmarked-af207c5.py` (SHA-256
`af207c5cfad07594a3ce0d2a92cebe016f18273e604876b30cc5561341831887`).
The current `src/switch_layers.py` is `3b51288a…`; its historical diff only
removed an unused import.

Historical executed harnesses retained here include common-slice
`4ed255c9…` and component factorial `7dbfeae5…`; see the manifest for full
hashes. Current campaign harness hashes are embedded in every sanitized per-SKU
bundle rather than being inferred from these historical snapshots.

`SHA256SUMS` covers every provenance file except itself.
