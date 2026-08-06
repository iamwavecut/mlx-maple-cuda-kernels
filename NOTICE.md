# Attribution and scope

The Maple model implementation is derived from
[`deepgrove-ai/mlx-lm-deepgrove`](https://github.com/deepgrove-ai/mlx-lm-deepgrove)
at commit `eba96c16158f032821b0bf374ea1421cfddef0a9`, which is based on Apple's
[`mlx-lm`](https://github.com/ml-explore/mlx-lm). Inherited MIT copyright and
license notices are preserved in source files and the generated patch.

The strict-exact research implementation was frozen in laboratory commit
`b3d03fb19b522f307d0df7ba2ea347711a2ee337`. The CUDA kernels, fallback gates,
benchmark harnesses, and published measurements were produced by Valeriy
Selitskiy (`iamwavecut`) in August 2026.

This is an independent experimental project. It is not an Apple, MLX,
DeepGrove AI, NVIDIA, or dataset-owner release.

The root MIT license covers this project's code and documentation. Generated
third-party regression questions retain their source licenses and attribution;
see [`DATASET-NOTICE.md`](DATASET-NOTICE.md). Nothing in the root license
relicenses that dataset content.
