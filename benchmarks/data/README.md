# Fixed-slice input

Question text is not redistributed in this repository. Generate the exact
20-case manifest from the pinned upstream DwarfStar source:

```bash
python benchmarks/prepare_maple_common_slice.py
```

The generator downloads commit
`b0309611041655f4e45671cfd9c9886aff161406`, verifies `ds4_eval.c` SHA-256
`19545bf6…`, extracts cases 1-20 without editing their content, and requires the
resulting manifest SHA-256 to be `d581a0a8…`.

The downloaded/generated question content remains under its applicable source
licenses. See [`../../DATASET-NOTICE.md`](../../DATASET-NOTICE.md).
