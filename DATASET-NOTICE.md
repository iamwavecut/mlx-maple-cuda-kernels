# Dataset notice

This repository does **not** redistribute the fixed-slice question text.
`benchmarks/prepare_maple_common_slice.py` is MIT-licensed extraction tooling
that users may run to obtain the input from the pinned upstream source. The
generated `benchmarks/data/maple_common_slice_20.json` is ignored by Git and
retains the licenses and notices of its source content.

The generator reads the first 20 interleaved evaluation cases from `ds4_eval.c`
in [antirez/ds4](https://github.com/antirez/ds4):

- pinned commit: `b0309611041655f4e45671cfd9c9886aff161406`;
- source file SHA-256:
  `19545bf6c0a55cb91b7e3120344ec69ad4cfb5c87cf91e82ec4191a590013f23`;
- expected generated-manifest SHA-256:
  `d581a0a825d6da798c17f30614823ec9cb1dfdd1487c572373afcf1690399323`;
- transformation: C evaluation structs are converted to JSON and limited to
  cases 1-20 in upstream order; question/choice/answer content is not edited.

Per the pinned upstream source comments:

1. **GPQA Diamond** content is released under
   [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Source mirror:
   [Wanfq/gpqa](https://huggingface.co/datasets/Wanfq/gpqa).
2. **SuperGPQA** content is released under
   [ODC Attribution 1.0](https://opendatacommons.org/licenses/by/1-0/) and may
   include transformed third-party material subject to additional notices.
   Source: [m-a-p/SuperGPQA](https://huggingface.co/datasets/m-a-p/SuperGPQA).
3. **AIME 2025** cases came through the mirror cited upstream:
   [test-time-compute/aime_2025](https://huggingface.co/datasets/test-time-compute/aime_2025).
   A mirror's software/data label may not resolve all underlying contest-problem
   rights; users are responsible for confirming their intended use.

DwarfStar describes its embedded subset as an engineering regression harness,
not an official GPQA, SuperGPQA, or AIME score. Published result records here
retain only case indices, aggregate grades, performance metadata, and
cryptographic hashes—not question, choice, expected-answer, or generated-answer
text.
