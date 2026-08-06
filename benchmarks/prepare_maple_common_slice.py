#!/usr/bin/env python3
"""Build the fixed Maple regression slice from a pinned upstream ds4 source."""

import argparse
import ast
import hashlib
import json
import re
import urllib.request
from pathlib import Path

UPSTREAM_COMMIT = "b0309611041655f4e45671cfd9c9886aff161406"
UPSTREAM_URL = (
    "https://raw.githubusercontent.com/antirez/ds4/"
    f"{UPSTREAM_COMMIT}/ds4_eval.c"
)
UPSTREAM_SHA256 = "19545bf6c0a55cb91b7e3120344ec69ad4cfb5c87cf91e82ec4191a590013f23"
EXPECTED_MANIFEST_SHA256 = (
    "d581a0a825d6da798c17f30614823ec9cb1dfdd1487c572373afcf1690399323"
)
STRING = r'"(?:\\.|[^"\\])*"'
FIELD_RE = re.compile(
    rf"\.(source|id|domain|title|question|answer)\s*=\s*({STRING})"
)
CHOICE_RE = re.compile(rf"\.choice\[(\d+)\]\s*=\s*({STRING})")


def _case_blocks(source: str, limit: int) -> list[str]:
    marker = "static const eval_case eval_cases[] = {"
    start = source.index(marker) + len(marker) - 1
    blocks = []
    depth = 0
    block_start = None
    in_string = False
    escaped = False
    for index in range(start, len(source)):
        char = source[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
            if depth == 2:
                block_start = index
        elif char == "}":
            if depth == 2 and block_start is not None:
                blocks.append(source[block_start : index + 1])
                block_start = None
                if len(blocks) == limit:
                    return blocks
            depth -= 1
            if depth == 0:
                break
    raise RuntimeError(f"found only {len(blocks)} evaluation cases")


def _decode(literal: str) -> str:
    value = ast.literal_eval(literal)
    if not isinstance(value, str):
        raise TypeError("expected a C/Python-compatible string literal")
    return value


def build_manifest(source: str, limit: int = 20) -> dict:
    cases = []
    for index, block in enumerate(_case_blocks(source, limit), start=1):
        fields = {name: _decode(value) for name, value in FIELD_RE.findall(block)}
        choices = {
            int(choice): _decode(value) for choice, value in CHOICE_RE.findall(block)
        }
        expected = {"source", "id", "domain", "title", "question", "answer"}
        if fields.keys() != expected:
            missing = sorted(expected - fields.keys())
            extra = sorted(fields.keys() - expected)
            raise RuntimeError(f"case {index}: fields missing={missing}, extra={extra}")
        if choices and sorted(choices) != list(range(len(choices))):
            raise RuntimeError(f"case {index}: choices are not contiguous")
        cases.append(
            {
                "index": index,
                "source": fields["source"],
                "source_id": fields["id"],
                "domain": fields["domain"],
                "title": fields["title"],
                "question": fields["question"],
                "choices": [choices[i] for i in sorted(choices)],
                "answer": fields["answer"],
            }
        )
    return {
        "name": "maple-common-slice-20-v1",
        "description": (
            "Fixed audited 20-question regression slice patterned after ds4-eval: "
            "first 20 interleaved GPQA Diamond, SuperGPQA, and AIME2025 cases."
        ),
        "upstream_repository": "https://github.com/antirez/ds4",
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_file": "ds4_eval.c",
        "selection": "eval_cases[0:20] in upstream order",
        "licenses": {
            "GPQA Diamond": "CC BY 4.0",
            "GPQA Diamond (modified)": "CC BY 4.0; see upstream modification note",
            "SuperGPQA": "ODC-BY (with upstream dataset caveats)",
            "AIME2025": "MIT-licensed mirror cited by upstream",
        },
        "cases": cases,
        "upstream_file_sha256": UPSTREAM_SHA256,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "data/maple_common_slice_20.json",
    )
    parser.add_argument("--source", type=Path, help="use an already downloaded ds4_eval.c")
    args = parser.parse_args()

    raw = args.source.read_bytes() if args.source else urllib.request.urlopen(
        UPSTREAM_URL, timeout=30
    ).read()
    actual_upstream = hashlib.sha256(raw).hexdigest()
    if actual_upstream != UPSTREAM_SHA256:
        raise RuntimeError(
            f"upstream SHA-256 mismatch: {actual_upstream} != {UPSTREAM_SHA256}"
        )
    manifest = build_manifest(raw.decode("utf-8"))
    encoded = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode()
    actual_manifest = hashlib.sha256(encoded).hexdigest()
    if actual_manifest != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            f"manifest SHA-256 mismatch: {actual_manifest} != {EXPECTED_MANIFEST_SHA256}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    print(f"wrote {args.output} ({actual_manifest})")


if __name__ == "__main__":
    main()
