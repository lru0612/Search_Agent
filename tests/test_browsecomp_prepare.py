"""Offline tests for BrowseComp preparation helpers.

运行：python -m tests.test_browsecomp_prepare
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from bench.browsecomp import decrypt, derive_key, write_cases_jsonl
from bench.cases import BenchCase, load_cases


def encrypt_for_test(plaintext: str, password: str) -> str:
    import base64

    raw = plaintext.encode()
    key = derive_key(password, len(raw))
    encrypted = bytes(a ^ b for a, b in zip(raw, key))
    return base64.b64encode(encrypted).decode()


def main() -> int:
    canary = "unit-canary"
    encrypted = encrypt_for_test("DART", canary)
    assert decrypt(encrypted, canary) == "DART"
    print("PASS: BrowseComp XOR decrypt round-trip", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "browsecomp.jsonl"
        write_cases_jsonl(
            [
                BenchCase(
                    id="case_1",
                    question="Question?",
                    answers=["Answer"],
                    metadata={"suite": "browsecomp"},
                )
            ],
            out,
        )
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        assert rows[0]["question"] == "Question?"
        loaded = load_cases("browsecomp", "test", data_file=str(out))
        assert loaded[0].answers == ["Answer"]
        print("PASS: BrowseComp JSONL writes and loads", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
