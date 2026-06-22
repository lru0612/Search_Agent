"""Utilities for preparing the official BrowseComp dataset."""
from __future__ import annotations

import base64
import csv
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from bench.cases import BenchCase

BROWSECOMP_CSV_URL = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"


def derive_key(password: str, length: int) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def decrypt(ciphertext_b64: str, password: str) -> str:
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
    return decrypted.decode()


def fetch_official_browsecomp_cases(url: str = BROWSECOMP_CSV_URL) -> list[BenchCase]:
    with urlopen(url, timeout=60) as resp:
        text = resp.read().decode("utf-8")
    rows = csv.DictReader(text.splitlines())
    cases: list[BenchCase] = []
    for idx, row in enumerate(rows, start=1):
        canary = row.get("canary", "")
        question = decrypt(row.get("problem", ""), canary)
        answer = decrypt(row.get("answer", ""), canary)
        cases.append(
            BenchCase(
                id=f"browsecomp_test_{idx:04d}",
                question=question,
                answers=[answer],
                metadata={
                    "suite": "browsecomp",
                    "split": "test",
                    "source": url,
                    "canary": canary,
                    "encrypted_problem": row.get("problem", ""),
                    "encrypted_answer": row.get("answer", ""),
                },
            )
        )
    return cases


def write_cases_jsonl(cases: list[BenchCase], out_path: str | Path, limit: int | None = None) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = cases[:limit] if limit else cases
    with path.open("w", encoding="utf-8") as f:
        for case in selected:
            f.write(
                json.dumps(
                    {
                        "id": case.id,
                        "question": case.question,
                        "answers": case.answers,
                        "metadata": case.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path
