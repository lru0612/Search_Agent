"""Benchmark case definitions and built-in smoke suites."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchCase:
    id: str
    question: str
    answers: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


_BROWSECOMP_SMOKE = [
    BenchCase(
        id="browsecomp_smoke_001",
        question="What is the name of the NASA mission that intentionally impacted asteroid Dimorphos?",
        answers=["DART", "Double Asteroid Redirection Test"],
        metadata={"topic": "science", "answer_type": "short"},
    ),
    BenchCase(
        id="browsecomp_smoke_002",
        question="Which company developed the Claude family of AI assistants?",
        answers=["Anthropic"],
        metadata={"topic": "ai", "answer_type": "entity"},
    ),
    BenchCase(
        id="browsecomp_smoke_003",
        question="What is the capital city of New Zealand?",
        answers=["Wellington"],
        metadata={"topic": "geography", "answer_type": "entity"},
    ),
    BenchCase(
        id="browsecomp_smoke_004",
        question="What programming language is FastAPI primarily used with?",
        answers=["Python"],
        metadata={"topic": "software", "answer_type": "entity"},
    ),
    BenchCase(
        id="browsecomp_smoke_005",
        question="What search API provider is used by this project for WebSearch?",
        answers=["Tavily"],
        metadata={"topic": "project", "answer_type": "entity"},
    ),
]


def load_cases(suite: str, split: str, limit: int | None = None, data_file: str | None = None) -> list[BenchCase]:
    if data_file:
        cases = _load_jsonl_cases(Path(data_file))
    elif suite == "browsecomp" and split == "smoke":
        cases = list(_BROWSECOMP_SMOKE)
    else:
        raise ValueError(
            f"No built-in cases for suite={suite!r} split={split!r}. "
            "Pass --data-file with JSONL cases for this split."
        )
    return cases[:limit] if limit else cases


def _load_jsonl_cases(path: Path) -> list[BenchCase]:
    cases: list[BenchCase] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            answers = obj.get("answers") or obj.get("expected_answers") or []
            if isinstance(answers, str):
                answers = [answers]
            cases.append(
                BenchCase(
                    id=str(obj.get("id") or f"{path.stem}_{line_no}"),
                    question=str(obj["question"]),
                    answers=[str(a) for a in answers],
                    metadata=dict(obj.get("metadata") or {}),
                )
            )
    return cases

