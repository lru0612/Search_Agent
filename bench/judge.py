"""Answer judging for benchmark runs."""
from __future__ import annotations

import re
import string
from dataclasses import dataclass


@dataclass(frozen=True)
class JudgeResult:
    passed: bool
    score: float
    judge_status: str
    reason: str


def judge_answer(answer: str, expected_answers: list[str], mode: str = "auto") -> JudgeResult:
    if mode == "none":
        return JudgeResult(False, 0.0, "skipped", "judge disabled")
    if not expected_answers:
        return JudgeResult(False, 0.0, "skipped", "no expected answers")

    normalized_answer = normalize(answer)
    for expected in expected_answers:
        normalized_expected = normalize(expected)
        if normalized_expected and normalized_expected in normalized_answer:
            return JudgeResult(True, 1.0, "heuristic", f"matched expected answer: {expected}")
    return JudgeResult(False, 0.0, "heuristic", "no expected answer found in final answer")


def answer_recall(text: str, expected_answers: list[str]) -> float:
    """Return the fraction of expected answers found in text."""
    if not expected_answers:
        return 0.0
    normalized_text = normalize(text)
    hits = 0
    for expected in expected_answers:
        normalized_expected = normalize(expected)
        if normalized_expected and normalized_expected in normalized_text:
            hits += 1
    return round(hits / len(expected_answers), 4)


def normalize(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"\s+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()
