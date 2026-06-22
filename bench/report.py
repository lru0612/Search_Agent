"""Report generation for benchmark runs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_results(run_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / "results.jsonl"
    results: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for r in results if r.get("passed"))
    timeout = sum(1 for r in results if r.get("failure_type") == "timeout")
    errors = sum(1 for r in results if r.get("failure_type") == "runtime_bug")
    citation_ready = [r for r in results if r.get("answer")]
    cited = sum(1 for r in citation_ready if r.get("citation_count", 0) > 0)
    return {
        "total": total,
        "passed": passed,
        "accuracy": round(passed / total, 4) if total else 0.0,
        "avg_recall": _avg(results, "recall"),
        "avg_fa_recall": _avg(results, "fa_recall"),
        "timeout_rate": round(timeout / total, 4) if total else 0.0,
        "error_rate": round(errors / total, 4) if total else 0.0,
        "avg_elapsed_s": _avg(results, "elapsed_s"),
        "avg_total_tokens": _avg(results, "total_tokens"),
        "avg_steps": _avg(results, "step_count"),
        "avg_invalid_actions": _avg(results, "invalid_action_count"),
        "avg_self_corrections": _avg(results, "self_correction_success_count"),
        "avg_ask_user": _avg(results, "ask_user_count"),
        "avg_tool_errors": _avg(results, "tool_error_count"),
        "avg_tool_error_recoveries": _avg(results, "tool_error_recovery_count"),
        "avg_planner_context_tokens": _avg(results, "planner_context_tokens"),
        "avg_answer_context_tokens": _avg(results, "answer_context_tokens"),
        "citation_coverage": round(cited / len(citation_ready), 4) if citation_ready else 0.0,
    }


def write_report(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    results = load_results(run_path)
    summary = summarize_results(results)
    (run_path / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_path / "failures.md").write_text(render_failures(results), encoding="utf-8")
    return summary


def render_failures(results: list[dict[str, Any]]) -> str:
    lines = ["# Benchmark Failures", ""]
    failures = [r for r in results if not r.get("passed")]
    if not failures:
        lines.append("No failures.")
        return "\n".join(lines) + "\n"
    for r in failures:
        lines.extend(
            [
                f"## {r.get('case_id')}",
                "",
                f"- failure_type: `{r.get('failure_type')}`",
                f"- score: `{r.get('score')}`",
                f"- recall: `{r.get('recall', 0.0)}`",
                f"- fa_recall: `{r.get('fa_recall', 0.0)}`",
                f"- invalid_actions: `{r.get('invalid_action_count', 0)}`",
                f"- ask_user_count: `{r.get('ask_user_count', 0)}`",
                f"- question: {r.get('question')}",
                f"- expected: {', '.join(r.get('expected_answers') or [])}",
                f"- answer: {r.get('answer') or '(empty)'}",
                f"- error: {r.get('error') or '(none)'}",
                f"- trace: `{r.get('trace_file')}`",
                "",
            ]
        )
    return "\n".join(lines)


def write_diagnosis(run_dir: str | Path) -> str:
    run_path = Path(run_dir)
    results = load_results(run_path)
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        if not r.get("passed"):
            groups.setdefault(str(r.get("failure_type") or "unknown"), []).append(r)

    lines = ["# Diagnosis", ""]
    if not groups:
        lines.append("No failing cases. Keep the current patch set and expand validation.")
    for failure_type, items in sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True):
        lines.extend([f"## {failure_type}", "", f"- cases: {len(items)}"])
        if failure_type == "timeout":
            lines.append("- patch idea: inspect traces for repeated searches and add earlier Finish/strategy change.")
        elif failure_type == "runtime_bug":
            lines.append("- patch idea: fix exception path first; do not tune prompts until runtime is stable.")
        elif failure_type == "answer_failure":
            lines.append("- patch idea: compare searched queries vs expected entity and strengthen evidence synthesis.")
        else:
            lines.append("- patch idea: review trace and classify before changing agent behavior.")
        lines.append("")
    text = "\n".join(lines) + "\n"
    (run_path / "diagnosis.md").write_text(text, encoding="utf-8")
    return text


def _avg(results: list[dict[str, Any]], key: str) -> float:
    nums = [float(r.get(key) or 0) for r in results]
    return round(sum(nums) / len(nums), 3) if nums else 0.0
