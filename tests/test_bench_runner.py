"""Offline tests for benchmark runner plumbing.

运行：python -m tests.test_bench_runner
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path
from typing import Any

from bench.cases import BenchCase
from bench.runner import BenchRunConfig, run_cases


class _State:
    def __init__(self, values: dict[str, Any]):
        self.values = values


class FakeGraph:
    def __init__(self, mode: str):
        self.mode = mode
        self._state: dict[str, Any] = {}

    async def astream(self, graph_input, config, stream_mode):
        if self.mode == "partial_timeout":
            yield {"planner": {"step_count": 1, "total_tokens": 99}}
            self._state = {
                "final_answer": "",
                "total_tokens": 99,
                "step_count": 1,
                "cited_sources": [],
                "sources": {1: {"id": 1, "url": "https://example.com", "title": "Example", "snippet": "Anthropic"}},
                "searched_queries": ["Claude developer"],
                "visited_urls": [],
                "phase": "SEARCHING",
            }
            await asyncio.sleep(1)
            return
        if self.mode == "timeout":
            await asyncio.sleep(1)
            return
        if self.mode == "error":
            raise RuntimeError("boom")
        assert "ask_user" in graph_input.get("disabled_actions", [])
        yield {"clarify": {"phase": "SEARCHING"}}
        yield {"answer": {"final_answer": "The answer is Anthropic [1]."}}
        self._state = {
            "final_answer": "The answer is Anthropic [1]." if self.mode == "pass" else "The answer is OpenAI [1].",
            "total_tokens": 123,
            "step_count": 3,
            "cited_sources": [{"id": 1}],
            "sources": {1: {"id": 1, "url": "https://example.com", "title": "Example", "snippet": "Anthropic"}},
            "searched_queries": ["Claude developer"],
            "visited_urls": ["https://example.com"],
            "phase": "DONE",
        }

    def get_state(self, config):
        return _State(self._state)


async def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cases = [BenchCase(id="case_ok", question="Who developed Claude?", answers=["Anthropic"])]
        config = BenchRunConfig(
            suite="browsecomp",
            split="smoke",
            run_id="test_run",
            run_dir=root / "runs" / "test_run",
            max_case_seconds=0.05,
            judge_mode="auto",
        )

        results = await run_cases(cases, config, graph=FakeGraph("pass"))
        assert results[0]["passed"] is True, results
        assert (config.run_dir / "results.jsonl").exists()
        assert (config.run_dir / "summary.json").exists()
        assert (config.run_dir / "failures.md").exists()
        trace_file = Path(results[0]["trace_file"])
        assert "browsecomp_smoke" in str(trace_file), trace_file
        assert trace_file.exists(), trace_file
        print("PASS: successful case writes results, report, and grouped trace", flush=True)

        wrong_config = BenchRunConfig(
            suite="browsecomp",
            split="smoke",
            run_id="wrong_run",
            run_dir=root / "runs" / "wrong_run",
            max_case_seconds=0.05,
            judge_mode="auto",
        )
        wrong = await run_cases(cases, wrong_config, graph=FakeGraph("wrong"))
        assert wrong[0]["passed"] is False
        assert wrong[0]["failure_type"] == "answer_failure"
        assert wrong[0]["recall"] == 1.0
        assert wrong[0]["fa_recall"] == 0.0
        print("PASS: wrong answer is classified as answer_failure", flush=True)

        error_config = BenchRunConfig(
            suite="browsecomp",
            split="smoke",
            run_id="error_run",
            run_dir=root / "runs" / "error_run",
            max_case_seconds=0.05,
            judge_mode="auto",
        )
        errored = await run_cases(cases, error_config, graph=FakeGraph("error"))
        assert errored[0]["failure_type"] == "runtime_bug"
        assert "boom" in errored[0]["error"]
        print("PASS: exception is classified as runtime_bug", flush=True)

        timeout_config = BenchRunConfig(
            suite="browsecomp",
            split="smoke",
            run_id="timeout_run",
            run_dir=root / "runs" / "timeout_run",
            max_case_seconds=0.01,
            judge_mode="auto",
        )
        timed_out = await run_cases(cases, timeout_config, graph=FakeGraph("timeout"))
        assert timed_out[0]["failure_type"] == "timeout"
        print("PASS: timeout is classified as timeout", flush=True)

        partial_timeout_config = BenchRunConfig(
            suite="browsecomp",
            split="smoke",
            run_id="partial_timeout_run",
            run_dir=root / "runs" / "partial_timeout_run",
            max_case_seconds=0.01,
            judge_mode="auto",
        )
        partial = await run_cases(cases, partial_timeout_config, graph=FakeGraph("partial_timeout"))
        assert partial[0]["failure_type"] == "timeout"
        assert partial[0]["step_count"] == 1
        assert partial[0]["total_tokens"] == 99
        assert partial[0]["recall"] == 1.0
        print("PASS: timeout preserves partial state", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
