"""Offline graph used to validate benchmark plumbing without API keys."""
from __future__ import annotations

from typing import Any


class _State:
    def __init__(self, values: dict[str, Any]):
        self.values = values


class MockAnswerGraph:
    """Small graph-shaped object that returns deterministic cited answers."""

    def __init__(self, answers_by_case: dict[str, str]):
        self.answers_by_case = answers_by_case
        self._states: dict[str, dict[str, Any]] = {}

    async def astream(self, graph_input, config, stream_mode):
        case_id = config["configurable"]["thread_id"]
        yield {"clarify": {"phase": "SEARCHING"}}
        yield {"rewrite": {"rewritten_queries": [graph_input["query"]]}}
        yield {"planner": {"step_count": 1}}
        yield {"parse_action": {"parsed_action": {"action": "finish"}}}
        yield {"answer": {"final_answer": self.answers_by_case.get(case_id, "Unknown [1].")}}
        answer = self.answers_by_case.get(case_id, "Unknown")
        self._states[case_id] = {
            "final_answer": f"{answer} [1].",
            "total_tokens": 42,
            "step_count": 1,
            "cited_sources": [{"id": 1}],
            "sources": {1: {"id": 1, "url": "mock://source", "title": "Mock", "snippet": answer}},
            "searched_queries": [graph_input["query"]],
            "visited_urls": [],
            "phase": "DONE",
            "invalid_action_count": 0,
            "self_correction_success_count": 0,
            "ask_user_count": 0,
            "tool_error_count": 0,
            "tool_error_recovery_count": 0,
            "planner_context_tokens": 100,
            "answer_context_tokens": 80,
        }

    def get_state(self, config):
        return _State(self._states[config["configurable"]["thread_id"]])
