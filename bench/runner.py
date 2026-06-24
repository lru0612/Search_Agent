"""Benchmark runner that executes Agentic Search graph cases."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from app.config import get_settings
from app.observability import Tracer
from bench.cases import BenchCase
from bench.judge import answer_recall, judge_answer
from bench.report import write_diagnosis, write_report


class RunnableGraph(Protocol):
    def astream(self, graph_input: Any, config: dict[str, Any], stream_mode: str): ...
    def get_state(self, config: dict[str, Any]): ...


@dataclass
class BenchRunConfig:
    suite: str
    split: str
    run_id: str
    run_dir: Path
    max_case_seconds: float
    judge_mode: str = "auto"


def make_run_id(suite: str, split: str) -> str:
    return f"{suite}_{split}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


async def run_cases(
    cases: list[BenchCase],
    config: BenchRunConfig,
    graph: RunnableGraph | None = None,
) -> list[dict[str, Any]]:
    config.run_dir.mkdir(parents=True, exist_ok=True)
    results_path = config.run_dir / "results.jsonl"
    if graph is None:
        from app.agent.graph import build_graph

        graph = build_graph()

    results: list[dict[str, Any]] = []
    with results_path.open("w", encoding="utf-8") as f:
        for case in cases:
            result = await run_case(case, config, graph)
            results.append(result)
            f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
            f.flush()
    write_report(config.run_dir)
    write_diagnosis(config.run_dir)
    return results


async def run_case(case: BenchCase, config: BenchRunConfig, graph: RunnableGraph) -> dict[str, Any]:
    started = time.monotonic()
    trace_group = f"{config.suite}_{config.split}"
    session_id = case.id
    tracer = Tracer(session_id, trace_group=trace_group, run_id=config.run_id)
    graph_config = _graph_config(case, config)
    print(f"[bench] START {case.id}: {case.question}", flush=True)
    tracer.emit("bench_case_start", case=asdict(case), run_dir=str(config.run_dir))

    try:
        result = await asyncio.wait_for(
            _run_graph_case(case, config, graph, tracer, graph_config),
            timeout=config.max_case_seconds,
        )
    except asyncio.TimeoutError:
        elapsed = round(time.monotonic() - started, 3)
        tracer.emit("bench_case_timeout", elapsed_s=elapsed)
        result = _result_from_state(
            case,
            elapsed,
            tracer.summary()["trace_file"],
            graph,
            graph_config,
            config.judge_mode,
        )
        result.update({"failure_type": "timeout", "error": f"timed out after {config.max_case_seconds}s"})
        print(f"[bench] TIMEOUT {case.id} elapsed={elapsed}s trace={result['trace_file']}", flush=True)
        return result
    except Exception as e:
        elapsed = round(time.monotonic() - started, 3)
        tracer.emit("bench_case_error", error=f"{type(e).__name__}: {e}", elapsed_s=elapsed)
        result = _result_from_state(
            case,
            elapsed,
            tracer.summary()["trace_file"],
            graph,
            graph_config,
            config.judge_mode,
        )
        result.update({"failure_type": "runtime_bug", "error": f"{type(e).__name__}: {e}"})
        print(f"[bench] ERROR {case.id} {result['error']} trace={result['trace_file']}", flush=True)
        return result

    status = "PASS" if result["passed"] else "FAIL"
    print(
        f"[bench] {status} {case.id} score={result['score']} "
        f"failure={result['failure_type']} elapsed={result['elapsed_s']}s trace={result['trace_file']}",
        flush=True,
    )
    return result


async def _run_graph_case(
    case: BenchCase,
    config: BenchRunConfig,
    graph: RunnableGraph,
    tracer: Tracer,
    graph_config: dict[str, Any],
) -> dict[str, Any]:
    graph_input = {
        "query": case.question,
        "clarifications": [],
        "clarify_rounds": 0,
        "step_count": 0,
        "total_tokens": 0,
        "stagnant_steps": 0,
        "reflect_rounds": 0,
        "budget_exhausted": False,
        "phase": "CLARIFYING",
        "action_history": [],
        "evidence": {},
        "candidate_docs": {},
        "curated_evidence": {},
        "verification_records": [],
        "pruned_candidate_ids": [],
        "prune_history": [],
        "active_error": "",
        "scratchpad_summary": "",
        "invalid_action_count": 0,
        "self_correction_success_count": 0,
        "ask_user_count": 0,
        "tool_error_count": 0,
        "tool_error_recovery_count": 0,
        "planner_context_tokens": 0,
        "answer_context_tokens": 0,
        "disabled_actions": ["ask_user"],
    }
    started = time.monotonic()

    stream = graph.astream(graph_input, config=graph_config, stream_mode="updates")
    try:
        async for chunk in stream:
            for node, update in chunk.items():
                if node == "__interrupt__":
                    payload = update[0].value if update else {}
                    tracer.emit("bench_interrupt", payload=payload)
                    raise RuntimeError("benchmark case requested user input")
                if isinstance(update, dict):
                    tracer.emit("bench_node", node=node, preview=_preview_update(update))
                else:
                    tracer.emit("bench_node", node=node, preview=str(update)[:300])
    finally:
        close = getattr(stream, "aclose", None)
        if close:
            await close()

    elapsed = round(time.monotonic() - started, 3)
    result = _result_from_state(
        case,
        elapsed,
        tracer.summary()["trace_file"],
        graph,
        graph_config,
        config.judge_mode,
    )
    tracer.emit(
        "bench_case_done",
        passed=result["passed"],
        score=result["score"],
        failure_type=result["failure_type"],
        elapsed_s=elapsed,
    )
    return result


def _graph_config(case: BenchCase, config: BenchRunConfig) -> dict[str, Any]:
    llm_timeout = min(30.0, max(5.0, config.max_case_seconds / 4))
    return {
        "configurable": {
            "thread_id": case.id,
            "model_override": {
                "timeout": llm_timeout,
                "max_retries": 0,
            },
        },
        "recursion_limit": 100,
    }


def _result_from_state(
    case: BenchCase,
    elapsed: float,
    trace_file: str,
    graph: RunnableGraph,
    graph_config: dict[str, Any],
    judge_mode: str,
) -> dict[str, Any]:
    try:
        state = graph.get_state(graph_config).values
    except Exception:
        state = {}
    answer = str(state.get("final_answer") or "")
    sources = state.get("sources") or {}
    source_text = _source_text(sources)
    recall = answer_recall(source_text, case.answers)
    fa_recall = answer_recall(answer, case.answers)
    judged = judge_answer(answer, case.answers, mode=judge_mode)
    failure_type = "" if judged.passed else "answer_failure"
    result = _base_result(case, elapsed, trace_file)
    result.update(
        {
            "answer": answer,
            "passed": judged.passed,
            "score": judged.score,
            "recall": recall,
            "fa_recall": fa_recall,
            "judge_status": judged.judge_status,
            "judge_reason": judged.reason,
            "failure_type": failure_type,
            "total_tokens": int(state.get("total_tokens") or 0),
            "step_count": int(state.get("step_count") or 0),
            "citation_count": len(state.get("cited_sources") or []),
            "source_count": len(sources),
            "candidate_count": len(state.get("candidate_docs") or {}),
            "curated_evidence_count": len(state.get("curated_evidence") or {}),
            "verification_count": len(state.get("verification_records") or []),
            "pruned_candidate_count": len(state.get("pruned_candidate_ids") or []),
            "searched_queries": state.get("searched_queries") or [],
            "visited_urls": state.get("visited_urls") or [],
            "phase": state.get("phase") or "",
            "invalid_action_count": int(state.get("invalid_action_count") or 0),
            "self_correction_success_count": int(state.get("self_correction_success_count") or 0),
            "ask_user_count": int(state.get("ask_user_count") or 0),
            "tool_error_count": int(state.get("tool_error_count") or 0),
            "tool_error_recovery_count": int(state.get("tool_error_recovery_count") or 0),
            "planner_context_tokens": int(state.get("planner_context_tokens") or 0),
            "answer_context_tokens": int(state.get("answer_context_tokens") or 0),
        }
    )
    return result


def make_config(
    suite: str,
    split: str,
    run_id: str | None = None,
    max_case_seconds: float | None = None,
    judge_mode: str = "auto",
) -> BenchRunConfig:
    settings = get_settings()
    rid = run_id or make_run_id(suite, split)
    return BenchRunConfig(
        suite=suite,
        split=split,
        run_id=rid,
        run_dir=Path(settings.bench_runs_dir) / rid,
        max_case_seconds=max_case_seconds or settings.bench_max_case_seconds,
        judge_mode=judge_mode,
    )


def _base_result(case: BenchCase, elapsed_s: float, trace_file: str) -> dict[str, Any]:
    return {
        "case_id": case.id,
        "question": case.question,
        "answer": "",
        "expected_answers": case.answers,
        "passed": False,
        "score": 0.0,
        "recall": 0.0,
        "fa_recall": 0.0,
        "failure_type": "",
        "elapsed_s": elapsed_s,
        "total_tokens": 0,
        "step_count": 0,
        "citation_count": 0,
        "candidate_count": 0,
        "curated_evidence_count": 0,
        "verification_count": 0,
        "pruned_candidate_count": 0,
        "invalid_action_count": 0,
        "self_correction_success_count": 0,
        "ask_user_count": 0,
        "tool_error_count": 0,
        "tool_error_recovery_count": 0,
        "planner_context_tokens": 0,
        "answer_context_tokens": 0,
        "trace_file": trace_file,
        "error": "",
    }


def _preview_update(update: dict[str, Any]) -> dict[str, Any]:
    preview: dict[str, Any] = {}
    for key in (
        "phase",
        "step_count",
        "total_tokens",
        "rewritten_queries",
        "reflect_feedback",
        "active_error",
        "parsed_action",
    ):
        if key in update:
            preview[key] = update[key]
    if "messages" in update:
        preview["messages"] = [str(getattr(m, "content", m))[:200] for m in update["messages"]]
    return preview


def _source_text(sources: Any) -> str:
    if isinstance(sources, dict):
        items = sources.values()
    else:
        items = sources or []
    parts: list[str] = []
    for source in items:
        if isinstance(source, dict):
            parts.extend(str(source.get(key) or "") for key in ("title", "url", "snippet"))
        else:
            parts.append(str(source))
    return "\n".join(parts)
