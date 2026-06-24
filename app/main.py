"""FastAPI 入口：/api/chat（SSE 流式）、interrupt 恢复、取消。"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, AsyncIterator

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langgraph.types import Command
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.agent.graph import graph
from app.observability import Tracer

app = FastAPI(title="Agentic Search")

# session_id -> 取消事件
_cancel_events: dict[str, asyncio.Event] = {}
# session_id -> Tracer（跨 ask_user 恢复复用）
_tracers: dict[str, Tracer] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    resume: bool = False  # 是否为 ask_user 中断后的恢复请求
    # 可选的临时模型覆盖（前端会话级，不持久化）：{model, api_key, base_url}
    model_override: dict[str, str] | None = None
    # benchmark / 自动化测试可传入，用于按组别归档 traces/{trace_group}/{run_id}/...
    trace_group: str | None = None
    run_id: str | None = None


def _sse(event: dict) -> dict:
    return {"event": event["type"], "data": json.dumps(event, ensure_ascii=False, default=str)}


def _node_event(tracer: Tracer, node: str, update: dict[str, Any]) -> list[dict]:
    """把节点的 state 更新转成前端可读事件。"""
    events: list[dict] = []
    if node == "clarify":
        events.append(tracer.emit("node", node=node, label="判断查询是否需要澄清"))
    elif node == "rewrite":
        events.append(
            tracer.emit("rewrite", queries=update.get("rewritten_queries", []), label="重写搜索查询")
        )
    elif node in ("agent", "planner"):
        msgs = update.get("messages") or []
        for m in msgs:
            for tc in getattr(m, "tool_calls", None) or []:
                events.append(
                    tracer.emit("action", tool=tc["name"], args=tc["args"], step=update.get("step_count"))
                )
        action = update.get("parsed_action") or {}
        if action:
            events.append(tracer.emit("action", tool=action.get("action"), args=action.get("args"), step=update.get("step_count")))
    elif node == "parse_action":
        if update.get("active_error"):
            events.append(tracer.emit("invalid_action", error=update.get("active_error")))
        action = update.get("parsed_action") or {}
        if action and update.get("action_source") == "json":
            events.append(
                tracer.emit(
                    "action",
                    tool=action.get("action"),
                    args=action.get("args"),
                    step=(history[-1].get("step") if (history := update.get("action_history") or []) else None),
                )
            )
    elif node in ("tools", "execute_action"):
        for m in update.get("messages") or []:
            events.append(tracer.emit("observation", preview=str(m.content)[:300]))
        history = update.get("action_history") or []
        last_action = history[-1] if history else {}
        action = last_action.get("action")
        if action in {"web_search", "visit_page"}:
            candidates = list((update.get("candidate_docs") or {}).values())
            active = [c for c in candidates if c.get("status") != "pruned"]
            events.append(
                tracer.emit(
                    "candidate",
                    count=len(active),
                    latest=active[-5:],
                    label=f"候选文档池更新：{len(active)} 条可用候选",
                )
            )
        elif action == "curate_evidence":
            curated = list((update.get("curated_evidence") or {}).values())
            events.append(tracer.emit("curate", evidence=curated[-1] if curated else {}, label="已保留关键证据"))
        elif action == "prune_candidates":
            prunes = update.get("prune_history") or []
            events.append(tracer.emit("prune", record=prunes[-1] if prunes else {}, label="已剪枝低价值候选"))
        elif action == "verify_claim":
            records = update.get("verification_records") or []
            events.append(tracer.emit("verify", record=records[-1] if records else {}, label="已核验关键论断"))
    elif node == "force_finish":
        events.append(tracer.emit("budget_exhausted", label="搜索预算已用尽，强制进入回答"))
    elif node == "reflect":
        feedback = update.get("reflect_feedback", "")
        events.append(
            tracer.emit(
                "reflect",
                passed=not feedback,
                feedback=feedback,
                label="反思自检通过" if not feedback else f"反思发现缺口：{feedback[:120]}",
            )
        )
    return events


async def _run_graph(req: ChatRequest) -> AsyncIterator[dict]:
    session_id = req.session_id or uuid.uuid4().hex
    tracer_key = f"{req.trace_group or ''}:{req.run_id or ''}:{session_id}"
    tracer = _tracers.setdefault(
        tracer_key,
        Tracer(session_id, trace_group=req.trace_group, run_id=req.run_id),
    )
    cancel = _cancel_events.setdefault(session_id, asyncio.Event())
    cancel.clear()

    config = {"configurable": {"thread_id": session_id}, "recursion_limit": 100}
    if req.model_override:
        config["configurable"]["model_override"] = {
            k: v for k, v in req.model_override.items() if k in ("model", "api_key", "base_url") and v
        }
    graph_input: Any
    if req.resume:
        graph_input = Command(resume=req.message)
    else:
        graph_input = {
            "query": req.message,
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
        }

    yield _sse(tracer.emit("session", session_id=session_id))

    interrupted = False
    try:
        async for mode, chunk in graph.astream(
            graph_input, config=config, stream_mode=["updates", "messages"]
        ):
            if cancel.is_set():
                yield _sse(tracer.emit("cancelled", label="用户已取消"))
                return

            if mode == "messages":
                msg, meta = chunk
                # 仅流式转发 answer 节点的 token
                if meta.get("langgraph_node") == "answer" and getattr(msg, "content", ""):
                    yield _sse({"type": "answer_chunk", "text": msg.content})
                continue

            # mode == "updates"
            for node, update in chunk.items():
                if node == "__interrupt__":
                    payload = update[0].value if update else {}
                    interrupted = True
                    yield _sse(tracer.emit("ask_user", **payload))
                    continue
                if not isinstance(update, dict):
                    continue
                for ev in _node_event(tracer, node, update):
                    yield _sse(ev)
                if node == "answer":
                    yield _sse(
                        tracer.emit(
                            "final_answer",
                            answer=update.get("final_answer", ""),
                            citations=update.get("cited_sources", []),
                            total_tokens=update.get("total_tokens", 0),
                        )
                    )
    except Exception as e:
        yield _sse(tracer.emit("error", message=f"{type(e).__name__}: {e}"))
        return

    if not interrupted:
        yield _sse(tracer.emit("done", summary=tracer.summary()))


@app.post("/api/chat")
async def chat(req: ChatRequest):
    # 注意：sse-starlette 以 \r\n 作为行结尾，前端解析时已做归一化
    return EventSourceResponse(_run_graph(req))


@app.get("/api/config")
async def api_config():
    from app.config import get_settings

    return {"default_model": get_settings().model_name}


@app.post("/api/cancel/{session_id}")
async def cancel(session_id: str):
    ev = _cancel_events.get(session_id)
    if ev:
        ev.set()
    return {"ok": ev is not None}


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")
