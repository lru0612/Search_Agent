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
    elif node == "agent":
        msgs = update.get("messages") or []
        for m in msgs:
            for tc in getattr(m, "tool_calls", None) or []:
                events.append(
                    tracer.emit("action", tool=tc["name"], args=tc["args"], step=update.get("step_count"))
                )
    elif node == "tools":
        for m in update.get("messages") or []:
            events.append(tracer.emit("observation", preview=str(m.content)[:300]))
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
    tracer = _tracers.setdefault(session_id, Tracer(session_id))
    cancel = _cancel_events.setdefault(session_id, asyncio.Event())
    cancel.clear()

    config = {"configurable": {"thread_id": session_id}, "recursion_limit": 100}
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
