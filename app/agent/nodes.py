"""各图节点实现。"""
from __future__ import annotations

import json
from datetime import date

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from app.agent import prompts
from app.agent.context import estimate_tokens, trim_messages
from app.agent.state import AgentState
from app.citations import extract_citations, format_sources_for_prompt, register_source
from app.config import get_settings
from app.tools.reader import visit_page
from app.tools.schemas import TOOL_NAME_MAP, TOOL_SCHEMAS
from app.tools.search import web_search


def _llm(temperature: float = 0.2, streaming: bool = False) -> ChatOpenAI:
    """内部调用默认关闭流式：部分网关（如 OpenRouter）流式时在多个 chunk 重复携带
    usage，LangChain 聚合后 token 统计会虚高几十倍，导致预算误判。
    只有 answer 节点需要流式（前端打字机效果）。"""
    s = get_settings()
    return ChatOpenAI(
        api_key=s.openai_api_key,
        base_url=s.openai_base_url,
        model=s.model_name,
        temperature=temperature,
        timeout=120,
        max_retries=2,
        disable_streaming=not streaming,
    )


def _usage(msg: AIMessage) -> int:
    meta = msg.usage_metadata or {}
    return meta.get("total_tokens", 0)


def _fmt_clarifications(state: AgentState) -> str:
    items = state.get("clarifications") or []
    if not items:
        return "（无）"
    return "\n".join(f"Q: {c['question']}\nA: {c['answer']}" for c in items)


# ---------------------------------------------------------------- clarify

class ClarifyResult(BaseModel):
    is_ambiguous: bool = Field(description="查询是否需要澄清")
    reasons: list[str] = Field(default_factory=list, description="需要澄清的原因")
    clarifying_question: str = Field(default="", description="向用户提出的一个最关键的澄清问题")
    options: list[str] = Field(default_factory=list, description="候选答案选项，可为空")


async def clarify_node(state: AgentState) -> dict:
    """只做 LLM 判断，不做 interrupt（interrupt 放在独立节点，保证恢复重放确定性）。"""
    settings = get_settings()
    rounds = state.get("clarify_rounds", 0)
    updates: dict = {"phase": "CLARIFYING", "pending_question": {}}

    if rounds < settings.max_clarify_rounds:
        llm = _llm().with_structured_output(ClarifyResult, include_raw=True)
        raw = await llm.ainvoke(
            [
                SystemMessage(content=prompts.CLARIFY_SYSTEM),
                HumanMessage(
                    content=prompts.CLARIFY_USER_TMPL.format(
                        query=state["query"], clarifications=_fmt_clarifications(state)
                    )
                ),
            ]
        )
        result: ClarifyResult = raw["parsed"]
        updates["total_tokens"] = state.get("total_tokens", 0) + _usage(raw["raw"])

        if result.is_ambiguous and result.clarifying_question:
            updates["pending_question"] = {
                "question": result.clarifying_question,
                "options": result.options,
                "reasons": result.reasons,
            }
            return updates

    updates["phase"] = "SEARCHING"
    return updates


def clarify_router(state: AgentState) -> str:
    return "ask_clarify" if state.get("pending_question") else "rewrite"


async def ask_clarify_node(state: AgentState) -> dict:
    """interrupt 挂起等用户回复；节点内无其他副作用，重放安全。"""
    q = state["pending_question"]
    answer = interrupt({"type": "ask_user", **q})
    return {
        "clarifications": (state.get("clarifications") or []) + [
            {"question": q["question"], "answer": str(answer)}
        ],
        "clarify_rounds": state.get("clarify_rounds", 0) + 1,
        "pending_question": {},
    }


# ---------------------------------------------------------------- rewrite

class RewriteResult(BaseModel):
    queries: list[str] = Field(description="改写后的 1~3 个搜索查询")


async def rewrite_node(state: AgentState) -> dict:
    llm = _llm(temperature=0.3).with_structured_output(RewriteResult, include_raw=True)
    raw = await llm.ainvoke(
        [
            SystemMessage(content=prompts.REWRITE_SYSTEM.format(today=date.today().isoformat())),
            HumanMessage(
                content=prompts.REWRITE_USER_TMPL.format(
                    query=state["query"], clarifications=_fmt_clarifications(state)
                )
            ),
        ]
    )
    queries = raw["parsed"].queries[:3] or [state["query"]]
    task = (
        f"用户问题：{state['query']}\n"
        f"澄清补充：{_fmt_clarifications(state)}\n"
        f"建议的搜索查询（已优化）：{json.dumps(queries, ensure_ascii=False)}\n"
        "请开始搜索并回答。"
    )
    return {
        "rewritten_queries": queries,
        "messages": [HumanMessage(content=task)],
        "total_tokens": state.get("total_tokens", 0) + _usage(raw["raw"]),
        "phase": "SEARCHING",
    }


# ---------------------------------------------------------------- agent (ReAct 决策)

async def agent_node(state: AgentState) -> dict:
    settings = get_settings()
    step = state.get("step_count", 0) + 1

    extra = ""
    if state.get("reflect_feedback"):
        extra += f"\n质检反馈（请优先补足以下缺口）：{state['reflect_feedback']}"
    if state.get("stagnant_steps", 0) >= 2:
        extra += "\n注意：最近几步没有获得新信息，请改变搜索策略或考虑 Finish。"
    remaining = settings.max_steps - step
    if remaining <= 1:
        extra += "\n注意：这是最后一步预算，若无关键缺口请直接 Finish。"

    sys = SystemMessage(
        content=prompts.AGENT_SYSTEM.format(
            step=step, max_steps=settings.max_steps, today=date.today().isoformat(), extra=extra
        )
    )
    messages = trim_messages(list(state["messages"]))
    llm = _llm().bind_tools(TOOL_SCHEMAS, tool_choice="required")
    resp: AIMessage = await llm.ainvoke([sys, *messages])

    return {
        "messages": [resp],
        "step_count": step,
        "total_tokens": state.get("total_tokens", 0) + _usage(resp),
    }


def agent_router(state: AgentState) -> str:
    settings = get_settings()
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    name = TOOL_NAME_MAP.get(tool_calls[0]["name"], "") if tool_calls else ""

    if name == "finish":
        return "reflect"
    # 预算耗尽：强制进入反思/回答
    if (
        state.get("step_count", 0) >= settings.max_steps
        or state.get("total_tokens", 0) >= settings.token_budget
    ):
        return "force_finish"
    if not tool_calls:
        return "reflect"
    return "tools"


# ---------------------------------------------------------------- tools 执行

async def tools_node(state: AgentState) -> dict:
    """执行 agent 选择的工具，登记来源、维护去重与停滞计数。"""
    last: AIMessage = state["messages"][-1]
    sources = dict(state.get("sources") or {})
    visited = list(state.get("visited_urls") or [])
    searched = list(state.get("searched_queries") or [])
    stagnant = state.get("stagnant_steps", 0)
    out_messages: list[ToolMessage] = []
    updates: dict = {}

    for tc in last.tool_calls:
        name = TOOL_NAME_MAP.get(tc["name"], tc["name"])
        args = tc["args"]
        try:
            if name == "web_search":
                q = args["query"]
                if q.strip().lower() in (s.strip().lower() for s in searched):
                    content = f"该查询「{q}」已搜索过，请换一个角度或直接阅读已有结果。"
                    stagnant += 1
                else:
                    searched.append(q)
                    results = await web_search(q, int(args.get("max_results", 5)))
                    if not results:
                        content = "未找到结果，请尝试换关键词。"
                        stagnant += 1
                    else:
                        lines = []
                        for r in results:
                            sources, sid = register_source(sources, r["url"], r["title"], r["snippet"])
                            lines.append(f"[{sid}] {r['title']}\nURL: {r['url']}\n摘要: {r['snippet']}")
                        content = f"搜索「{q}」返回 {len(results)} 条结果：\n\n" + "\n\n".join(lines)
                        stagnant = 0

            elif name == "visit_page":
                url = args["url"]
                if url in visited:
                    content = f"页面 {url} 已访问过，内容见之前的记录。"
                    stagnant += 1
                else:
                    page = await visit_page(url)
                    visited.append(url)
                    sources, sid = register_source(sources, url, page["title"], page["content"][:500])
                    content = f"[{sid}] 页面《{page['title']}》正文：\n\n{page['content']}"
                    stagnant = 0

            elif name == "ask_user":
                answer = interrupt(
                    {
                        "type": "ask_user",
                        "question": args["question"],
                        "options": args.get("options") or [],
                    }
                )
                content = f"用户回复：{answer}"
                updates["clarifications"] = (state.get("clarifications") or []) + [
                    {"question": args["question"], "answer": str(answer)}
                ]
                stagnant = 0
            else:
                content = f"未知工具 {name}"
        except Exception as e:  # 工具失败作为 observation 回喂
            content = f"工具 {name} 执行失败：{type(e).__name__}: {e}。请调整参数或换一种方式。"

        out_messages.append(ToolMessage(content=content, tool_call_id=tc["id"]))

    updates.update(
        {
            "messages": out_messages,
            "sources": sources,
            "visited_urls": visited,
            "searched_queries": searched,
            "stagnant_steps": stagnant,
        }
    )
    return updates


# ---------------------------------------------------------------- force_finish

async def force_finish_node(state: AgentState) -> dict:
    """预算耗尽：补一条 ToolMessage 维持消息配对，再走反思。"""
    last = state["messages"][-1]
    out = []
    for tc in getattr(last, "tool_calls", None) or []:
        out.append(ToolMessage(content="[系统] 搜索预算已用尽，进入回答阶段。", tool_call_id=tc["id"]))
    return {
        "messages": out,
        "budget_exhausted": True,
        "finish_outline": state.get("finish_outline", "") or "（预算耗尽，基于现有证据回答）",
    }


# ---------------------------------------------------------------- reflect

class ReflectResult(BaseModel):
    sufficient: bool = Field(description="证据是否足以回答")
    reasoning: str = Field(description="简要理由")
    missing_aspects: list[str] = Field(default_factory=list, description="缺失的方面")
    conflicts: list[str] = Field(default_factory=list, description="来源间的矛盾点")


async def reflect_node(state: AgentState) -> dict:
    settings = get_settings()
    last = state["messages"][-1]
    updates: dict = {"phase": "REFLECTING"}

    # 提取 finish 提纲并补 ToolMessage 维持配对
    outline = state.get("finish_outline", "")
    tool_calls = getattr(last, "tool_calls", None) or []
    if tool_calls and TOOL_NAME_MAP.get(tool_calls[0]["name"]) == "finish":
        outline = tool_calls[0]["args"].get("answer_outline", "")
        updates["messages"] = [
            ToolMessage(content="[系统] 已收到提纲，正在质检证据……", tool_call_id=tool_calls[0]["id"])
        ]
    updates["finish_outline"] = outline

    rounds = state.get("reflect_rounds", 0)
    # 预算耗尽或反思次数用完：直接放行
    if state.get("budget_exhausted") or rounds >= settings.max_reflect_rounds:
        updates["reflect_feedback"] = ""
        return updates

    history = "\n".join(
        f"- {str(m.content)[:150]}" for m in state["messages"][-8:] if isinstance(m, ToolMessage)
    )
    llm = _llm().with_structured_output(ReflectResult, include_raw=True)
    raw = await llm.ainvoke(
        [
            SystemMessage(content=prompts.REFLECT_SYSTEM),
            HumanMessage(
                content=prompts.REFLECT_USER_TMPL.format(
                    query=state["query"],
                    outline=outline or "（无提纲）",
                    sources=format_sources_for_prompt(state.get("sources") or {}),
                    history=history or "（无）",
                )
            ),
        ]
    )
    result: ReflectResult = raw["parsed"]
    updates["total_tokens"] = state.get("total_tokens", 0) + _usage(raw["raw"])
    updates["reflect_rounds"] = rounds + 1

    if not result.sufficient and (result.missing_aspects or result.conflicts):
        feedback = "；".join(result.missing_aspects + [f"需核实矛盾：{c}" for c in result.conflicts])
        updates["reflect_feedback"] = feedback
    else:
        updates["reflect_feedback"] = ""
    return updates


def reflect_router(state: AgentState) -> str:
    return "agent" if state.get("reflect_feedback") else "answer"


# ---------------------------------------------------------------- answer

async def answer_node(state: AgentState) -> dict:
    sources = state.get("sources") or {}
    # 证据 = 全部 ToolMessage 内容（截断控制长度）
    evidence_parts = []
    budget = 30_000
    for m in reversed(state["messages"]):
        if isinstance(m, ToolMessage):
            text = str(m.content)[:4000]
            if budget - estimate_tokens(text) < 0:
                break
            budget -= estimate_tokens(text)
            evidence_parts.append(text)
    evidence = "\n\n---\n\n".join(reversed(evidence_parts)) or "（无）"

    budget_note = (
        "搜索预算耗尽，信息可能不完整，请在回答开头声明这一点"
        if state.get("budget_exhausted")
        else "信息充分，直接回答"
    )
    llm = _llm(temperature=0.3, streaming=True)
    resp = await llm.ainvoke(
        [
            SystemMessage(content=prompts.ANSWER_SYSTEM.format(budget_note=budget_note)),
            HumanMessage(
                content=prompts.ANSWER_USER_TMPL.format(
                    query=state["query"],
                    clarifications=_fmt_clarifications(state),
                    outline=state.get("finish_outline") or "（无）",
                    sources=format_sources_for_prompt(sources),
                    evidence=evidence,
                )
            ),
        ]
    )
    answer = str(resp.content)
    return {
        "final_answer": answer,
        "cited_sources": extract_citations(answer, sources),
        "total_tokens": state.get("total_tokens", 0) + _usage(resp),
        "phase": "DONE",
        "messages": [AIMessage(content=answer)],
    }
