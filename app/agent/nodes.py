"""各图节点实现。"""
from __future__ import annotations

import json
import re
import uuid
from datetime import date

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.config import get_config
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from app.agent import prompts
from app.agent.context import (
    build_answer_context,
    build_planner_context,
    build_reflect_context,
    estimate_tokens,
)
from app.agent.state import AgentState
from app.citations import extract_citations, format_sources_for_prompt, register_source
from app.config import get_settings
from app.tools.reader import visit_page
from app.tools.schemas import AskUser, Finish, TOOL_NAME_MAP, TOOL_SCHEMAS, VisitPage, WebSearch
from app.tools.search import web_search


def _llm(temperature: float = 0.2, streaming: bool = False) -> ChatOpenAI:
    """内部调用默认关闭流式：部分网关（如 OpenRouter）流式时在多个 chunk 重复携带
    usage，LangChain 聚合后 token 统计会虚高几十倍，导致预算误判。
    只有 answer 节点需要流式（前端打字机效果）。

    运行时 config["configurable"]["model_override"] 可携带前端传来的临时模型配置。"""
    s = get_settings()
    try:
        config = get_config()
    except RuntimeError:  # 不在图运行上下文中（如单测直接调用）
        config = {}
    o = (config.get("configurable") or {}).get("model_override") or {}
    return ChatOpenAI(
        api_key=o.get("api_key") or s.openai_api_key,
        base_url=o.get("base_url") or s.openai_base_url,
        model=o.get("model") or s.model_name,
        temperature=temperature,
        timeout=float(o.get("timeout") or s.llm_timeout_s),
        max_retries=int(o.get("max_retries") if o.get("max_retries") is not None else s.llm_max_retries),
        disable_streaming=not streaming,
    )


def _usage(msg: AIMessage) -> int:
    meta = msg.usage_metadata or {}
    return meta.get("total_tokens", 0)


def _extract_json_object(text: str) -> dict:
    """从模型输出中提取第一个 JSON object，兼容 ```json fenced block。"""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    return json.loads(text)


async def _structured_invoke(schema: type[BaseModel], messages: list) -> tuple[BaseModel, AIMessage]:
    """优先使用原生 structured output；不兼容时降级为 JSON prompt 手动解析。

    一些 OpenAI 兼容网关（Render 上常见的代理 / SiliconFlow / OpenRouter 组合）
    对 LangChain 的 with_structured_output 支持不完全，可能返回 parsed=None 或抛
    TypeError。这里做降级，避免首次 clarify 就中断。
    """
    try:
        raw = await _llm().with_structured_output(schema, include_raw=True).ainvoke(messages)
        parsed = raw.get("parsed")
        raw_msg = raw.get("raw")
        if parsed is not None and raw_msg is not None:
            return parsed, raw_msg
        if raw_msg is not None and getattr(raw_msg, "content", None):
            return schema.model_validate(_extract_json_object(str(raw_msg.content))), raw_msg
    except Exception:
        # 继续走 JSON fallback；错误会在 fallback 也失败时暴露。
        pass

    fallback_messages = [
        *messages,
        HumanMessage(
            content=(
                "请严格只输出一个 JSON 对象，不要输出 Markdown 或解释文字。\n"
                f"JSON schema: {json.dumps(schema.model_json_schema(), ensure_ascii=False)}"
            )
        ),
    ]
    raw_msg = await _llm().ainvoke(fallback_messages)
    return schema.model_validate(_extract_json_object(str(raw_msg.content))), raw_msg


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

    if "ask_user" in set(state.get("disabled_actions") or []):
        updates["phase"] = "SEARCHING"
        return updates

    if rounds < settings.max_clarify_rounds:
        result, raw_msg = await _structured_invoke(
            ClarifyResult,
            [
                SystemMessage(content=prompts.CLARIFY_SYSTEM),
                HumanMessage(
                    content=prompts.CLARIFY_USER_TMPL.format(
                        query=state["query"], clarifications=_fmt_clarifications(state)
                    )
                ),
            ],
        )
        updates["total_tokens"] = state.get("total_tokens", 0) + _usage(raw_msg)

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
    result, raw_msg = await _structured_invoke(
        RewriteResult,
        [
            SystemMessage(content=prompts.REWRITE_SYSTEM.format(today=date.today().isoformat())),
            HumanMessage(
                content=prompts.REWRITE_USER_TMPL.format(
                    query=state["query"], clarifications=_fmt_clarifications(state)
                )
            ),
        ],
    )
    queries = result.queries[:3] or [state["query"]]
    task = (
        f"用户问题：{state['query']}\n"
        f"澄清补充：{_fmt_clarifications(state)}\n"
        f"建议的搜索查询（已优化）：{json.dumps(queries, ensure_ascii=False)}\n"
        "请开始搜索并回答。"
    )
    return {
        "rewritten_queries": queries,
        "messages": [HumanMessage(content=task)],
        "total_tokens": state.get("total_tokens", 0) + _usage(raw_msg),
        "phase": "SEARCHING",
    }


# ---------------------------------------------------------------- planner (决策 sub-agent)

async def planner_node(state: AgentState) -> dict:
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
    planner_context, context_tokens = build_planner_context(state)
    user = HumanMessage(content=planner_context)
    try:
        llm = _llm().bind_tools(TOOL_SCHEMAS, tool_choice="required")
        resp: AIMessage = await llm.ainvoke([sys, user])
    except Exception as e:
        # 部分 OpenRouter 路由/模型不支持 tool_choice="required"。
        # 降级为普通 JSON 动作输出；解析和修复交给 parse_action_node。
        if "tool_choice" not in str(e) and "No endpoints found" not in str(e):
            raise
        action_msg = HumanMessage(
            content=(
                "当前模型不支持原生工具调用。请只输出一个 JSON 对象，不要输出 Markdown。\n"
                "schema: {\"action\":\"web_search|visit_page|ask_user|finish\", \"args\":{...}, \"reason\": string}\n"
                "参数要求：\n"
                "- web_search: {\"query\": string, \"max_results\": number}\n"
                "- visit_page: {\"url\": string, \"reason\": string}\n"
                "- ask_user: {\"question\": string, \"options\": string[]}\n"
                "- finish: {\"answer_outline\": string}"
            )
        )
        resp = await _llm().ainvoke([sys, user, action_msg])

    return {
        "messages": [resp],
        "step_count": step,
        "total_tokens": state.get("total_tokens", 0) + _usage(resp),
        "planner_context_tokens": state.get("planner_context_tokens", 0) + context_tokens,
        "phase": "SEARCHING",
    }


def _make_action_record(
    state: AgentState,
    action: str,
    args: dict,
    reason: str,
    status: str,
    observation_summary: str = "",
    error: str = "",
) -> dict:
    return {
        "step": state.get("step_count", 0),
        "action": action,
        "args": args,
        "reason": reason,
        "status": status,
        "observation_summary": observation_summary,
        "error": error,
    }


def _append_action_record(state: AgentState, record: dict) -> list[dict]:
    history = list(state.get("action_history") or [])
    history.append(record)
    return history[-30:]


def _validate_action(action: str, args: dict, disabled_actions: list[str] | None = None) -> None:
    if action in set(disabled_actions or []):
        raise ValueError(f"action disabled in this run: {action}")
    schema_by_action = {
        "web_search": WebSearch,
        "visit_page": VisitPage,
        "ask_user": AskUser,
        "finish": Finish,
    }
    schema = schema_by_action.get(action)
    if not schema:
        raise ValueError(f"unknown action: {action}")
    schema.model_validate(args)


def _action_from_tool_call(tc: dict) -> dict:
    name = TOOL_NAME_MAP.get(tc["name"], tc["name"])
    return {
        "action": name,
        "args": tc.get("args") or {},
        "reason": (tc.get("args") or {}).get("reason", ""),
        "id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
    }


def _action_from_json(content: str) -> dict:
    obj = _extract_json_object(content)
    return {
        "action": str(obj.get("action") or ""),
        "args": obj.get("args") or {},
        "reason": str(obj.get("reason") or ""),
        "id": f"call_{uuid.uuid4().hex[:12]}",
    }


async def parse_action_node(state: AgentState) -> dict:
    last: AIMessage = state["messages"][-1]
    try:
        tool_calls = getattr(last, "tool_calls", None) or []
        action = _action_from_tool_call(tool_calls[0]) if tool_calls else _action_from_json(str(last.content))
        _validate_action(action["action"], action["args"], state.get("disabled_actions") or [])
        correction = 1 if state.get("active_error") else 0
        updates: dict = {
            "parsed_action": action,
            "active_error": "",
            "self_correction_success_count": state.get("self_correction_success_count", 0) + correction,
            "action_history": _append_action_record(
                state,
                _make_action_record(
                    state,
                    action["action"],
                    action["args"],
                    action.get("reason", ""),
                    "planned",
                    "动作解析成功",
                ),
            ),
        }
        if action["action"] == "finish":
            updates["finish_outline"] = action["args"].get("answer_outline", "")
        return updates
    except Exception as e:
        err = f"InvalidAction: {type(e).__name__}: {e}. Please repair the next action."
        return {
            "parsed_action": {},
            "active_error": err,
            "invalid_action_count": state.get("invalid_action_count", 0) + 1,
            "action_history": _append_action_record(
                state,
                _make_action_record(state, "invalid", {}, "", "invalid", error=err),
            ),
        }


def parse_action_router(state: AgentState) -> str:
    settings = get_settings()
    if (
        state.get("step_count", 0) >= settings.max_steps
        or state.get("total_tokens", 0) >= settings.token_budget
        or state.get("invalid_action_count", 0) >= 3
    ):
        return "force_finish"
    action = (state.get("parsed_action") or {}).get("action")
    if not action:
        return "planner"
    if action == "finish":
        return "reflect"
    return "execute_action"


# ---------------------------------------------------------------- action 执行

def _tool_message_id(state: AgentState) -> str:
    action = state.get("parsed_action") or {}
    return action.get("id") or f"call_{uuid.uuid4().hex[:12]}"


def _evidence_from_source(
    sid: int,
    url: str,
    title: str,
    snippet: str,
    source_type: str,
    confidence: float = 0.0,
) -> dict:
    return {
        "id": sid,
        "url": url,
        "title": title,
        "snippet": snippet[:800],
        "key_facts": [snippet[:240].replace("\n", " ")] if snippet else [],
        "source_type": source_type,
        "confidence": confidence,
    }


async def execute_action_node(state: AgentState) -> dict:
    """执行 planner 选择的动作，登记来源、维护结构化记忆。"""
    action = state.get("parsed_action") or {}
    name = action.get("action", "")
    args = action.get("args") or {}
    reason = action.get("reason", "")
    tool_call_id = _tool_message_id(state)
    sources = dict(state.get("sources") or {})
    evidence = dict(state.get("evidence") or {})
    visited = list(state.get("visited_urls") or [])
    searched = list(state.get("searched_queries") or [])
    stagnant = state.get("stagnant_steps", 0)
    updates: dict = {}
    status = "ok"
    error = ""
    recovered = 1 if state.get("active_error") else 0

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
                        evidence[sid] = _evidence_from_source(
                            sid,
                            r["url"],
                            r["title"],
                            r["snippet"],
                            "search",
                            float(r.get("score") or 0.0),
                        )
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
                evidence[sid] = _evidence_from_source(sid, url, page["title"], page["content"], "page", 1.0)
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
            updates["ask_user_count"] = state.get("ask_user_count", 0) + 1
            stagnant = 0
        else:
            content = f"未知动作 {name}"
            status = "error"
            error = content
            updates["tool_error_count"] = state.get("tool_error_count", 0) + 1
    except Exception as e:  # 工具失败作为 observation 回喂
        status = "error"
        error = f"工具 {name} 执行失败：{type(e).__name__}: {e}。请调整参数或换一种方式。"
        content = error
        updates["active_error"] = error
        updates["tool_error_count"] = state.get("tool_error_count", 0) + 1
    else:
        updates["active_error"] = ""
        if recovered:
            updates["tool_error_recovery_count"] = state.get("tool_error_recovery_count", 0) + 1

    updates.update(
        {
            "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
            "sources": sources,
            "evidence": evidence,
            "visited_urls": visited,
            "searched_queries": searched,
            "stagnant_steps": stagnant,
            "parsed_action": {},
            "action_history": _append_action_record(
                state,
                _make_action_record(
                    state,
                    name,
                    args,
                    reason,
                    status,
                    str(content)[:500],
                    error,
                ),
            ),
        }
    )
    return updates


# Backwards-compatible names for existing tests/imports.
agent_node = planner_node
tools_node = execute_action_node
agent_router = parse_action_router


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
    updates: dict = {"phase": "REFLECTING"}

    outline = state.get("finish_outline", "")
    updates["finish_outline"] = outline

    rounds = state.get("reflect_rounds", 0)
    # 预算耗尽或反思次数用完：直接放行
    if state.get("budget_exhausted") or rounds >= settings.max_reflect_rounds:
        updates["reflect_feedback"] = ""
        return updates

    reflect_context, _ = build_reflect_context(state)
    result, raw_msg = await _structured_invoke(
        ReflectResult,
        [
            SystemMessage(content=prompts.REFLECT_SYSTEM),
            HumanMessage(content=reflect_context),
        ],
    )
    updates["total_tokens"] = state.get("total_tokens", 0) + _usage(raw_msg)
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
    answer_context, answer_context_tokens = build_answer_context(state)

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
                    evidence=answer_context,
                )
            ),
        ]
    )
    answer = str(resp.content)
    return {
        "final_answer": answer,
        "cited_sources": extract_citations(answer, sources),
        "total_tokens": state.get("total_tokens", 0) + _usage(resp),
        "answer_context_tokens": state.get("answer_context_tokens", 0) + answer_context_tokens,
        "phase": "DONE",
        "messages": [AIMessage(content=answer)],
    }
