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
from app.tools.schemas import (
    AskUser,
    CurateEvidence,
    Finish,
    PruneCandidates,
    TOOL_NAME_MAP,
    TOOL_SCHEMAS,
    VerifyClaim,
    VisitPage,
    WebSearch,
)
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
    if state.get("total_tokens", 0) >= int(settings.token_budget * 0.8):
        extra += "\n注意：token 预算接近耗尽，请优先剪枝、保留证据、核验证据或 Finish，避免继续扩大搜索。"

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
                "schema: {\"action\":\"web_search|visit_page|ask_user|curate_evidence|prune_candidates|verify_claim|finish\", \"args\":{...}, \"reason\": string}\n"
                "参数要求：\n"
                "- web_search: {\"query\": string, \"max_results\": number}\n"
                "- visit_page: {\"url\": string, \"reason\": string}\n"
                "- ask_user: {\"question\": string, \"options\": string[]}\n"
                "- curate_evidence: {\"candidate_id\": number, \"claim\": string, \"quote_or_summary\": string, \"confidence\": number}\n"
                "- prune_candidates: {\"candidate_ids\": number[], \"reason\": string}\n"
                "- verify_claim: {\"claim\": string, \"source_ids\": number[]}\n"
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


def _validate_action(
    action: str,
    args: dict,
    disabled_actions: list[str] | None = None,
    state: AgentState | None = None,
) -> None:
    if action in set(disabled_actions or []):
        raise ValueError(f"action disabled in this run: {action}")
    schema_by_action = {
        "web_search": WebSearch,
        "visit_page": VisitPage,
        "ask_user": AskUser,
        "curate_evidence": CurateEvidence,
        "prune_candidates": PruneCandidates,
        "verify_claim": VerifyClaim,
        "finish": Finish,
    }
    schema = schema_by_action.get(action)
    if not schema:
        raise ValueError(f"unknown action: {action}")
    schema.model_validate(args)
    if state and action in {"web_search", "visit_page"}:
        settings = get_settings()
        if int(state.get("total_tokens") or 0) >= int(settings.token_budget * 0.95):
            raise ValueError(
                "token budget is nearly exhausted; curate/prune/verify existing evidence or finish"
            )


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
        action_source = "tool_call" if tool_calls else "json"
        action = _action_from_tool_call(tool_calls[0]) if tool_calls else _action_from_json(str(last.content))
        _validate_action(action["action"], action["args"], state.get("disabled_actions") or [], state)
        correction = 1 if state.get("active_error") else 0
        updates: dict = {
            "parsed_action": action,
            "action_source": action_source,
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


class VerifyResult(BaseModel):
    verdict: str = Field(description="supported / contradicted / insufficient / mixed")
    reasoning: str = Field(description="核验理由")
    missing_or_conflict: str = Field(default="", description="缺失证据或冲突点")


def _next_id(items: dict[int, dict]) -> int:
    return max((int(k) for k in items.keys()), default=0) + 1


def _int_keyed(items: dict | None) -> dict[int, dict]:
    return {int(k): v for k, v in (items or {}).items()}


def _candidate_text(candidate: dict) -> str:
    return str(candidate.get("content") or candidate.get("snippet") or "")


def _candidate_snippet(candidate: dict, limit: int = 800) -> str:
    text = _candidate_text(candidate)
    return text[:limit].replace("\n", " ")


def _register_candidate(
    candidate_docs: dict[int, dict],
    *,
    url: str,
    title: str,
    snippet: str,
    source_type: str,
    state: AgentState,
    found_by_query: str = "",
    content: str = "",
    confidence: float = 0.0,
) -> tuple[dict[int, dict], int, bool]:
    """Register or update a recoverable search candidate, deduped by URL."""
    for cid, existing in candidate_docs.items():
        if existing.get("url") != url:
            continue
        updated = dict(existing)
        if title and (not updated.get("title") or len(title) > len(str(updated.get("title")))):
            updated["title"] = title
        if snippet and len(snippet) > len(str(updated.get("snippet") or "")):
            updated["snippet"] = snippet[:1200]
        if content and len(content) > len(str(updated.get("content") or "")):
            updated["content"] = content
        if source_type == "page" and updated.get("status") != "pruned":
            updated["status"] = "read"
        updated["token_estimate"] = estimate_tokens(_candidate_text(updated))
        if confidence:
            updated["confidence"] = max(float(updated.get("confidence") or 0.0), confidence)
        return {**candidate_docs, int(cid): updated}, int(cid), False

    cid = _next_id(candidate_docs)
    text = content or snippet
    candidate = {
        "id": cid,
        "url": url,
        "title": title or url,
        "snippet": snippet[:1200],
        "content": content,
        "key_facts": [text[:240].replace("\n", " ")] if text else [],
        "source_type": source_type,
        "status": "read" if source_type == "page" else "candidate",
        "found_by_step": state.get("step_count", 0),
        "found_by_query": found_by_query,
        "token_estimate": estimate_tokens(text),
        "confidence": confidence,
    }
    return {**candidate_docs, cid: candidate}, cid, True


def _candidate_to_evidence(sid: int, candidate: dict, claim: str, quote_or_summary: str, confidence: float) -> dict:
    snippet = quote_or_summary or _candidate_snippet(candidate)
    facts = []
    if claim:
        facts.append(claim)
    if quote_or_summary and quote_or_summary != claim:
        facts.append(quote_or_summary[:240].replace("\n", " "))
    return {
        "id": sid,
        "url": candidate.get("url", ""),
        "title": candidate.get("title", candidate.get("url", "")),
        "snippet": snippet[:800],
        "key_facts": facts or [snippet[:240].replace("\n", " ")],
        "source_type": candidate.get("source_type", "candidate"),
        "confidence": confidence,
    }


def _resolve_verify_sources(source_ids: list[int], candidate_docs: dict[int, dict], sources: dict[int, dict]) -> str:
    lines: list[str] = []
    seen_candidates: set[int] = set()
    for raw_id in source_ids:
        sid = int(raw_id)
        candidate = candidate_docs.get(sid)
        if not candidate:
            for c in candidate_docs.values():
                if int(c.get("source_id") or -1) == sid:
                    candidate = c
                    break
        if candidate and int(candidate.get("id", sid)) not in seen_candidates:
            seen_candidates.add(int(candidate.get("id", sid)))
            lines.append(
                f"[C{candidate.get('id')}] {candidate.get('title')} — {candidate.get('url')}\n"
                f"{_candidate_text(candidate)[:4000]}"
            )
            continue
        source = sources.get(sid)
        if source:
            lines.append(f"[{sid}] {source.get('title')} — {source.get('url')}\n{source.get('snippet', '')}")
    return "\n\n".join(lines)


async def _verify_claim_with_llm(
    claim: str,
    source_ids: list[int],
    candidate_docs: dict[int, dict],
    sources: dict[int, dict],
) -> tuple[dict, AIMessage]:
    context = _resolve_verify_sources(source_ids, candidate_docs, sources)
    if not context:
        raise ValueError(f"no matching candidate/source ids for verification: {source_ids}")
    result, raw_msg = await _structured_invoke(
        VerifyResult,
        [
            SystemMessage(
                content=(
                    "你是证据核验模块。只根据给定来源判断 claim 是否被支持。"
                    "verdict 只能使用 supported、contradicted、insufficient 或 mixed。"
                )
            ),
            HumanMessage(content=f"Claim: {claim}\n\nSources:\n{context}"),
        ],
    )
    return {
        "claim": claim,
        "source_ids": source_ids,
        "verdict": result.verdict,
        "reasoning": result.reasoning,
        "missing_or_conflict": result.missing_or_conflict,
    }, raw_msg


def _ensure_fallback_answer_sources(state: AgentState) -> tuple[dict[int, dict], dict[int, dict], dict[int, dict]]:
    """If no evidence was curated, expose active candidates as normal sources for legacy answer flow."""
    sources = _int_keyed(state.get("sources") or {})
    evidence = _int_keyed(state.get("evidence") or {})
    candidate_docs = _int_keyed(state.get("candidate_docs") or {})
    if sources or evidence:
        return sources, evidence, candidate_docs

    pruned = set(state.get("pruned_candidate_ids") or [])
    active_candidates = [
        c for c in candidate_docs.values() if int(c.get("id", 0)) not in pruned and c.get("status") != "pruned"
    ][:10]
    for candidate in active_candidates:
        snippet = _candidate_snippet(candidate)
        sources, sid = register_source(
            sources,
            str(candidate.get("url") or ""),
            str(candidate.get("title") or candidate.get("url") or ""),
            snippet,
        )
        candidate_docs[int(candidate["id"])] = {**candidate, "source_id": sid}
        evidence[sid] = _evidence_from_source(
            sid,
            str(candidate.get("url") or ""),
            str(candidate.get("title") or candidate.get("url") or ""),
            snippet,
            str(candidate.get("source_type") or "candidate"),
            float(candidate.get("confidence") or 0.0),
        )
    return sources, evidence, candidate_docs


async def execute_action_node(state: AgentState) -> dict:
    """执行 planner 选择的动作，登记来源、维护结构化记忆。"""
    action = state.get("parsed_action") or {}
    name = action.get("action", "")
    args = action.get("args") or {}
    reason = action.get("reason", "")
    tool_call_id = _tool_message_id(state)
    sources = _int_keyed(state.get("sources") or {})
    evidence = _int_keyed(state.get("evidence") or {})
    candidate_docs = _int_keyed(state.get("candidate_docs") or {})
    curated_evidence = _int_keyed(state.get("curated_evidence") or {})
    verification_records = list(state.get("verification_records") or [])
    pruned_candidate_ids = list(state.get("pruned_candidate_ids") or [])
    prune_history = list(state.get("prune_history") or [])
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
                    new_count = 0
                    for r in results:
                        candidate_docs, cid, created = _register_candidate(
                            candidate_docs,
                            url=r["url"],
                            title=r["title"],
                            snippet=r["snippet"],
                            source_type="search",
                            state=state,
                            found_by_query=q,
                            confidence=float(r.get("score") or 0.0),
                        )
                        new_count += 1 if created else 0
                        lines.append(f"[C{cid}] {r['title']}\nURL: {r['url']}\n摘要: {r['snippet']}")
                    content = (
                        f"搜索「{q}」返回 {len(results)} 条结果，新增候选 {new_count} 条：\n\n"
                        + "\n\n".join(lines)
                    )
                    stagnant = 0 if new_count else stagnant + 1

        elif name == "visit_page":
            url = args["url"]
            if url in visited:
                content = f"页面 {url} 已访问过，内容见之前的记录。"
                stagnant += 1
            else:
                page = await visit_page(url)
                visited.append(url)
                candidate_docs, cid, created = _register_candidate(
                    candidate_docs,
                    url=url,
                    title=page["title"],
                    snippet=page["content"][:1200],
                    content=page["content"],
                    source_type="page",
                    state=state,
                    found_by_query=str(args.get("reason") or ""),
                    confidence=1.0,
                )
                content = f"[C{cid}] 页面《{page['title']}》正文：\n\n{page['content']}"
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
        elif name == "curate_evidence":
            cid = int(args["candidate_id"])
            candidate = candidate_docs.get(cid)
            if not candidate:
                raise ValueError(f"candidate_id not found: {cid}")
            if cid in set(pruned_candidate_ids) or candidate.get("status") == "pruned":
                raise ValueError(f"candidate_id was pruned: {cid}")

            claim = str(args["claim"])
            quote_or_summary = str(args["quote_or_summary"])
            confidence = float(args.get("confidence", 0.8))
            sources, sid = register_source(
                sources,
                str(candidate.get("url") or ""),
                str(candidate.get("title") or candidate.get("url") or ""),
                quote_or_summary or _candidate_snippet(candidate),
            )
            candidate_docs[cid] = {**candidate, "status": "curated", "source_id": sid}

            existing = evidence.get(sid)
            item = _candidate_to_evidence(sid, candidate_docs[cid], claim, quote_or_summary, confidence)
            if existing:
                facts = list(existing.get("key_facts") or [])
                for fact in item.get("key_facts") or []:
                    if fact and fact not in facts:
                        facts.append(fact)
                item = {**existing, **item, "key_facts": facts}
            evidence[sid] = item

            eid = _next_id(curated_evidence)
            curated_evidence[eid] = {
                "id": eid,
                "candidate_id": cid,
                "source_id": sid,
                "url": candidate_docs[cid].get("url", ""),
                "title": candidate_docs[cid].get("title", ""),
                "claim": claim,
                "quote_or_summary": quote_or_summary,
                "confidence": confidence,
            }
            content = f"已保留证据 E{eid}：候选 C{cid} -> 来源 [{sid}]；claim: {claim}"
            stagnant = 0
        elif name == "prune_candidates":
            ids = [int(x) for x in args["candidate_ids"]]
            missing = [cid for cid in ids if cid not in candidate_docs]
            if missing:
                raise ValueError(f"candidate_ids not found: {missing}")
            reason_text = str(args["reason"])
            pruned_set = set(pruned_candidate_ids)
            for cid in ids:
                candidate_docs[cid] = {**candidate_docs[cid], "status": "pruned"}
                pruned_set.add(cid)
            pruned_candidate_ids = sorted(pruned_set)
            prune_history.append({"step": state.get("step_count", 0), "candidate_ids": ids, "reason": reason_text})
            prune_history = prune_history[-30:]
            content = f"已剪枝候选：{', '.join(f'C{cid}' for cid in ids)}。理由：{reason_text}"
            stagnant = 0
        elif name == "verify_claim":
            source_ids = [int(x) for x in args["source_ids"]]
            record, raw_msg = await _verify_claim_with_llm(
                str(args["claim"]),
                source_ids,
                candidate_docs,
                sources,
            )
            verification_records.append(record)
            verification_records = verification_records[-30:]
            updates["total_tokens"] = state.get("total_tokens", 0) + _usage(raw_msg)
            content = (
                f"核验结论：{record['verdict']}\n"
                f"Claim: {record['claim']}\n"
                f"理由：{record['reasoning']}\n"
                f"缺口/冲突：{record.get('missing_or_conflict') or '无'}"
            )
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
            "candidate_docs": candidate_docs,
            "curated_evidence": curated_evidence,
            "verification_records": verification_records,
            "pruned_candidate_ids": pruned_candidate_ids,
            "prune_history": prune_history,
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
    sources, evidence, candidate_docs = _ensure_fallback_answer_sources(state)
    answer_state = {
        **state,
        "sources": sources,
        "evidence": evidence,
        "candidate_docs": candidate_docs,
    }
    answer_context, answer_context_tokens = build_answer_context(answer_state)

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
        "sources": sources,
        "evidence": evidence,
        "candidate_docs": candidate_docs,
        "total_tokens": state.get("total_tokens", 0) + _usage(resp),
        "answer_context_tokens": state.get("answer_context_tokens", 0) + answer_context_tokens,
        "phase": "DONE",
        "messages": [AIMessage(content=answer)],
    }
