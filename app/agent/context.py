"""上下文裁剪：保留最近步骤完整内容，压缩较早的 tool observation。"""
from __future__ import annotations

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage

from app.config import get_settings

KEEP_RECENT_STEPS = 3  # 最近 N 轮 (AI tool_call + ToolMessage) 完整保留


def estimate_tokens(text: str) -> int:
    # 粗略估算：中文约 1.5 字符/token，英文约 4 字符/token，取 2.5 折中
    return int(len(text) / 2.5) + 1


def messages_token_estimate(messages: list[AnyMessage]) -> int:
    return sum(estimate_tokens(str(m.content)) for m in messages)


def trim_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
    """超过上下文预算时，把较早的 ToolMessage 压缩成一行摘要。

    保持 AIMessage(tool_calls) 与 ToolMessage 的配对结构不变，只缩短内容。
    """
    settings = get_settings()
    if messages_token_estimate(messages) <= settings.context_token_limit:
        return messages

    # 找出所有 ToolMessage 下标，最近 KEEP_RECENT_STEPS 条保留
    tool_idx = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
    protect = set(tool_idx[-KEEP_RECENT_STEPS:])

    trimmed: list[AnyMessage] = []
    for i, m in enumerate(messages):
        if isinstance(m, ToolMessage) and i not in protect and len(str(m.content)) > 400:
            head = str(m.content)[:300].replace("\n", " ")
            trimmed.append(
                ToolMessage(
                    content=f"[已压缩的早期结果摘要] {head} ...（详细内容已省略，关键信息见来源列表）",
                    tool_call_id=m.tool_call_id,
                )
            )
        elif isinstance(m, AIMessage) and m.content and len(str(m.content)) > 600 and i < (tool_idx[-KEEP_RECENT_STEPS] if protect else len(messages)):
            trimmed.append(AIMessage(content=str(m.content)[:400] + " ...", tool_calls=m.tool_calls))
        else:
            trimmed.append(m)
    return trimmed


def build_planner_context(state: dict) -> tuple[str, int]:
    """Build a compact decision context for the planner sub-agent."""
    settings = get_settings()
    remaining = settings.max_steps - int(state.get("step_count") or 0)
    lines = [
        f"用户问题：{state.get('query', '')}",
        f"用户补充：{_format_clarifications(state)}",
        f"剩余步骤：{max(remaining, 0)}",
        f"累计 token：{int(state.get('total_tokens') or 0)} / {settings.token_budget}",
        f"禁用动作：{_compact_list(state.get('disabled_actions') or [])}",
        f"已搜索：{_compact_list(state.get('searched_queries') or [])}",
        f"已访问：{_compact_list(state.get('visited_urls') or [])}",
    ]
    if state.get("reflect_feedback"):
        lines.append(f"质检反馈：{state['reflect_feedback']}")
    if state.get("active_error"):
        lines.append(f"上一步可修复错误：{state['active_error']}")
    if state.get("scratchpad_summary"):
        lines.append(f"历史摘要：{state['scratchpad_summary']}")
    lines.extend(
        [
            "",
            "最近动作：",
            _format_action_history(state.get("action_history") or [], limit=5),
            "",
            "候选文档池（C#，未剪枝）：",
            _format_candidates(state.get("candidate_docs") or {}, state.get("pruned_candidate_ids") or [], limit=12),
            "",
            "已保留证据（E#，可用于最终引用）：",
            _format_curated_evidence(state.get("curated_evidence") or {}, limit=10),
            "",
            "核验记录：",
            _format_verifications(state.get("verification_records") or [], limit=6),
            "",
            "剪枝记录：",
            _format_prune_history(state.get("prune_history") or [], limit=4),
        ]
    )
    text = "\n".join(lines)
    return text, estimate_tokens(text)


def build_reflect_context(state: dict) -> tuple[str, int]:
    lines = [
        f"用户原始问题：{state.get('query', '')}",
        f"用户补充：{_format_clarifications(state)}",
        f"回答提纲：{state.get('finish_outline') or '（无）'}",
        "",
        "已保留证据：",
        _format_curated_evidence(state.get("curated_evidence") or {}, limit=20),
        "",
        "兼容证据表：",
        _format_evidence(state.get("evidence") or {}, limit=20),
        "",
        "核验记录：",
        _format_verifications(state.get("verification_records") or [], limit=10),
        "",
        "未剪枝候选补充：",
        _format_candidates(state.get("candidate_docs") or {}, state.get("pruned_candidate_ids") or [], limit=8),
        "",
        "近期动作摘要：",
        _format_action_history(state.get("action_history") or [], limit=6),
    ]
    text = "\n".join(lines)
    return text, estimate_tokens(text)


def build_answer_context(state: dict) -> tuple[str, int]:
    lines = [
        f"用户问题：{state.get('query', '')}",
        f"用户补充：{_format_clarifications(state)}",
        f"回答提纲：{state.get('finish_outline') or '（无）'}",
        "",
        "已保留证据：",
        _format_curated_evidence(state.get("curated_evidence") or {}, limit=30),
        "",
        "兼容证据表：",
        _format_evidence(state.get("evidence") or {}, limit=30),
        "",
        "核验记录：",
        _format_verifications(state.get("verification_records") or [], limit=10),
        "",
        "未剪枝候选补充：",
        _format_candidates(state.get("candidate_docs") or {}, state.get("pruned_candidate_ids") or [], limit=10),
    ]
    text = "\n".join(lines)
    return text, estimate_tokens(text)


def _format_clarifications(state: dict) -> str:
    items = state.get("clarifications") or []
    if not items:
        return "（无）"
    return "；".join(f"{c.get('question', '')} -> {c.get('answer', '')}" for c in items)


def _format_action_history(history: list[dict], limit: int) -> str:
    if not history:
        return "（无）"
    items = history[-limit:]
    return "\n".join(
        (
            f"- step={r.get('step', '?')} action={r.get('action', '')} "
            f"status={r.get('status', '')} result={str(r.get('observation_summary') or r.get('error') or '')[:240]}"
        )
        for r in items
    )


def _format_evidence(evidence: dict[int, dict] | dict, limit: int) -> str:
    if not evidence:
        return "（暂无证据）"
    values = list(evidence.values()) if isinstance(evidence, dict) else list(evidence)
    lines = []
    for e in values[:limit]:
        facts = "；".join(e.get("key_facts") or [])
        snippet = str(e.get("snippet") or "")[:500].replace("\n", " ")
        lines.append(
            f"[{e.get('id')}] {e.get('title')} — {e.get('url')}\n"
            f"类型：{e.get('source_type', 'source')}；摘要：{snippet}\n"
            f"关键事实：{facts or '（未抽取）'}"
        )
    return "\n\n".join(lines)


def _format_candidates(candidates: dict[int, dict] | dict, pruned_ids: list[int], limit: int) -> str:
    if not candidates:
        return "（暂无候选）"
    pruned = {int(x) for x in pruned_ids}
    values = list(candidates.values()) if isinstance(candidates, dict) else list(candidates)
    active = [c for c in values if int(c.get("id", 0)) not in pruned and c.get("status") != "pruned"]
    if not active:
        return "（无未剪枝候选）"
    lines = []
    for c in active[:limit]:
        text = str(c.get("content") or c.get("snippet") or "")[:600].replace("\n", " ")
        lines.append(
            f"[C{c.get('id')}] {c.get('title')} — {c.get('url')}\n"
            f"状态：{c.get('status', 'candidate')}；来源：{c.get('source_type', 'source')}；"
            f"token≈{c.get('token_estimate', 0)}；found_by={c.get('found_by_query') or '（无）'}\n"
            f"摘要：{text}"
        )
    return "\n\n".join(lines)


def _format_curated_evidence(curated: dict[int, dict] | dict, limit: int) -> str:
    if not curated:
        return "（暂无已保留证据）"
    values = list(curated.values()) if isinstance(curated, dict) else list(curated)
    lines = []
    for e in values[:limit]:
        source_id = e.get("source_id")
        cite = f"[{source_id}]" if source_id else "（未登记来源）"
        lines.append(
            f"[E{e.get('id')}] C{e.get('candidate_id')} -> {cite} {e.get('title')} — {e.get('url')}\n"
            f"claim：{e.get('claim')}\n"
            f"证据：{str(e.get('quote_or_summary') or '')[:500]}\n"
            f"confidence：{e.get('confidence')}"
        )
    return "\n\n".join(lines)


def _format_verifications(records: list[dict], limit: int) -> str:
    if not records:
        return "（暂无核验记录）"
    items = records[-limit:]
    return "\n".join(
        (
            f"- verdict={r.get('verdict')} sources={r.get('source_ids')} "
            f"claim={str(r.get('claim') or '')[:160]} "
            f"reason={str(r.get('reasoning') or r.get('missing_or_conflict') or '')[:220]}"
        )
        for r in items
    )


def _format_prune_history(history: list[dict], limit: int) -> str:
    if not history:
        return "（暂无剪枝）"
    items = history[-limit:]
    return "\n".join(
        f"- step={r.get('step')} candidates={r.get('candidate_ids')} reason={str(r.get('reason') or '')[:180]}"
        for r in items
    )


def _compact_list(items: list, limit: int = 8) -> str:
    if not items:
        return "（无）"
    shown = [str(x) for x in items[-limit:]]
    prefix = f"…共 {len(items)} 项；最近：" if len(items) > limit else ""
    return prefix + " | ".join(shown)
