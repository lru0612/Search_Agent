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
