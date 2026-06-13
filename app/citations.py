"""来源注册表与 [n] 引用解析。

来源以编号字典形式保存在 AgentState.sources 中（LangGraph 状态可序列化），
本模块提供纯函数操作。
"""
from __future__ import annotations

import re
from typing import TypedDict


class Source(TypedDict):
    id: int
    url: str
    title: str
    snippet: str  # 支撑片段：搜索 snippet 或正文摘录


def register_source(
    sources: dict[int, Source], url: str, title: str, snippet: str
) -> tuple[dict[int, Source], int]:
    """登记来源，按 URL 去重，返回 (新 sources, 编号)。"""
    for sid, s in sources.items():
        if s["url"] == url:
            # 已有来源，若新片段更长则更新支撑片段
            if len(snippet) > len(s["snippet"]):
                sources = {**sources, sid: {**s, "snippet": snippet[:500]}}
            return sources, sid
    sid = max(sources.keys(), default=0) + 1
    new = {**sources, sid: Source(id=sid, url=url, title=title, snippet=snippet[:500])}
    return new, sid


_CITE_RE = re.compile(r"\[(\d{1,3})\]")


def extract_citations(answer: str, sources: dict[int, Source]) -> list[Source]:
    """解析回答中出现的 [n]，返回被实际引用的来源列表（按首次出现排序）。"""
    seen: list[int] = []
    for m in _CITE_RE.finditer(answer):
        n = int(m.group(1))
        if n in sources and n not in seen:
            seen.append(n)
    return [sources[n] for n in seen]


def format_sources_for_prompt(sources: dict[int, Source]) -> str:
    if not sources:
        return "（暂无来源）"
    lines = [f"[{s['id']}] {s['title']} — {s['url']}" for s in sources.values()]
    return "\n".join(lines)
