"""web_search 工具：Tavily 检索。"""
from __future__ import annotations

import asyncio
from typing import Any

from tavily import TavilyClient

from app.config import get_settings


def _search_sync(query: str, max_results: int) -> list[dict[str, Any]]:
    client = TavilyClient(api_key=get_settings().tavily_api_key)
    resp = client.search(query=query, max_results=max_results, search_depth="basic")
    results = []
    for r in resp.get("results", []):
        results.append(
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": (r.get("content") or "")[:600],
                "score": r.get("score", 0.0),
            }
        )
    return results


async def web_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    settings = get_settings()
    return await asyncio.wait_for(
        asyncio.to_thread(_search_sync, query, max_results),
        timeout=settings.tool_timeout_s,
    )
