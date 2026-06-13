"""visit_page 工具：Jina Reader 抓取网页正文，带 SSRF 防护与截断。"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

import httpx

from app.config import get_settings


class UnsafeURLError(ValueError):
    pass


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise UnsafeURLError(f"仅支持 http/https，收到: {parsed.scheme or '(空)'}")
    host = parsed.hostname
    if not host:
        raise UnsafeURLError("URL 缺少主机名")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        raise UnsafeURLError(f"无法解析主机名: {host}")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise UnsafeURLError(f"禁止访问内网/保留地址: {ip}")


async def visit_page(url: str) -> dict[str, str]:
    """返回 {url, title, content}，content 为截断后的 markdown 正文。"""
    settings = get_settings()
    _validate_url(url)

    headers = {"X-Return-Format": "markdown"}
    if settings.jina_api_key:
        headers["Authorization"] = f"Bearer {settings.jina_api_key}"

    async with httpx.AsyncClient(timeout=settings.tool_timeout_s) as client:
        resp = await client.get(f"https://r.jina.ai/{url}", headers=headers)
        resp.raise_for_status()
        text = resp.text

    # Jina Reader 文本头部通常含 "Title: xxx" 行
    title = url
    for line in text.splitlines()[:5]:
        if line.startswith("Title:"):
            title = line[len("Title:"):].strip()
            break

    limit = settings.page_content_max_chars
    truncated = len(text) > limit
    content = text[:limit]
    if truncated:
        content += "\n\n...[正文过长已截断]"
    return {"url": url, "title": title, "content": content}
