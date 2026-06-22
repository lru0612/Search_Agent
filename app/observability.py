"""Tracer：把 LangGraph 执行过程转为结构化事件，SSE 推送 + JSONL 落盘。"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any

from app.config import get_settings

logger = logging.getLogger("agentic_search")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)

# LangChain structured output (include_raw=True) 的已知序列化噪音，不影响结果
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")


_SAFE_PATH_PART = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_path_part(value: str) -> str:
    cleaned = _SAFE_PATH_PART.sub("_", value.strip())
    return cleaned.strip("._") or "default"


class Tracer:
    """每个会话一个实例。emit() 产出事件 dict（供 SSE 下发），同时写 JSONL。"""

    def __init__(self, session_id: str, trace_group: str | None = None, run_id: str | None = None):
        self.session_id = session_id
        self.trace_group = trace_group
        self.run_id = run_id
        self.started_at = time.time()
        settings = get_settings()
        traces_dir = Path(settings.traces_dir)
        if trace_group:
            traces_dir = traces_dir / _safe_path_part(trace_group)
            if run_id:
                traces_dir = traces_dir / _safe_path_part(run_id)
        traces_dir.mkdir(parents=True, exist_ok=True)
        self._file = traces_dir / f"{_safe_path_part(session_id)}.jsonl"
        self.counters: dict[str, int] = {}

    def emit(self, event_type: str, **data: Any) -> dict:
        event = {
            "type": event_type,
            "session_id": self.session_id,
            "ts": round(time.time() - self.started_at, 3),
            **data,
        }
        if self.trace_group:
            event["trace_group"] = self.trace_group
        if self.run_id:
            event["run_id"] = self.run_id
        self.counters[event_type] = self.counters.get(event_type, 0) + 1
        try:
            with self._file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        except OSError:
            logger.exception("trace 落盘失败")
        logger.info(
            "[%s%s] %s %s",
            self.session_id[:8],
            f" {self.trace_group}/{self.run_id or '-'}" if self.trace_group else "",
            event_type,
            _brief(data),
        )
        return event

    def summary(self) -> dict:
        return {
            "elapsed_s": round(time.time() - self.started_at, 1),
            "event_counts": self.counters,
            "trace_file": str(self._file),
        }


def _brief(data: dict, limit: int = 200) -> str:
    s = json.dumps(data, ensure_ascii=False, default=str)
    return s[:limit] + ("…" if len(s) > limit else "")
