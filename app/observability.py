"""Tracer：把 LangGraph 执行过程转为结构化事件，SSE 推送 + JSONL 落盘。"""
from __future__ import annotations

import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any

from app.config import get_settings

logger = logging.getLogger("agentic_search")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

# LangChain structured output (include_raw=True) 的已知序列化噪音，不影响结果
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")


class Tracer:
    """每个会话一个实例。emit() 产出事件 dict（供 SSE 下发），同时写 JSONL。"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.started_at = time.time()
        settings = get_settings()
        traces_dir = Path(settings.traces_dir)
        traces_dir.mkdir(parents=True, exist_ok=True)
        self._file = traces_dir / f"{session_id}.jsonl"
        self.counters: dict[str, int] = {}

    def emit(self, event_type: str, **data: Any) -> dict:
        event = {
            "type": event_type,
            "session_id": self.session_id,
            "ts": round(time.time() - self.started_at, 3),
            **data,
        }
        self.counters[event_type] = self.counters.get(event_type, 0) + 1
        try:
            with self._file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        except OSError:
            logger.exception("trace 落盘失败")
        logger.info("[%s] %s %s", self.session_id[:8], event_type, _brief(data))
        return event

    def summary(self) -> dict:
        return {
            "elapsed_s": round(time.time() - self.started_at, 1),
            "event_counts": self.counters,
        }


def _brief(data: dict, limit: int = 200) -> str:
    s = json.dumps(data, ensure_ascii=False, default=str)
    return s[:limit] + ("…" if len(s) > limit else "")
