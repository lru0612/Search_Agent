"""LangGraph AgentState 定义。"""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from app.citations import Source


def _replace(_old: Any, new: Any) -> Any:
    return new


class AgentState(TypedDict, total=False):
    # 对话与任务
    messages: Annotated[list[AnyMessage], add_messages]  # ReAct 循环消息
    query: str  # 用户原始 query
    clarifications: list[dict[str, str]]  # [{question, answer}]
    clarify_rounds: int
    pending_question: dict  # 待向用户提出的澄清问题 {question, options, reasons}
    rewritten_queries: list[str]

    # 证据与来源
    sources: Annotated[dict[int, Source], _replace]
    visited_urls: Annotated[list[str], _replace]
    searched_queries: Annotated[list[str], _replace]

    # 预算与生命周期
    step_count: int
    total_tokens: int
    stagnant_steps: int  # 连续无新信息步数
    phase: str  # CLARIFYING / SEARCHING / REFLECTING / ANSWERING / DONE
    budget_exhausted: bool

    # 反思
    reflect_rounds: int
    reflect_feedback: str  # 反思指出的缺口，回喂 agent

    # 输出
    finish_outline: str
    final_answer: str
    cited_sources: list[Source]
