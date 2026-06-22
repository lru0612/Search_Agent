"""StateGraph 构建。"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agent.nodes import (
    answer_node,
    ask_clarify_node,
    clarify_node,
    clarify_router,
    execute_action_node,
    force_finish_node,
    parse_action_node,
    parse_action_router,
    planner_node,
    reflect_node,
    reflect_router,
    rewrite_node,
)
from app.agent.state import AgentState

checkpointer = MemorySaver()


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("clarify", clarify_node)
    g.add_node("ask_clarify", ask_clarify_node)
    g.add_node("rewrite", rewrite_node)
    g.add_node("planner", planner_node)
    g.add_node("parse_action", parse_action_node)
    g.add_node("execute_action", execute_action_node)
    g.add_node("force_finish", force_finish_node)
    g.add_node("reflect", reflect_node)
    g.add_node("answer", answer_node)

    g.add_edge(START, "clarify")
    g.add_conditional_edges("clarify", clarify_router, {"ask_clarify": "ask_clarify", "rewrite": "rewrite"})
    g.add_edge("ask_clarify", "clarify")
    g.add_edge("rewrite", "planner")
    g.add_edge("planner", "parse_action")
    g.add_conditional_edges(
        "parse_action",
        parse_action_router,
        {
            "planner": "planner",
            "execute_action": "execute_action",
            "reflect": "reflect",
            "force_finish": "force_finish",
        },
    )
    g.add_edge("execute_action", "planner")
    g.add_edge("force_finish", "reflect")
    g.add_conditional_edges("reflect", reflect_router, {"agent": "planner", "answer": "answer"})
    g.add_edge("answer", END)

    return g.compile(checkpointer=checkpointer)


graph = build_graph()
