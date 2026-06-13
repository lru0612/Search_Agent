"""无 API key 的端到端流程测试：mock LLM 与搜索工具，验证图接线与 interrupt 恢复。

运行：python -m tests.test_flow
"""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langgraph.types import Command

from app.agent import nodes


_structured_idx: dict[type, int] = {}


class FakeStructuredLLM:
    """同一 schema 的多次调用共享消费进度（跨 with_structured_output 实例）。"""

    def __init__(self, schema: type, results: list):
        self.schema = schema
        self.results = results

    async def ainvoke(self, _msgs):
        i = _structured_idx.get(self.schema, 0)
        r = self.results[min(i, len(self.results) - 1)]
        _structured_idx[self.schema] = i + 1
        return {"parsed": r, "raw": AIMessage(content="", usage_metadata={"total_tokens": 10, "input_tokens": 5, "output_tokens": 5})}


class FakeChat:
    """按脚本依次返回回复的假 ChatOpenAI。"""

    script: list[AIMessage] = []
    structured_scripts: dict[type, list] = {}
    _idx = 0

    def __init__(self, *a, **kw):
        pass

    def with_structured_output(self, schema, include_raw=True):
        return FakeStructuredLLM(schema, FakeChat.structured_scripts[schema])

    def bind_tools(self, tools, tool_choice=None):
        return self

    async def ainvoke(self, _msgs):
        msg = FakeChat.script[FakeChat._idx]
        FakeChat._idx += 1
        return msg


async def fake_web_search(query: str, max_results: int = 5):
    return [
        {"title": f"结果A about {query}", "url": "https://example.com/a", "snippet": "苹果 2026 春季发布会发布了新 MacBook。", "score": 0.9},
        {"title": "结果B", "url": "https://example.com/b", "snippet": "发布会日期为 2026 年 3 月。", "score": 0.8},
    ]


async def fake_visit_page(url: str):
    return {"url": url, "title": "示例页面", "content": "详细内容：新 MacBook 搭载 M5 芯片，售价 1299 美元起。"}


def make_tool_call(name: str, args: dict) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": f"call_{name}_{FakeChat._idx}", "type": "tool_call"}],
        usage_metadata={"total_tokens": 20, "input_tokens": 10, "output_tokens": 10},
    )


async def main() -> int:
    # 脚本：clarify 判定歧义 → 用户补充 → 判定清晰 → rewrite → WebSearch → VisitPage → Finish → reflect 通过 → answer
    FakeChat.structured_scripts = {
        nodes.ClarifyResult: [
            nodes.ClarifyResult(is_ambiguous=True, reasons=["缺少时间"], clarifying_question="你指哪一年的发布会？", options=["2025", "2026"]),
            nodes.ClarifyResult(is_ambiguous=False),
        ],
        nodes.RewriteResult: [nodes.RewriteResult(queries=["苹果 2026 春季发布会 新品"])],
        nodes.ReflectResult: [nodes.ReflectResult(sufficient=True, reasoning="证据充分")],
    }
    FakeChat.script = [
        make_tool_call("WebSearch", {"query": "苹果 2026 春季发布会 新品", "max_results": 5}),
        make_tool_call("VisitPage", {"url": "https://example.com/a", "reason": "查看详情"}),
        make_tool_call("Finish", {"answer_outline": "发布会内容 [1][2]"}),
        AIMessage(content="苹果 2026 春季发布会发布了搭载 M5 芯片的新 MacBook [1]，发布会在 2026 年 3 月举行 [2]。", usage_metadata={"total_tokens": 30, "input_tokens": 15, "output_tokens": 15}),
    ]

    with patch.object(nodes, "ChatOpenAI", FakeChat), \
         patch.object(nodes, "web_search", fake_web_search), \
         patch.object(nodes, "visit_page", fake_visit_page):
        from app.agent.graph import build_graph

        graph = build_graph()
        config = {"configurable": {"thread_id": "test-1"}}
        init = {
            "query": "苹果发布会有什么新品", "clarifications": [], "clarify_rounds": 0,
            "step_count": 0, "total_tokens": 0, "stagnant_steps": 0,
            "reflect_rounds": 0, "budget_exhausted": False, "phase": "CLARIFYING",
        }

        # 第一段：应在 clarify 处 interrupt
        interrupted = False
        async for chunk in graph.astream(init, config=config, stream_mode="updates"):
            if "__interrupt__" in chunk:
                payload = chunk["__interrupt__"][0].value
                assert payload["type"] == "ask_user", payload
                assert "哪一年" in payload["question"]
                interrupted = True
        assert interrupted, "应触发 ask_user interrupt"
        print("PASS: clarify interrupt 触发，问题 =", payload["question"])

        # 第二段：恢复并跑完
        nodes_visited = []
        async for chunk in graph.astream(Command(resume="2026年"), config=config, stream_mode="updates"):
            nodes_visited += [k for k in chunk if k != "__interrupt__"]

        state = graph.get_state(config).values
        assert state["phase"] == "DONE", state["phase"]
        assert "M5" in state["final_answer"]
        assert state["clarifications"][0]["answer"] == "2026年"
        # 2 个搜索结果；访问页与结果 A 同 URL，去重后复用编号 1
        assert len(state["sources"]) == 2, state["sources"]
        cited_ids = [c["id"] for c in state["cited_sources"]]
        assert cited_ids == [1, 2], cited_ids
        assert state["total_tokens"] > 0
        print("PASS: 节点路径 =", " → ".join(nodes_visited))
        print("PASS: 来源数 =", len(state["sources"]), "| 引用 =", cited_ids)
        print("PASS: 最终回答 =", state["final_answer"])
        print("\n全部断言通过 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
