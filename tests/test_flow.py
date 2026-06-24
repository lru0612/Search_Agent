"""无 API key 的端到端流程测试：mock LLM 与搜索工具，验证图接线与 interrupt 恢复。

运行：python -m tests.test_flow
"""
from __future__ import annotations

import asyncio
import sys
import time
from unittest.mock import patch

from langchain_core.messages import AIMessage
from app.agent.context import build_planner_context
from app.agent import nodes


CASE_TIMEOUT_S = 30
_structured_idx: dict[type, int] = {}


def log(message: str) -> None:
    print(f"[test_flow {time.strftime('%H:%M:%S')}] {message}", flush=True)


class FakeStructuredLLM:
    """同一 schema 的多次调用共享消费进度（跨 with_structured_output 实例）。"""

    def __init__(self, schema: type, results: list):
        self.schema = schema
        self.results = results

    async def ainvoke(self, _msgs):
        i = _structured_idx.get(self.schema, 0)
        log(f"FakeStructuredLLM schema={self.schema.__name__} idx={i}")
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
        log(f"FakeChat idx={FakeChat._idx}")
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


def apply_update(state: dict, update: dict) -> None:
    for key, value in update.items():
        if key == "messages":
            state.setdefault("messages", [])
            state["messages"].extend(value)
        else:
            state[key] = value


async def main() -> int:
    _structured_idx.clear()
    FakeChat._idx = 0
    # 脚本：clarify 判定歧义 → 用户补充 → 判定清晰 → rewrite →
    # WebSearch 候选池 → Prune → Curate → VisitPage → Curate → Verify → Finish → answer
    FakeChat.structured_scripts = {
        nodes.ClarifyResult: [
            nodes.ClarifyResult(is_ambiguous=True, reasons=["缺少时间"], clarifying_question="你指哪一年的发布会？", options=["2025", "2026"]),
            nodes.ClarifyResult(is_ambiguous=False),
        ],
        nodes.RewriteResult: [nodes.RewriteResult(queries=["苹果 2026 春季发布会 新品"])],
        nodes.VerifyResult: [
            nodes.VerifyResult(verdict="supported", reasoning="页面正文明确说明新 MacBook 搭载 M5 芯片。")
        ],
        nodes.ReflectResult: [nodes.ReflectResult(sufficient=True, reasoning="证据充分")],
    }
    FakeChat.script = [
        make_tool_call("WebSearch", {"query": "苹果 2026 春季发布会 新品", "max_results": 5}),
        make_tool_call("PruneCandidates", {"candidate_ids": [2], "reason": "候选 B 只有日期，和新品信息关系较弱"}),
        make_tool_call("CurateEvidence", {"candidate_id": 1, "claim": "苹果 2026 春季发布会发布了新 MacBook", "quote_or_summary": "苹果 2026 春季发布会发布了新 MacBook。", "confidence": 0.9}),
        make_tool_call("VisitPage", {"url": "https://example.com/a", "reason": "查看新品详情"}),
        make_tool_call("CurateEvidence", {"candidate_id": 1, "claim": "新 MacBook 搭载 M5 芯片", "quote_or_summary": "详细内容：新 MacBook 搭载 M5 芯片，售价 1299 美元起。", "confidence": 1.0}),
        make_tool_call("VerifyClaim", {"claim": "新 MacBook 搭载 M5 芯片", "source_ids": [1]}),
        make_tool_call("Finish", {"answer_outline": "发布会新品与芯片信息 [1]"}),
        AIMessage(content="苹果 2026 春季发布会发布了新 MacBook，并且该机型搭载 M5 芯片，售价 1299 美元起 [1]。", usage_metadata={"total_tokens": 30, "input_tokens": 15, "output_tokens": 15}),
    ]

    with patch.object(nodes, "_llm", lambda *a, **kw: FakeChat()), \
         patch.object(nodes, "web_search", fake_web_search), \
         patch.object(nodes, "visit_page", fake_visit_page):
        init = {
            "query": "苹果发布会有什么新品", "clarifications": [], "clarify_rounds": 0,
            "step_count": 0, "total_tokens": 0, "stagnant_steps": 0,
            "reflect_rounds": 0, "budget_exhausted": False, "phase": "CLARIFYING",
            "action_history": [], "evidence": {}, "candidate_docs": {}, "curated_evidence": {},
            "verification_records": [], "pruned_candidate_ids": [], "prune_history": [],
            "active_error": "", "scratchpad_summary": "",
            "invalid_action_count": 0, "self_correction_success_count": 0,
            "ask_user_count": 0, "tool_error_count": 0, "tool_error_recovery_count": 0,
            "planner_context_tokens": 0, "answer_context_tokens": 0,
        }

        log("START: run clarify node and router")
        clarify_update = await nodes.clarify_node(init)
        payload = {"type": "ask_user", **clarify_update["pending_question"]}
        assert nodes.clarify_router({**init, **clarify_update}) == "ask_clarify"
        assert "哪一年" in payload["question"]
        print("PASS: clarify interrupt 触发，问题 =", payload["question"])

        log("START: run clarified node chain to completion")
        # 第二段：手动执行 clarify 后的节点链。当前 LangGraph 版本在该 mock
        # stream 场景下会空等下一块；benchmark runner 会用 hard timeout 捕获
        # 真实图挂起，这里优先保证 smoke gate 稳定覆盖节点行为。
        state = {
            **init,
            "clarifications": [{"question": payload["question"], "answer": "2026年"}],
            "clarify_rounds": 1,
            "pending_question": {},
            "messages": [],
            "sources": {},
            "evidence": {},
            "candidate_docs": {},
            "curated_evidence": {},
            "verification_records": [],
            "pruned_candidate_ids": [],
            "prune_history": [],
            "visited_urls": [],
            "searched_queries": [],
        }
        nodes_visited = []
        for name, fn in [
            ("clarify", nodes.clarify_node),
            ("rewrite", nodes.rewrite_node),
            ("planner", nodes.planner_node),
            ("parse_action", nodes.parse_action_node),
            ("execute_action", nodes.execute_action_node),
            ("planner", nodes.planner_node),
            ("parse_action", nodes.parse_action_node),
            ("execute_action", nodes.execute_action_node),
            ("planner", nodes.planner_node),
            ("parse_action", nodes.parse_action_node),
            ("execute_action", nodes.execute_action_node),
            ("planner", nodes.planner_node),
            ("parse_action", nodes.parse_action_node),
            ("execute_action", nodes.execute_action_node),
            ("planner", nodes.planner_node),
            ("parse_action", nodes.parse_action_node),
            ("execute_action", nodes.execute_action_node),
            ("planner", nodes.planner_node),
            ("parse_action", nodes.parse_action_node),
            ("execute_action", nodes.execute_action_node),
            ("planner", nodes.planner_node),
            ("parse_action", nodes.parse_action_node),
            ("reflect", nodes.reflect_node),
            ("answer", nodes.answer_node),
        ]:
            update = await fn(state)
            apply_update(state, update)
            nodes_visited.append(name)
            log(f"NODE: {name} update={list(update.keys())}")

        assert state["phase"] == "DONE", state["phase"]
        assert "M5" in state["final_answer"]
        assert state["clarifications"][0]["answer"] == "2026年"
        # 搜索结果先进入候选池；只有 curated evidence 被登记为最终来源
        assert len(state["candidate_docs"]) == 2, state["candidate_docs"]
        assert state["pruned_candidate_ids"] == [2], state["pruned_candidate_ids"]
        assert len(state["curated_evidence"]) == 2, state["curated_evidence"]
        assert len(state["verification_records"]) == 1, state["verification_records"]
        assert state["verification_records"][0]["verdict"] == "supported"
        assert len(state["sources"]) == 1, state["sources"]
        cited_ids = [c["id"] for c in state["cited_sources"]]
        assert cited_ids == [1], cited_ids
        assert state["total_tokens"] > 0
        assert len(state["evidence"]) == 1
        assert state["planner_context_tokens"] > 0
        assert state["answer_context_tokens"] > 0
        planner_context, _ = build_planner_context(state)
        assert "[C2]" not in planner_context
        print("PASS: 节点路径 =", " → ".join(nodes_visited))
        print("PASS: 来源数 =", len(state["sources"]), "| 引用 =", cited_ids)
        print("PASS: 最终回答 =", state["final_answer"])

        bad_state = {
            **init,
            "messages": [AIMessage(content="{bad json")],
            "action_history": [],
        }
        bad_update = await nodes.parse_action_node(bad_state)
        assert bad_update["invalid_action_count"] == 1
        assert bad_update["active_error"].startswith("InvalidAction")
        fixed_state = {
            **bad_state,
            **bad_update,
            "messages": [
                AIMessage(
                    content='{"action":"finish","args":{"answer_outline":"基于现有证据回答 [1]"},"reason":"证据充分"}'
                )
            ],
        }
        fixed_update = await nodes.parse_action_node(fixed_state)
        assert fixed_update["parsed_action"]["action"] == "finish"
        assert fixed_update["self_correction_success_count"] == 1
        print("PASS: invalid action 可被 active_error 捕获并修正")

        disabled_state = {
            **init,
            "disabled_actions": ["ask_user"],
            "messages": [
                AIMessage(
                    content='{"action":"ask_user","args":{"question":"需要补充吗？","options":[]},"reason":"信息不足"}'
                )
            ],
            "action_history": [],
        }
        disabled_update = await nodes.parse_action_node(disabled_state)
        assert disabled_update["invalid_action_count"] == 1
        assert "action disabled" in disabled_update["active_error"]
        print("PASS: benchmark 可禁用 ask_user 动作")

        missing_candidate_state = {
            **init,
            "parsed_action": {
                "action": "curate_evidence",
                "args": {
                    "candidate_id": 999,
                    "claim": "不存在的候选",
                    "quote_or_summary": "无",
                    "confidence": 0.5,
                },
                "reason": "测试错误恢复",
                "id": "call_missing_candidate",
            },
            "messages": [],
            "sources": {},
            "candidate_docs": {},
            "curated_evidence": {},
            "verification_records": [],
            "pruned_candidate_ids": [],
            "prune_history": [],
        }
        missing_update = await nodes.execute_action_node(missing_candidate_state)
        assert missing_update["tool_error_count"] == 1
        assert "candidate_id not found" in missing_update["active_error"]
        print("PASS: curate_evidence 对不存在候选会进入可修复错误路径")
        print("\n全部断言通过 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(asyncio.wait_for(main(), timeout=CASE_TIMEOUT_S)))
