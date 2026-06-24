# Agentic Search

基于 **LangGraph** 的多步搜索 Agent：判断查询歧义并向用户澄清、重写查询、循环「搜索 / 读页 / 提问 / 收尾」，经反思自检后输出**带来源引用**的回答。Web 界面为 Claude 风格的流式聊天页。

## 功能特性

- **歧义澄清**：LLM 先判断查询是否歧义/过宽，必要时通过 LangGraph `interrupt()` 挂起流程向用户提问，回复后从断点恢复
- **Query 重写**：将口语化查询改写为 1~3 个高质量搜索查询
- **Stateful retrieval harness**：搜索结果先进入可恢复的候选池，模型显式执行证据保留、剪枝、核验后再回答
- **ReAct 循环**：动作包括 `web_search`（Tavily）/ `visit_page`（Jina Reader）/ `curate_evidence` / `prune_candidates` / `verify_claim` / `ask_user` / `finish`，至多 `MAX_STEPS` 步
- **反思自检**：`finish` 前质检证据覆盖度与来源矛盾，不足则带着缺口继续搜索（亮点）
- **句级引用**：回答中每个论断带 `[n]` 引用，前端悬停展示来源片段、点击跳转（亮点）
- **生命周期控制**：步数/token 预算、重复搜索停滞检测、用户取消、预算耗尽强制收尾
- **可观测性**：全过程结构化事件实时推送前端时间线，并落盘 `traces/{session_id}.jsonl`

## 架构

```
clarify ──有歧义──> ask_clarify (interrupt) ──┐
   │ 清晰                                      │ 用户回复后回到 clarify
   v
rewrite ──> planner ──> parse_action ──> execute_action (search / read / curate / prune / verify / ask_user)
              ↑              │ finish / 预算耗尽
              │              v
              └──── reflect <─┘
                       │ 通过
                       v
                    answer（流式输出 + [n] 引用）
```

- `app/agent/graph.py`：StateGraph 与 MemorySaver checkpointer
- `app/agent/nodes.py`：各节点实现（澄清、重写、planner、动作解析/修复、执行器、反思、回答）
- `app/agent/context.py`：按 sub-agent 构造干净上下文（planner / reflect / answer）
- `AgentState.action_history` / `candidate_docs` / `curated_evidence` / `verification_records`：结构化动作历史、候选池、保留证据与核验记录；全量 trace 落盘，不直接回灌给模型
- `app/citations.py`：来源注册表与 `[n]` 引用解析
- `app/observability.py`：Tracer（SSE 事件 + JSONL trace）
- `app/main.py`：FastAPI，`/api/chat`（SSE）、`/api/cancel/{session_id}`

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env   # 填入 OPENAI_API_KEY / OPENAI_BASE_URL / MODEL_NAME / TAVILY_API_KEY
uvicorn app.main:app --reload
```

打开 http://localhost:8000 即可使用。模型走 OpenAI 兼容接口，DeepSeek / Qwen / OpenAI 均可（在 `.env` 中配 `OPENAI_BASE_URL` 与 `MODEL_NAME`）。

### ask_user 的暂停与恢复

agent 提澄清问题时，SSE 下发 `ask_user` 事件后本次请求结束，图状态由 checkpointer 按 `thread_id`（即 session_id）保存；前端把用户回复以 `{"resume": true, "session_id": ...}` 再次 POST `/api/chat`，后端用 `Command(resume=回复)` 从断点继续。

## 测试

无需 API key 的全流程 mock 测试（验证图接线、interrupt 恢复、harness 候选/剪枝/保留/核验、来源去重与引用提取）：

```bash
python -m tests.test_flow
```

Benchmark pipeline（按组别保存 trace，终端实时输出 log）：

```bash
python -m bench fetch-browsecomp --out data/browsecomp_test.jsonl
python -m bench run --suite browsecomp --split test --data-file data/browsecomp_test.jsonl --limit 30
python -m bench run --suite browsecomp --split smoke --limit 5
python -m bench report runs/<run_id>
python -m bench diagnose runs/<run_id>
```

离线验证 pipeline 骨架（不调用外部模型/搜索）：

```bash
python -m bench run --suite browsecomp --split smoke --limit 2 --mock-agent
```

运行产物位于 `runs/<run_id>/`，trace 位于 `traces/<suite>_<split>/<run_id>/`。
真实 BrowseComp 数据来自 OpenAI `simple-evals` 的公开 CSV，`fetch-browsecomp` 会按官方脚本用每行 `canary` 解密题目和答案并转为本项目 JSONL。

## 配置项（.env）

| 变量 | 说明 | 默认 |
|---|---|---|
| `MAX_STEPS` | 搜索循环最大步数 | 16 |
| `MAX_REFLECT_ROUNDS` | 反思最多打回次数 | 2 |
| `MAX_CLARIFY_ROUNDS` | 最多澄清轮数 | 2 |
| `TOKEN_BUDGET` | 单次会话 token 预算 | 180000 |
| `CONTEXT_TOKEN_LIMIT` | 触发上下文裁剪的阈值 | 90000 |
| `JINA_API_KEY` | 可选，提升 Jina Reader 限速 | 空 |
