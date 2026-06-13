# Agentic Search

基于 **LangGraph** 的多步搜索 Agent：判断查询歧义并向用户澄清、重写查询、循环「搜索 / 读页 / 提问 / 收尾」，经反思自检后输出**带来源引用**的回答。Web 界面为 Claude 风格的流式聊天页。

## 功能特性

- **歧义澄清**：LLM 先判断查询是否歧义/过宽，必要时通过 LangGraph `interrupt()` 挂起流程向用户提问，回复后从断点恢复
- **Query 重写**：将口语化查询改写为 1~3 个高质量搜索查询
- **ReAct 循环**：四个动作 `web_search`（Tavily）/ `visit_page`（Jina Reader）/ `ask_user` / `finish`，至多 `MAX_STEPS` 步
- **反思自检**：`finish` 前质检证据覆盖度与来源矛盾，不足则带着缺口继续搜索（亮点）
- **句级引用**：回答中每个论断带 `[n]` 引用，前端悬停展示来源片段、点击跳转（亮点）
- **生命周期控制**：步数/token 预算、重复搜索停滞检测、用户取消、预算耗尽强制收尾
- **可观测性**：全过程结构化事件实时推送前端时间线，并落盘 `traces/{session_id}.jsonl`

## 架构

```
clarify ──有歧义──> ask_clarify (interrupt) ──┐
   │ 清晰                                      │ 用户回复后回到 clarify
   v
rewrite ──> agent <──────> tools (web_search / visit_page / ask_user)
              │ finish / 预算耗尽
              v
           reflect ──缺口──> agent（继续搜）
              │ 通过
              v
           answer（流式输出 + [n] 引用）
```

- `app/agent/graph.py`：StateGraph 与 MemorySaver checkpointer
- `app/agent/nodes.py`：各节点实现（澄清、重写、决策、工具执行、反思、回答）
- `app/agent/context.py`：上下文裁剪（早期 observation 压缩为摘要）
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

无需 API key 的全流程 mock 测试（验证图接线、interrupt 恢复、来源去重与引用提取）：

```bash
python -m tests.test_flow
```

## 配置项（.env）

| 变量 | 说明 | 默认 |
|---|---|---|
| `MAX_STEPS` | 搜索循环最大步数 | 10 |
| `MAX_REFLECT_ROUNDS` | 反思最多打回次数 | 2 |
| `MAX_CLARIFY_ROUNDS` | 最多澄清轮数 | 2 |
| `TOKEN_BUDGET` | 单次会话 token 预算 | 120000 |
| `CONTEXT_TOKEN_LIMIT` | 触发上下文裁剪的阈值 | 45000 |
| `JINA_API_KEY` | 可选，提升 Jina Reader 限速 | 空 |
