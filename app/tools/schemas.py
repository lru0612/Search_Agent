"""暴露给 ReAct 决策节点的工具 schema（仅用于 bind_tools，执行在 tools 节点中分发）。"""
from pydantic import BaseModel, Field


class WebSearch(BaseModel):
    """使用 Tavily 搜索引擎检索网页。适合获取新信息或寻找候选页面。"""

    query: str = Field(description="搜索查询，应具体、含关键限定词")
    max_results: int = Field(default=5, ge=1, le=10, description="返回结果数")


class VisitPage(BaseModel):
    """访问指定 URL，获取网页正文（markdown）。适合深入阅读搜索结果中有价值的页面。"""

    url: str = Field(description="要访问的完整 http/https 链接")
    reason: str = Field(description="为什么要访问这个页面（一句话）")


class AskUser(BaseModel):
    """当现有信息不足以继续、且只有用户能补充时，向用户提问。会暂停搜索流程。"""

    question: str = Field(description="向用户提出的具体问题")
    options: list[str] = Field(default_factory=list, description="可选的候选答案，便于用户快速选择")


class CurateEvidence(BaseModel):
    """把候选文档中的关键信息提升为最终可引用证据。"""

    candidate_id: int = Field(description="candidate_docs 中的候选文档编号")
    claim: str = Field(description="这条证据支撑的具体事实或论断")
    quote_or_summary: str = Field(description="候选文档中的关键摘录或简短摘要")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="对该证据支撑力度的置信度")


class PruneCandidates(BaseModel):
    """从后续上下文中剪掉与任务无关或价值低的候选文档。"""

    candidate_ids: list[int] = Field(description="要剪掉的候选文档编号列表")
    reason: str = Field(description="剪枝理由")


class VerifyClaim(BaseModel):
    """核验一个关键论断是否被当前候选或来源支持。"""

    claim: str = Field(description="需要核验的事实性论断")
    source_ids: list[int] = Field(description="用于核验的来源编号或候选文档编号")


class Finish(BaseModel):
    """信息已足够回答用户问题时调用，进入回答阶段。"""

    answer_outline: str = Field(description="回答要点提纲，列出将覆盖的论点及对应来源编号")


TOOL_SCHEMAS = [WebSearch, VisitPage, AskUser, CurateEvidence, PruneCandidates, VerifyClaim, Finish]

# bind_tools 后模型产出的 tool name 与此对应
TOOL_NAME_MAP = {
    "WebSearch": "web_search",
    "VisitPage": "visit_page",
    "AskUser": "ask_user",
    "CurateEvidence": "curate_evidence",
    "PruneCandidates": "prune_candidates",
    "VerifyClaim": "verify_claim",
    "Finish": "finish",
}
