"""KnowledgeSearchTool — 知识库工具（P3）。

expandable → ``knowledge_add_document / knowledge_add_text / knowledge_search / knowledge_qa``
四个子工具；search/qa 返回紧凑 LLM 文本 + data 明细。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from hello_agents.tools import Tool, ToolResponse, tool_action
from hello_agents.tools.errors import ToolErrorCode


class KnowledgeSearchTool(Tool):
    """个人知识库：摄取文档、混合检索、带引用问答。"""

    def __init__(self, service, *, weak_llm=None, strong_llm=None) -> None:
        self.service = service  # KnowledgeService
        self.weak_llm = weak_llm
        self.strong_llm = strong_llm
        super().__init__(
            name="knowledge",
            description="个人知识库检索与问答（RAG）：摄取文档、混合检索、引用作答",
            expandable=True,
        )

    # 方法名 → 子工具名：knowledge_add_document / knowledge_add_text /
    # knowledge_search / knowledge_qa
    @tool_action(description="摄取一个本地文件（md/txt/pdf/docx…）入知识库")
    def add_document(self, path: str) -> ToolResponse:
        """摄取文件。

        Args:
            path: 文件绝对或相对路径
        """
        p = Path(path)
        if not p.exists():
            return ToolResponse.error(ToolErrorCode.NOT_FOUND, f"文件不存在: {path}")
        try:
            info = self.service.ingest_document(p, weak_llm=self.weak_llm)
        except Exception as exc:
            return ToolResponse.error(ToolErrorCode.EXECUTION_ERROR, f"摄取失败: {exc}")
        return ToolResponse.success(
            text=f"已摄取「{info['title']}」→ doc_id={info['doc_id']}，{info['n_chunks']} 个块",
            data=info,
        )

    @tool_action(description="直接摄取一段文本入知识库（无需落盘文件）")
    def add_text(self, title: str, content: str) -> ToolResponse:
        """摄取文本。

        Args:
            title: 文档标题（作为 doc 标识的一部分）
            content: 正文文本
        """
        if not content.strip():
            return ToolResponse.error(ToolErrorCode.INVALID_PARAM, "内容不能为空")
        try:
            info = self.service.ingest_text(title=title, text=content, weak_llm=self.weak_llm)
        except Exception as exc:
            return ToolResponse.error(ToolErrorCode.EXECUTION_ERROR, f"摄取失败: {exc}")
        return ToolResponse.success(
            text=f"已摄取「{title}」→ doc_id={info['doc_id']}，{info['n_chunks']} 个块",
            data=info,
        )

    @tool_action(description="混合检索知识库，返回相关片段列表")
    def search(self, query: str, top_k: int = 5) -> ToolResponse:
        """混合检索。

        Args:
            query: 检索问题
            top_k: 返回条数上限
        """
        result = self.service.search(query, top_k=int(top_k), weak_llm=self.weak_llm)
        if result.band == "refuse":
            return ToolResponse.success(
                text=f"知识库中没有找到与「{query}」相关的资料",
                data={"band": "refuse", "hits": []},
            )
        lines = [
            f"[{h.chunk_id}] ({h.heading_path or '无标题'}) {h.text[:200]}…"
            for h in result.hits
        ]
        return ToolResponse.success(
            text="\n".join(lines),
            data={
                "band": result.band,
                "hits": [
                    {"chunk_id": h.chunk_id, "doc_id": h.doc_id, "source": h.source,
                     "heading_path": h.heading_path, "text": h.text}
                    for h in result.hits
                ],
            },
        )

    @tool_action(description="对知识库提问，得到带 [doc_id/chunk_id] 引用的回答；证据不足时拒答")
    def qa(self, question: str) -> ToolResponse:
        """带引用问答。

        Args:
            question: 问题
        """
        if self.strong_llm is None:
            return ToolResponse.error(ToolErrorCode.INVALID_PARAM, "knowledge_qa 需要 strong LLM（未配置）")
        ans = self.service.ask(question, self.strong_llm, weak_llm=self.weak_llm)
        header = "" if ans.band == "answer" else f"（证据{ '有限' if ans.band == 'uncertain' else '缺失'}）\n"
        return ToolResponse.success(
            text=header + ans.answer,
            data={"band": ans.band, "citations": ans.citations},
        )

    # 抽象契约（展开路径不会直调父工具）
    def get_parameters(self) -> List[Any]:  # pragma: no cover
        from hello_agents.tools import ToolParameter

        return [ToolParameter(name="action", type="string", description="动作名")]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:  # pragma: no cover
        return ToolResponse.error(
            ToolErrorCode.INVALID_PARAM, "KnowledgeSearchTool 应通过 knowledge_* 子工具调用"
        )
