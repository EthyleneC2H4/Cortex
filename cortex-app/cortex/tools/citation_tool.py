"""CitationTool — 引用核验：取 chunk 原文 → 弱模型蕴含判定。"""

from __future__ import annotations

from typing import Any, Dict, List

from hello_agents.tools import Tool, ToolResponse, tool_action
from hello_agents.tools.errors import ToolErrorCode

from ..services.knowledge.qa import normalize_chunk_id

_VERIFY_PROMPT = """判断下面的"主张"是否被"证据原文"支持。
只输出 JSON：{"supported": true/false, "confidence": 0.0-1.0}

主张：{claim}

证据原文：
{evidence}
"""


def _parse_verdict(raw: str) -> tuple[bool, float]:
    import json
    import re

    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text, flags=re.DOTALL).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            return bool(data.get("supported")), min(1.0, max(0.0, float(data.get("confidence", 0.5))))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    lowered = text.lower()
    return ("true" in lowered), 0.5


class CitationTool(Tool):
    """citation_verify(claim, chunk_id)：核验陈述是否有知识库证据支撑。"""

    def __init__(self, knowledge, *, weak_llm=None) -> None:
        self.knowledge = knowledge  # KnowledgeService（用 db 取 chunk 原文）
        self.weak_llm = weak_llm
        super().__init__(
            name="citation",
            description="引用核验：判断一个主张是否被指定知识库片段支持",
            expandable=True,
        )

    @tool_action(description="核验主张是否被指定 chunk 支持")
    def verify(self, claim: str, chunk_id: str) -> ToolResponse:
        """蕴含判定。

        Args:
            claim: 待核验的主张
            chunk_id: 知识库片段 ID；宽容接受 doc:idx / doc/doc:idx / doc/idx 等写法
        """
        cid = normalize_chunk_id(chunk_id)
        row = None
        if cid and not cid.startswith(":"):
            row = self.knowledge.db.query_one("SELECT text FROM chunks WHERE chunk_id=?", (cid,))
        elif cid:
            # 残缺标记（'/3'、':3'）：仅当全库唯一后缀命中时才采纳
            rows = self.knowledge.db.query(
                "SELECT chunk_id, text FROM chunks WHERE chunk_id LIKE ?", (f"%{cid}",))
            if len(rows) == 1:
                row = rows[0]
        if not row or not cid:
            return ToolResponse.error(ToolErrorCode.NOT_FOUND,
                                      f"chunk 不存在: {chunk_id}（归一为 {cid}）")
        evidence = row["text"]
        chunk_id = row.get("chunk_id", cid)

        if self.weak_llm is None:
            # 无 LLM 时退化为词面重叠启发式
            overlap = sum(1 for tok in set(claim) & set(evidence)) / max(len(set(claim)), 1)
            supported, conf = overlap > 0.5, round(min(0.9, overlap), 2)
            basis = "heuristic"
        else:
            from ..llm.resilience import resilient_invoke

            try:
                resp = resilient_invoke(
                    self.weak_llm,
                    [{"role": "user", "content": _VERIFY_PROMPT.format(claim=claim[:800], evidence=evidence[:2500])}],
                )
                supported, conf = _parse_verdict(str(getattr(resp, "content", "")))
                basis = "llm"
            except Exception as exc:
                return ToolResponse.error(ToolErrorCode.EXECUTION_ERROR, f"核验失败: {exc}")

        verdict = "支持" if supported else "不支持"
        return ToolResponse.success(
            text=f"[{verdict} conf={conf}] {claim[:80]}",
            data={"supported": supported, "confidence": conf, "chunk_id": chunk_id, "basis": basis},
        )

    def get_parameters(self) -> List[Any]:  # pragma: no cover
        from hello_agents.tools import ToolParameter

        return [ToolParameter(name="action", type="string", description="动作名")]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:  # pragma: no cover
        return ToolResponse.error(ToolErrorCode.INVALID_PARAM, "应通过 citation_verify 子工具调用")
