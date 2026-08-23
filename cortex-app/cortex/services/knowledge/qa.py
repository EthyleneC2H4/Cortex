"""带引用问答：检索命中拼上下文 → 强模型生成 `[doc_id/chunk_id]` 标记答案。

三带策略：answer 直接答；uncertain 答但声明证据不足；refuse 固定拒答话术
+ 提示可联网（researcher 子代理）补充。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .retrieve import RetrievalResult, retrieve

_REFUSE_TEXT = "知识库中没有找到与该问题相关的资料，无法给出有依据的回答。"

_QA_PROMPT = """基于且仅基于下面的资料回答问题。

规则：
1. 每个事实性陈述后必须紧跟引用标记，格式 [doc_id/chunk_id]，如 [a1b2/a1b2:3]
2. 资料不足以支撑的部分，明确说"资料未涵盖"
3. 不要编造资料中没有的内容
{band_note}

资料：
{context}

问题：{question}
"""


@dataclass
class QAAnswer:
    question: str
    band: str
    answer: str
    citations: List[dict] = field(default_factory=list)
    result: Optional[RetrievalResult] = None


def _format_context(result: RetrievalResult) -> str:
    blocks = []
    for i, h in enumerate(result.hits, 1):
        head = f"[{i}] {h.heading_path}（来源: {h.source}）" if h.heading_path else f"[{i}] {h.source}"
        blocks.append(f"{head}\nchunk_id={h.chunk_id}\n{h.text}")
    return "\n\n".join(blocks)


def _band_note(band: str) -> str:
    if band == "uncertain":
        return "4. 当前检索到的证据相关性偏低：先说明证据有限，再给出最可能的回答。"
    return ""


def answer_question(
    deps,
    question: str,
    strong_llm,
    *,
    weak_llm=None,
    top_k: int = 5,
    min_score: float = 0.3,
) -> QAAnswer:
    from ...llm.resilience import resilient_invoke  # 局部导入避免环

    result = retrieve(deps, question, weak_llm=weak_llm, top_k=top_k, min_score=min_score)
    if result.band == "refuse":
        return QAAnswer(
            question=question,
            band="refuse",
            answer=_REFUSE_TEXT + "\n（提示：可将该问题交给联网研究子代理补充外部资料后重试）",
            result=result,
        )

    prompt = _QA_PROMPT.format(
        band_note=_band_note(result.band), context=_format_context(result), question=question
    )
    try:
        resp = resilient_invoke(strong_llm, [{"role": "user", "content": prompt}])
        answer = str(getattr(resp, "content", ""))
    except Exception as exc:
        answer = f"（生成答案失败：{exc}）\n以下是检索到的相关资料片段，请自行查阅。\n\n{_format_context(result)}"

    cited = {c for c in retrieve_citation_ids(answer)}
    citations = [
        {"chunk_id": h.chunk_id, "doc_id": h.doc_id, "source": h.source, "heading_path": h.heading_path}
        for h in result.hits
        if h.chunk_id in cited
    ]
    return QAAnswer(question=question, band=result.band, answer=answer.strip(), citations=citations, result=result)


_CITE = __import__("re").compile(r"\[([0-9a-f]+/[^\]\s]+)\]")


def normalize_chunk_id(raw: str) -> Optional[str]:
    """引用标记宽容归一为规范 chunk_id（doc:idx）。

    实测模型会混用多种写法：'doc/doc:idx'（qa 标准标记）、'doc/idx'（写作简写）、
    'doc:idx'、'/idx' 与 ':idx'（残缺片段）。doc 缺失时返回 ':idx' 交由调用方
    模糊匹配；无法解析返回 None。
    """
    cid = (raw or "").strip().strip("[]").strip()
    if not cid:
        return None
    if ":" in cid:
        return cid.rpartition("/")[2] if "/" in cid else cid
    doc, _, idx = cid.rpartition("/")
    return f"{doc}:{idx}" if doc else f":{idx}"


def retrieve_citation_ids(text: str) -> List[str]:
    """从答案文本提取被引用的 chunk_id（宽容归一见 normalize_chunk_id）。"""
    out = []
    for marker in (m.group(1) for m in _CITE.finditer(text)):
        cid = normalize_chunk_id(marker)
        if cid and not cid.startswith(":") and cid not in out:
            out.append(cid)
    return out
