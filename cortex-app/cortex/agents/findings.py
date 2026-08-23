"""Findings 结构化中间产物：流水线各阶段间的标准交接格式（pydantic + 宽容解析）。"""

from __future__ import annotations

import json
import re
from typing import List

from pydantic import BaseModel, Field


class Citation(BaseModel):
    doc_id: str = ""
    chunk_id: str = ""
    location: str = ""  # 章节/页码等人类可读位置


class Findings(BaseModel):
    """子代理回传的研究发现；摘要控制在 1000–2000 token。"""

    conclusions: List[str] = Field(default_factory=list)
    supporting: List[str] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    confidence: float = 0.5
    conflicts: List[str] = Field(default_factory=list)
    raw: str = ""  # 解析失败时保留原文，供降级使用


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _balanced_objects(text: str):
    """按括号深度扫描所有平衡的 {...} 片段（跳过字符串内的括号），后出现先返回。"""
    out, depth, start, in_str, esc = [], 0, -1, False, False
    for i, ch in enumerate(text):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if in_str:
            if ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    out.append(text[start : i + 1])
    return list(reversed(out))


def parse_findings(text: str, *, max_tokens: int = 2000) -> Findings:
    """宽容解析：剥围栏 → 平衡 {...} 逐个尝试（取含 conclusions 的最后一个）
    → 截断 JSON 抢救 → 降级为原文。"""
    body = (text or "").strip()
    if not body:
        return Findings(raw="(空回复)")
    candidate = body
    if candidate.startswith("```"):
        candidate = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", candidate.strip(), flags=re.DOTALL).strip()
    for chunk in _balanced_objects(candidate):
        try:
            data = json.loads(chunk)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict) and any(
            k in data for k in ("conclusions", "supporting", "citations", "conflicts", "confidence")
        ):
            data["raw"] = ""
            return _clamp(Findings(**_coerce(data)), max_tokens)
    # 截断抢救：JSON 被生成上限截断时，逐字段正则捞回可用部分
    salvaged = _salvage_truncated(candidate)
    if salvaged is not None:
        return _clamp(salvaged, max_tokens)
    # 降级：整段当单条结论
    return _clamp(Findings(conclusions=[body[:1500]], confidence=0.3, raw=body[:2000]), max_tokens)


_LIST_ITEMS = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"')


def _unescape(s: str) -> str:
    """按 JSON 字符串语义反转义（保 UTF-8；失败原样返回）。"""
    try:
        return json.loads(f'"{s}"')
    except Exception:  # noqa: BLE001
        return s


def _salvage_truncated(text: str) -> Findings | None:
    """从被截断的 JSON 文本中按字段名抢救列表与置信度；毫无收获返回 None。"""
    found: dict = {}

    def grab(field: str) -> List[str] | None:
        m = re.search(rf'"{field}"\s*:\s*\[(.*?)\]', text, re.DOTALL)
        if not m:
            # 未闭合数组：取到文本末尾
            m2 = re.search(rf'"{field}"\s*:\s*\[(.*)$', text, re.DOTALL)
            if not m2:
                return None
            seg = m2.group(1)
        else:
            seg = m.group(1)
        items = [_unescape(s) for s in _LIST_ITEMS.findall(seg)]
        cleaned = [it.strip() for it in items if it.strip()]
        return cleaned or None

    for field in ("conclusions", "supporting", "conflicts"):
        vals = grab(field)
        if vals:
            found[field] = vals
    cm = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
    if not found.get("conclusions"):
        return None
    conf = float(cm.group(1)) if cm else 0.5
    return Findings(
        conclusions=found.get("conclusions", []),
        supporting=found.get("supporting", []),
        conflicts=found.get("conflicts", []),
        confidence=min(1.0, max(0.0, conf)),
        citations=[],
        raw="(截断抢救)",
    )


def _coerce(data: dict) -> dict:
    """字段类型矫正：字符串→列表、数字置信度 clamp、引用 dict 列表容错。"""
    out = dict(data)
    for k in ("conclusions", "supporting", "conflicts"):
        v = out.get(k)
        if isinstance(v, str):
            out[k] = [v]
        elif v is None:
            out[k] = []
    cites = out.get("citations") or []
    fixed = []
    for c in cites:
        if isinstance(c, str):
            fixed.append(Citation(chunk_id=c).model_dump())
        elif isinstance(c, dict):
            fixed.append(c)
    out["citations"] = fixed
    try:
        conf = float(out.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    out["confidence"] = min(1.0, max(0.0, conf))
    return out


def _clamp(f: Findings, max_tokens: int) -> Findings:
    """粗粒度截断：总字符 ≈ token×2.5。"""
    budget_chars = max(500, int(max_tokens * 2.5))

    def clip_list(items: List[str], share: float) -> List[str]:
        cap = int(budget_chars * share)
        used, out = 0, []
        for it in items:
            piece = it[:600]
            if used + len(piece) > cap and out:
                break
            out.append(piece)
            used += len(piece)
        return out

    n = sum(len(x) for x in f.conclusions + f.supporting + f.conflicts)
    if n <= budget_chars:
        return f
    return Findings(
        conclusions=clip_list(f.conclusions, 0.5),
        supporting=clip_list(f.supporting, 0.35),
        citations=f.citations,
        confidence=f.confidence,
        conflicts=clip_list(f.conflicts, 0.15),
        raw="",
    )
