"""知识三元组抽取：弱模型 JSON 模式，宽容解析（截取首个 [...] 块）。"""

from __future__ import annotations

import json
import re
from typing import List, Optional

_PROMPT = """你是知识图谱构建助手。从下面的文本中抽取 (主体, 关系, 客体) 三元组。
要求：
- 最多 {max_triples} 条，只保留置信度 ≥ {min_conf} 的
- relation 用简短中文谓词（如"提出"、"属于"、"对比"、"依赖"）
- 严格输出 JSON 数组，不要任何解释文字：
[{{"subject": "...", "relation": "...", "object": "...", "confidence": 0.9}}]

文本：
{text}
"""

_ARRAY = re.compile(r"\[.*\]", re.DOTALL)


def _parse_json_array(raw: str) -> Optional[List[dict]]:
    """宽容解析：剥 markdown 围栏 → 截取首个平衡 [...]。"""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text, flags=re.DOTALL).strip()
    m = _ARRAY.search(text)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) else None


def extract_triples(
    text: str,
    weak_llm,
    *,
    max_triples: int = 20,
    min_conf: float = 0.8,
    source_doc: str = "",
) -> List[dict]:
    """返回规范化三元组列表；LLM 失败/超限输入一律空列表（不阻塞摄取）。"""
    snippet = text[:6000]
    prompt = _PROMPT.format(max_triples=max_triples, min_conf=min_conf, text=snippet)
    try:
        resp = weak_llm.invoke([{"role": "user", "content": prompt}])
        raw = getattr(resp, "content", None) or str(resp)
    except Exception:
        return []
    items = _parse_json_array(str(raw)) or []
    out: List[dict] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        try:
            conf = float(it.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        if conf < min_conf:
            continue
        triple = {
            "subject": str(it.get("subject", "")).strip()[:120],
            "relation": str(it.get("relation", "")).strip()[:60],
            "object": str(it.get("object", "")).strip()[:120],
            "confidence": conf,
            "source_doc": source_doc,
        }
        if triple["subject"] and triple["object"] and triple["relation"]:
            out.append(triple)
        if len(out) >= max_triples:
            break
    return out
