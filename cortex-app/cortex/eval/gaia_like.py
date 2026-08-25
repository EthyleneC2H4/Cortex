"""GAIA 式准精确答案匹配与分层指标（纯函数，零 LLM 依赖）。

参照 GAIA 基准的 exact_match 规则做中文友好的规范化：
去冠词、去货币符号与百分号、合并数字千位逗号、折叠空白、去尾部标点，
再比较规范化后的字符串；逗号分隔列表按排序后的 token 集比较。
"""

from __future__ import annotations

import re
from typing import Sequence

# 冠词按整词移除（\b 保证不误伤 and / about 等）
_ARTICLE_RE = re.compile(r"\b(?:a|an|the)\b")
# 货币符号与百分号
_CURRENCY_RE = re.compile(r"[$€¥%]")
# 数字之间的千位逗号（"1,234"→"1234"；不影响列表逗号，因其后非数字）
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d)")
# 空白折叠
_SPACES_RE = re.compile(r"\s+")
# 尾部标点
_TRAILING_PUNCT = ".,!?;:。！？；"


def normalize(answer: str) -> str:
    """规范化答案：strip→lower→全角逗号折半角→去冠词→去货币符号与%→千位逗号合并→空白折叠→去尾部标点。"""
    s = str(answer).strip().lower()
    s = s.replace("，", ", ")  # 中文数据集的全角列表分隔符按半角处理
    s = _ARTICLE_RE.sub("", s)
    s = _CURRENCY_RE.sub("", s)
    s = _THOUSANDS_RE.sub("", s)
    s = _SPACES_RE.sub(" ", s).strip()
    return s.rstrip(_TRAILING_PUNCT).strip()


def quasi_exact_match(pred: str, gold: str) -> bool:
    """准精确匹配：规范化后相等；两者都含逗号分隔列表时按排序 token 集比较。

    列表切分用 ``re.split(r"\\s*,\\s*")``：逗号前后空白不参与 token，
    "a , b" 与 "a,b" 同形。
    """
    norm_pred, norm_gold = normalize(pred), normalize(gold)
    if norm_pred == norm_gold:
        return True
    if "," in norm_pred and "," in norm_gold:
        pred_toks = sorted(t for t in re.split(r"\s*,\s*", norm_pred) if t)
        gold_toks = sorted(t for t in re.split(r"\s*,\s*", norm_gold) if t)
        return pred_toks == gold_toks
    return False


def level_metrics(rows: Sequence[dict]) -> dict:
    """按 level 分层统计：{"L1": {"n": 总数, "correct": 命中数}, ...}，键按层级排序。"""
    stats: dict = {}
    for row in rows:
        level = str(row.get("level", "?"))
        slot = stats.setdefault(level, {"n": 0, "correct": 0})
        slot["n"] += 1
        if row.get("correct"):
            slot["correct"] += 1
    return {level: stats[level] for level in sorted(stats)}
