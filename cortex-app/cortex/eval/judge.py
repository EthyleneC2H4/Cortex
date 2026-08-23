"""LLM-as-Judge：逐维打分、成对对比胜率与 Cohen's κ 一致性。

评测套件中唯一允许触碰 LLM 的模块：judge_llm 只需实现 ``invoke(messages)``
并返回带 ``content`` 的对象（同框架 LLM 惯例）。解析一律宽容：
剥 markdown 围栏后截取首个平衡 {...}；LLM 异常或输出不可读时确定性降级。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Optional, Sequence

# 评审维度（固定四维，键为中文）
JUDGE_DIMS = ("准确性", "相关性", "完整性", "引用可靠性")

_SCORE_PROMPT = """你是严格的评测裁判。依据参考资料对「回答」逐维打 1-5 分：
- 准确性：与参考资料一致，无事实错误
- 相关性：切题不跑题，无注水
- 完整性：要点覆盖程度
- 引用可靠性：结论忠于所给参考资料

问题：
{question}

参考资料：
{references}

回答：
{answer}

严格只输出如下 JSON（不要输出任何其他文字）：
{{"准确性": <1-5>, "相关性": <1-5>, "完整性": <1-5>, "引用可靠性": <1-5>, "rationale": "<一句话理由>"}}
"""

_PAIR_PROMPT = """你是严格的评审。同一问题有两个候选回答，标签固定：【回答 A】与【回答 B】。
只判断哪个更好。若不相上下输出 tie。

问题：
{question}

{blocks}

只输出一个单词：A、B 或 tie。
"""


def _extract_first_object(text: str) -> Optional[str]:
    """扫描文本，截取首个括号平衡的 {...} 子串；字符串内的花括号不计深度。"""
    depth = 0
    in_str = False
    escaped = False
    start = -1
    for i, ch in enumerate(text):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
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
                if depth == 0:
                    return text[start:i + 1]
    return None


def _parse_json_object(raw: str) -> Optional[dict]:
    """宽容解析：剥 markdown 围栏 → 截取首个平衡 {...} → json.loads；失败返回 None。"""
    text = str(raw).strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"```\s*$", "", text).strip()
    snippet = _extract_first_object(text)
    if snippet is None:
        return None
    try:
        data = json.loads(snippet)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _score_of(value) -> int:
    """分数收敛到 1..5 的整数：非数值或缺失记 3 分，越界 clamp。"""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return 3
    return int(max(1, min(5, round(num))))


def judge_answer(question: str, answer: str, references: str, judge_llm) -> dict:
    """单回答逐维评分 → {"准确性": 1-5, ..., "rationale": str}。

    prompt 要求严格 JSON（键=四个中文维度+rationale）；宽容解析；
    维度缺失补 3 分；LLM 异常或输出不可解析时全部 3 分且 rationale="judge_failed"。
    """
    prompt = _SCORE_PROMPT.format(question=question, references=references, answer=answer)
    raw = ""
    try:
        resp = judge_llm.invoke([{"role": "user", "content": prompt}])
        raw = str(getattr(resp, "content", None) or resp)
    except Exception:
        raw = ""
    parsed = _parse_json_object(raw) if raw else None
    if not isinstance(parsed, dict):
        result = {dim: 3 for dim in JUDGE_DIMS}
        result["rationale"] = "judge_failed"
        return result
    result = {dim: _score_of(parsed.get(dim)) for dim in JUDGE_DIMS}
    result["rationale"] = str(parsed.get("rationale", ""))
    return result


def _pair_prompt(question: str, answer_a: str, answer_b: str, b_first: bool) -> str:
    """构造对比 prompt：标签恒定（A/B 绑定身份），仅交换展示顺序以探测位置偏置。"""
    block_a = f"【回答 A】\n{answer_a}"
    block_b = f"【回答 B】\n{answer_b}"
    blocks = "\n\n".join([block_b, block_a] if b_first else [block_a, block_b])
    return _PAIR_PROMPT.format(question=question, blocks=blocks)


def _verdict_of(resp) -> Optional[str]:
    """从裁判回复中提取首个独立词元 a/b/tie；无有效判定返回 None。"""
    try:
        raw = str(getattr(resp, "content", None) or resp)
    except Exception:
        return None
    for token in re.findall(r"[A-Za-z]+", raw):
        low = token.lower()
        if low in ("a", "b", "tie"):
            return low
    return None


def pairwise_winrate(question: str, answer_a: str, answer_b: str,
                     judge_llm, swap: bool = True) -> str:
    """成对对比：返回 "a" | "b" | "tie"。

    swap=True 时先 A 前后 B 前（展示顺序交换、标签不变）各判一次；
    两次胜者一致→该胜者；不一致或任一次无效→tie。swap=False 仅判一次。
    """
    def _ask(b_first: bool) -> Optional[str]:
        try:
            resp = judge_llm.invoke(
                [{"role": "user", "content": _pair_prompt(question, answer_a, answer_b, b_first)}]
            )
        except Exception:
            return None
        return _verdict_of(resp)

    first = _ask(False)
    if not swap:
        return first or "tie"
    second = _ask(True)
    if first is None or second is None or first != second:
        return "tie"
    return first


def cohen_kappa(votes_a: Sequence[str], votes_b: Sequence[str]) -> float:
    """Cohen's κ：两评标者一致性（标签集合取两者并集）。

    κ = (po - pe) / (1 - pe)；分母为 0（双方都只投同一标签）时定义返回 1.0；
    空输入返回 0.0；两序列长度不一致抛 ValueError。
    """
    if len(votes_a) != len(votes_b):
        raise ValueError("votes_a 与 votes_b 长度必须一致")
    n = len(votes_a)
    if n == 0:
        return 0.0
    count_a, count_b = Counter(votes_a), Counter(votes_b)
    po = sum(1 for x, y in zip(votes_a, votes_b) if x == y) / n
    labels = set(count_a) | set(count_b)
    pe = sum((count_a[label] / n) * (count_b[label] / n) for label in labels)
    denom = 1.0 - pe
    if abs(denom) < 1e-12:
        return 1.0
    return (po - pe) / denom
