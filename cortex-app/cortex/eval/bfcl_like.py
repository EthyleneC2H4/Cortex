"""BFCL 式工具调用准确率匹配器（纯函数，零 LLM 依赖）。

参照 Berkeley Function-Calling Leaderboard 的 AST 匹配思路：
工具名精确相等 + 参数集合完全一致（多余实参视为不等）+ 值递归等价
（int/float 数值互通、bool 严格、str 精确）。另提供 trace JSONL 解析
与批量准确率计算。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Sequence, Tuple

CallPair = Tuple[str, dict]


def value_eq(a: Any, b: Any) -> bool:
    """递归值等价：int/float 数值互通（1==1.0），bool 严格（True!=1），str 精确，list/dict 递归。"""
    # bool 是 int 的子类，必须最先单独判断，避免 True == 1 被误判为相等
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b  # Python 跨类型数值比较：1 == 1.0 为真
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(value_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(value_eq(a[k], b[k]) for k in a)
    return a == b


def args_eq(expected: dict, actual: dict) -> bool:
    """参数集等价：键集合必须完全一致（多余实参=不等），顺序无关，值递归 value_eq。"""
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return False
    if set(expected.keys()) != set(actual.keys()):
        return False
    return all(value_eq(expected[k], actual[k]) for k in expected)


def ast_match(expected_name: str, expected_args: dict,
              actual_name: str, actual_args: dict) -> bool:
    """单次调用的 AST 匹配：工具名相同且参数集等价。"""
    return expected_name == actual_name and args_eq(expected_args, actual_args)


def multi_call_match(expected: Sequence[CallPair],
                     actual: Sequence[CallPair]) -> bool:
    """多重调用匹配：数量相等且按集合贪心指派存在一一对应（顺序无关）。"""
    if len(expected) != len(actual):
        return False
    used = [False] * len(actual)
    for exp_name, exp_args in expected:
        for j, (act_name, act_args) in enumerate(actual):
            if not used[j] and ast_match(exp_name, exp_args, act_name, act_args):
                used[j] = True
                break
        else:  # 当前金标调用找不到任何未占用且匹配的实调用
            return False
    return True


def parse_trace_tool_calls(jsonl_path) -> List[CallPair]:
    """从观测 trace JSONL 中提取 event=="tool_call" 的调用序列；坏行静默跳过。

    兼容两种载体形态：payload 直接平铺在行上，或收纳在行内 "payload" 键下；
    名称键兼容 ``tool_name``/``name``，参数键兼容 ``parameters``/``arguments``；
    arguments 为 JSON 字符串时尝试解析（容错），解析失败视为坏行跳过。
    """
    calls: List[CallPair] = []
    lines = Path(jsonl_path).read_text(encoding="utf-8").splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue  # 坏 JSON 行跳过
        if not isinstance(obj, dict) or obj.get("event") != "tool_call":
            continue
        payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else obj
        name = payload.get("tool_name", payload.get("name"))
        raw_args = payload.get("parameters", payload.get("arguments"))
        if not isinstance(name, str) or not name:
            continue
        if isinstance(raw_args, str):  # arguments 可能是 JSON 字符串
            try:
                raw_args = json.loads(raw_args)
            except json.JSONDecodeError:
                continue
        if raw_args is None:
            raw_args = {}
        if not isinstance(raw_args, dict):
            continue
        calls.append((name, raw_args))
    return calls


def _to_calls(side: Any) -> List[CallPair]:
    """把「单调用 (name, args)」「{"name":..., "args":...}」或「调用列表」统一成 [(name, args), ...]。"""
    if isinstance(side, dict) and "name" in side:
        name, args = side["name"], side.get("args", {})
        if isinstance(name, list):  # 多调用场景：name 列表 + args 列表并行
            arg_list = args if isinstance(args, list) else [dict(args)] * len(name)
            if len(arg_list) != len(name):
                raise ValueError("多调用金标中 name 与 args 数量不一致")
            return list(zip(name, arg_list))
        return [(name, args if isinstance(args, dict) else {})]
    if (isinstance(side, tuple) and len(side) == 2 and isinstance(side[0], str)):
        return [(side[0], side[1] if isinstance(side[1], dict) else {})]  # 单调用元组
    if isinstance(side, list):
        calls: List[CallPair] = []
        for item in side:
            if isinstance(item, dict) and "name" in item:
                calls.extend(_to_calls(item))
            else:
                name, args = item
                calls.append((name, args if isinstance(args, dict) else {}))
        return calls
    raise TypeError(f"无法识别的调用形态: {type(side)!r}")


def tool_call_accuracy(cases: Sequence[dict]) -> float:
    """批量准确率：cases 每项 {"expected": 金标调用(列表), "actual": 实际调用(列表)}。

    返回命中比例；空用例表返回 0.0。单调用与多重调用自动识别。
    """
    if not cases:
        return 0.0
    hit = sum(
        1
        for case in cases
        if multi_call_match(_to_calls(case["expected"]), _to_calls(case["actual"]))
    )
    return hit / len(cases)
