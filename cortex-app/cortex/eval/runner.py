"""评测运行器：bfcl / gaia / judge 三套件 + 消融开关 + 断点续跑。

用法：
    python -m cortex.eval.run --suite all
    python -m cortex.eval.run --suite gaia --no-memory

消融开关：no_rerank 会覆写 cfg.enable_rerank（唯一的运行时开关）；no_memory /
single_agent 尚无对应全局配置（记忆服务恒装配、编排模式由请求级 --mode 决定），
仅如实记录在报告头。且现有三套件均为纯 LLM 评测，不消费检索/记忆/多代理路径
——待套件接入相应链路后覆写才真正影响指标。报告落 ``data/eval/<ts>/report.md``。
按条目 checkpoint（data/eval/<ts>/checkpoint.json）应对免费档限流中断；
checkpoint 写入为「临时文件 + 原子改名」，中断不会留下半截 JSON。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

DATASETS_DIR = Path(__file__).resolve().parent / "datasets"


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


@dataclass
class RunOptions:
    suites: list[str] = field(default_factory=lambda: ["bfcl", "gaia", "judge"])
    no_rerank: bool = False
    no_memory: bool = False
    single_agent: bool = False


@dataclass
class Checkpoint:
    path: Path
    done: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.path.exists():
            try:
                self.done = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.done = {}

    def is_done(self, key: str) -> bool:
        return key in self.done

    def mark(self, key: str, value) -> None:
        self.done[key] = value
        # 原子写：临时文件 + 同目录 rename。限流中断若发生在 write_text 中途，
        # 整个 checkpoint 会因半截 JSON 作废、全部进度归零；rename 不可分割。
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.done, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)


# --------------------------------------------------------------------- #
# 套件实现                                                                 #
# --------------------------------------------------------------------- #
_TOOL_TYPE_MAP = {str: "string", int: "integer", float: "number", bool: "boolean",
                  list: "array", dict: "object"}


def _tool_schema(name: str, args: dict) -> dict:
    """从金标 args 反推最小工具 schema（类型取金标值类型）。"""
    props = {}
    for k, v in args.items():
        props[k] = {"type": _TOOL_TYPE_MAP.get(type(v), "string")}
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"评测工具 {name}",
            "parameters": {"type": "object", "properties": props,
                           "required": list(args.keys())},
        },
    }


def _extract_tool_calls(resp) -> list:
    """从响应提取工具调用并归一为 [(name, args_dict)]。

    兼容两种形态：框架适配器返回的**原生 OpenAI 响应**
    （choices[0].message.tool_calls，function.name/arguments）与
    已扁平化的对象（resp.tool_calls，c.name/c.arguments）。

    参数 JSON 解析失败或非 object 时置 **None 哨兵**（而非静默变 {}）：
    None 在匹配器中恒不等，防止损坏的参数与「无参金标」假匹配。
    """
    calls = getattr(resp, "tool_calls", None)
    if calls is None:
        choices = getattr(resp, "choices", None) or []
        message = getattr(choices[0], "message", None) if choices else None
        calls = getattr(message, "tool_calls", None) if message else None
    out = []
    for c in calls or []:
        fn = getattr(c, "function", None)
        name = (getattr(fn, "name", None) or getattr(c, "name", "") or "")
        raw = getattr(fn, "arguments", None) if fn is not None else getattr(c, "arguments", None)
        try:
            args = json.loads(raw) if isinstance(raw, str) and raw else (raw or {})
        except (json.JSONDecodeError, TypeError):
            args = None  # 解析失败 ≠ 空参数：哨兵使其不可匹配
        if args is not None and not isinstance(args, dict):
            args = None  # 裸字符串/数组/标量等非 object JSON 同样不可信
        out.append((name, args))
    return out


def _bootstrap_ci(correct: list, *, iterations: int = 1000,
                  seed: int = 42) -> tuple[float, float]:
    """比例型指标的自展 95% 分位置信区间。固定种子保证报告可复现。

    小样本（30 条）下点估计不稳，CI 让"0.90 与 0.87 的差异是否显著"有据可依。
    """
    if not correct:
        return (0.0, 0.0)
    vals = [1.0 if c else 0.0 for c in correct]
    n = len(vals)
    rng = random.Random(seed)
    means = sorted(
        sum(vals[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(iterations)
    )
    lo = means[max(0, int(0.025 * iterations))]
    hi = means[min(iterations - 1, int(0.975 * iterations))]
    return (round(lo, 4), round(hi, 4))


def run_bfcl(rows, llm, ckpt: Checkpoint) -> dict:
    from .bfcl_like import ast_match, multi_call_match

    def _golds(row):
        """行 → [(name, args)]；多调用行 name/args 为平行列表。"""
        names = row["name"] if isinstance(row["name"], list) else [row["name"]]
        argss = row.get("args", {})
        if not isinstance(argss, list):
            argss = [argss]
        pairs = list(zip(names, argss))
        return [(n, a if isinstance(a, dict) else {}) for n, a in pairs]

    results = {}
    for i, row in enumerate(rows):
        key = f"bfcl/{i}"
        if ckpt.is_done(key):
            results[key] = ckpt.done[key]
            continue
        golds = _golds(row)
        instruction = row.get("instruction") or f"请调用合适的工具完成：{row['name']}"
        try:
            from ..llm.resilience import resilient_invoke_with_tools

            schemas = [_tool_schema(n, a) for n, a in golds]
            resp = resilient_invoke_with_tools(llm, [{"role": "user", "content": instruction}],
                                               schemas, tool_choice="auto")
            actuals = _extract_tool_calls(resp)
            if len(golds) == 1:
                ok = bool(actuals) and ast_match(golds[0][0], golds[0][1], *actuals[0])
            else:
                ok = multi_call_match(golds, actuals)
            results[key] = {"correct": ok, "actual": actuals}
        except Exception as exc:
            results[key] = {"correct": False, "error": str(exc)}
        ckpt.mark(key, results[key])
    n = len(results) or 1
    flags = [bool(r.get("correct")) for r in results.values()]
    return {
        "n": len(results),
        "accuracy": round(sum(flags) / n, 4),
        "accuracy_ci95": str(_bootstrap_ci(flags)),
        "detail": results,
    }


def run_gaia(rows, strong_llm, ckpt: Checkpoint) -> dict:
    from .gaia_like import level_metrics, quasi_exact_match

    detail = []
    for i, row in enumerate(rows):
        key = f"gaia/{i}"
        answer = ""
        if ckpt.is_done(key):
            answer = ckpt.done[key].get("pred", "")
        else:
            try:
                from ..llm.resilience import resilient_invoke

                resp = resilient_invoke(
                    strong_llm,
                    [{"role": "user", "content":
                      f"直接回答下面的问题，只给最终答案本身（不要解释、不要句号结尾）：\n{row['question']}"}],
                )
                answer = str(getattr(resp, "content", "")).strip()
            except Exception as exc:
                answer = f"(error: {exc})"
            ckpt.mark(key, {"pred": answer})
        correct = quasi_exact_match(answer, row["answer"])
        detail.append({"id": row.get("id", key), "level": row.get("level", "L1"),
                       "correct": correct, "pred": answer[:120], "gold": row["answer"]})
    levels = level_metrics([{"level": d["level"], "correct": d["correct"]} for d in detail])
    flags = [bool(d["correct"]) for d in detail]
    overall_n = len(detail) or 1
    return {"n": len(detail),
            "quasi_exact_match": round(sum(flags) / overall_n, 4),
            "quasi_exact_match_ci95": str(_bootstrap_ci(flags)),
            "levels": levels, "detail": detail}


def run_judge(rows, judge_llm, ckpt: Checkpoint) -> dict:
    from .judge import cohen_kappa, pairwise_judgement

    detail, agree, swap_flags = [], [], []
    for i, row in enumerate(rows):
        key = f"judge/{i}"
        if ckpt.is_done(key):
            saved = ckpt.done[key]
            verdict, swap_agree = saved.get("verdict", "tie"), saved.get("swap_agree")
        else:
            verdict, swap_agree = pairwise_judgement(
                row["question"], row["answer_a"], row["answer_b"], judge_llm)
            ckpt.mark(key, {"verdict": verdict, "swap_agree": swap_agree})
        gold = row.get("better", "tie")
        agree.append(verdict == gold)
        if swap_agree is not None:
            swap_flags.append(swap_agree)
        detail.append({"id": row.get("id", key), "judge": verdict, "gold": gold,
                       "swap_agree": swap_agree})
    gold_votes = [d["gold"] for d in detail]
    judge_votes = [d["judge"] for d in detail]
    n = len(detail) or 1
    return {
        "n": len(detail),
        "agreement_with_gold": round(sum(agree) / n, 4),
        # 交换一致率：位置交换双评中两次判定一致的占比——低值即裁判位置偏置的
        # 直接量化（这些条目被记为 tie 是偏置所致而非真的难分伯仲）
        "swap_agreement_rate": (round(sum(swap_flags) / len(swap_flags), 4)
                                if swap_flags else None),
        "win_rate_a": round(sum(1 for v in judge_votes if v == "a") / n, 4),
        "kappa_vs_gold": round(cohen_kappa(judge_votes, gold_votes), 4),
        "detail": detail,
    }


# --------------------------------------------------------------------- #
def run_eval(cfg, *, strong_llm=None, weak_llm=None, options: Optional[RunOptions] = None,
             progress: Optional[Callable[[str], None]] = None) -> Path:
    """跑选定套件，写 report.md，返回报告路径。"""
    options = options or RunOptions()

    def say(msg: str) -> None:
        if progress:
            progress(msg)

    if strong_llm is None or weak_llm is None:
        from ..llm.factory import build_llms
        from ..llm.model_resolver import resolve_model_id

        mid = resolve_model_id(cfg)
        strong_llm, weak_llm = build_llms(cfg, model_id=mid)

    # 消融覆写：仅 enable_rerank 存在运行时开关；其余两旗标无全局配置可覆写，
    # 只随报告头记录（见模块 docstring 的诚实说明）
    if options.no_rerank:
        cfg.enable_rerank = False

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = cfg.data_dir / "eval" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = Checkpoint(out_dir / "checkpoint.json")

    sections = []
    if "bfcl" in options.suites:
        say("bfcl: 开始")
        rows = _load_jsonl(DATASETS_DIR / "bfcl.jsonl")
        res = run_bfcl(rows, weak_llm, ckpt)
        sections.append(("BFCL 工具调用准确率", res))
        say(f"bfcl: accuracy={res['accuracy']}")
    if "gaia" in options.suites:
        say("gaia: 开始")
        rows = _load_jsonl(DATASETS_DIR / "gaia.jsonl")
        res = run_gaia(rows, strong_llm, ckpt)
        sections.append(("GAIA 准确匹配（分级）", res))
        say(f"gaia: qem={res['quasi_exact_match']}")
    if "judge" in options.suites:
        say("judge: 开始")
        rows = _load_jsonl(DATASETS_DIR / "judge_pairs.jsonl")
        res = run_judge(rows, strong_llm, ckpt)
        sections.append(("LLM-as-Judge（与人工金标一致性）", res))
        say(f"judge: agreement={res['agreement_with_gold']}")

    report = _render_report(options, sections)
    out = out_dir / "report.md"
    out.write_text(report, encoding="utf-8")
    say(f"报告已写入 {out}")
    return out


def _render_report(options: RunOptions, sections) -> str:
    ablated = options.no_rerank or options.no_memory or options.single_agent
    lines = [
        "# Cortex 评测报告",
        "",
        f"- 时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 消融设置：no_rerank={options.no_rerank} no_memory={options.no_memory} "
        f"single_agent={options.single_agent}",
    ]
    if ablated:
        lines.append("- 说明：三套件为纯 LLM 评测，暂不消费检索/记忆/多代理路径；"
                     "no_rerank 已覆写配置，其余开关仅记录在案")
    lines.append("")
    for title, res in sections:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("|---|---|")
        for k, v in res.items():
            if isinstance(v, (int, float, str)):
                lines.append(f"| {k} | {v} |")
        for lk, lv in res.get("levels", {}).items():
            lines.append(f"| {lk} | {lv.get('correct', 0)}/{lv.get('n', 0)} |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m cortex.eval.run", description="Cortex 评测")
    parser.add_argument("--suite", default="all",
                        help="all | bfcl | gaia | judge（逗号分隔可多选）")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--single-agent", action="store_true")
    args = parser.parse_args(argv)

    suites = ["bfcl", "gaia", "judge"] if args.suite == "all" else \
        [s.strip() for s in args.suite.split(",") if s.strip()]
    from ..config import load_config

    cfg = load_config()
    opts = RunOptions(suites=suites, no_rerank=args.no_rerank,
                      no_memory=args.no_memory, single_agent=args.single_agent)
    out = run_eval(cfg, options=opts, progress=print)
    print(f"\n📄 {out}")


if __name__ == "__main__":
    main()
