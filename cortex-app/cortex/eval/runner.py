"""评测运行器：bfcl / gaia / judge 三套件 + 消融开关 + 断点续跑。

用法：
    python -m cortex.eval.run --suite all
    python -m cortex.eval.run --suite gaia --no-memory

消融开关在构建应用前改配置/服务组合；报告落 ``data/eval/<ts>/report.md``。
按条目 checkpoint（data/eval/<ts>/checkpoint.json）应对免费档限流中断。
"""

from __future__ import annotations

import argparse
import json
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
        self.path.write_text(json.dumps(self.done, ensure_ascii=False), encoding="utf-8")


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
            args = {}
        if not isinstance(args, dict):
            args = {}
        out.append((name, args))
    return out


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
    correct = sum(1 for r in results.values() if r.get("correct"))
    return {"n": len(results), "accuracy": round(correct / n, 4), "detail": results}


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
    overall_n = len(detail) or 1
    overall_c = sum(1 for d in detail if d["correct"])
    return {"n": len(detail), "quasi_exact_match": round(overall_c / overall_n, 4),
            "levels": levels, "detail": detail}


def run_judge(rows, judge_llm, ckpt: Checkpoint) -> dict:
    from .judge import cohen_kappa, pairwise_winrate

    detail, agree = [], []
    for i, row in enumerate(rows):
        key = f"judge/{i}"
        if ckpt.is_done(key):
            verdict = ckpt.done[key]["verdict"]
        else:
            verdict = pairwise_winrate(
                row["question"], row["answer_a"], row["answer_b"], judge_llm)
            ckpt.mark(key, {"verdict": verdict})
        gold = row.get("better", "tie")
        agree.append(verdict == gold)
        detail.append({"id": row.get("id", key), "judge": verdict, "gold": gold})
    gold_votes = [d["gold"] for d in detail]
    judge_votes = [d["judge"] for d in detail]
    n = len(detail) or 1
    return {
        "n": len(detail),
        "agreement_with_gold": round(sum(agree) / n, 4),
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
    lines = [
        "# Cortex 评测报告",
        "",
        f"- 时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 消融设置：no_rerank={options.no_rerank} no_memory={options.no_memory} "
        f"single_agent={options.single_agent}",
        "",
    ]
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
