"""研究任务双派发：pipeline（确定性五段链）/ explore（orchestrator 自主规划）。

pipeline：Gather→Research→Write→Critique→(≤1 次修订)，阶段间以 Findings
JSON 传递（子代理无历史回放，任务串必须自包含）。
explore：TaskTool 已注册 orchestrator registry，由其自主派发子代理。
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from .findings import parse_findings
from .subagent_factory import dispatch_subagent
from ..tools.filters import filter_for_role

_NO_REVISION_MARK = "无需改进"


@dataclass
class PipelineTrace:
    """逐段执行记录（评测与调试共用）。"""

    steps: List[dict] = field(default_factory=list)

    def add(self, stage: str, **payload) -> None:
        self.steps.append({"stage": stage, "ts": time.time(), **payload})

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(s, ensure_ascii=False) for s in self.steps)


@dataclass
class ResearchResult:
    topic: str
    mode: str                    # pipeline | explore
    report_md: str
    stages: dict                 # {"gather": Findings, "research": Findings, ...}
    revised: bool
    duration_seconds: float


class ResearchPipeline:
    """确定性流水线；agent_set 来自 builders.build_agents。

    每阶段套**墙钟看门狗**：免费档经代理偶发 HTTP 读挂死（框架流式读
    无法触发超时），守护线程超时后放弃该阶段并继续；被放弃阶段可能
    未走 run_as_subagent 的 finally 还原，用 ``registry_repair`` 兜底修复。
    """

    def __init__(self, agent_set, *, knowledge=None, memory=None, context_engine=None,
                 traces_dir: Optional[Path] = None,
                 registry_repair: Optional[Callable[[], None]] = None,
                 stage_timeout: float = 0.0) -> None:
        self.agents = agent_set
        self.knowledge = knowledge
        self.memory = memory
        self.context_engine = context_engine
        self.traces_dir = Path(traces_dir) if traces_dir else None
        self.registry_repair = registry_repair
        # 0 = 不限时；默认取 CORTEX_STAGE_TIMEOUT（秒）
        self.stage_timeout = stage_timeout or float(os.getenv("CORTEX_STAGE_TIMEOUT", "900"))
        self._repair_pending = False

    # ---------------------------------------------------------------- #
    def _run_with_timeout(self, fn: Callable[[], object], label: str):
        """(fn, 超时) → (结果, timed_out)；超时放弃守护线程并标记需修注册表。"""
        timeout = getattr(self, "stage_timeout", 0.0) or 0.0
        if timeout <= 0:
            return fn(), False
        out: dict = {}

        def _worker():
            try:
                out["value"] = fn()
            except BaseException as exc:  # noqa: BLE001 —— 线程内异常带回主线程
                out["error"] = exc

        t = threading.Thread(target=_worker, daemon=True, name=f"stage-{label}")
        t.start()
        t.join(timeout)
        if t.is_alive():
            self._repair_pending = True
            return None, True
        if "error" in out:
            raise out["error"]
        return out.get("value"), False

    def _maybe_repair_registry(self) -> None:
        if self._repair_pending and self.registry_repair is not None:
            try:
                self.registry_repair()
            except Exception:  # noqa: BLE001 —— 修复尽力而为
                pass
            self._repair_pending = False

    # ---------------------------------------------------------------- #
    def _evidence_block(self, topic: str) -> str:
        if self.knowledge is None:
            return ""
        try:
            hits = self.knowledge.search_for_context(topic, limit=5)
        except Exception:
            hits = []
        if not hits:
            return ""
        blocks = [f"[{h['chunk_id']}] ({h.get('source', '')})\n{h['text']}" for h in hits]
        return "\n\n".join(blocks)

    def _gssc_blocks(self, topic: str) -> tuple[str, str]:
        """经 ContextEngine（GSSC）取 (证据块, 记忆块)；引擎缺席/故障退化直查。"""
        if self.context_engine is None:
            return self._evidence_block(topic), ""
        try:
            built = self.context_engine.build_system_prompt(topic, history=[])
            ev = [p.text for p in built.packets if p.type == "knowledge_base"]
            mem = [p.text for p in built.packets if p.type == "related_memory"]
            return "\n\n".join(ev), "\n".join(mem)
        except Exception:
            return self._evidence_block(topic), ""

    def run(self, topic: str) -> ResearchResult:
        t0 = time.time()
        trace = PipelineTrace()
        stages: dict = {}

        # ---- 1. Gather（本地检索） ---------------------------------- #
        gather_task = (
            f"研究主题：{topic}\n\n"
            "请从本地知识库与记忆中检索与主题相关的证据，"
            "输出 Findings JSON（conclusions/supporting/citations/confidence/conflicts）。"
        )
        g_raw, g_timeout = self._run_with_timeout(
            lambda: dispatch_subagent(self.agents.retriever, gather_task, role="retriever"),
            "gather")
        if g_timeout:
            g_raw = {"success": False, "summary": "（gather 阶段执行超时）", "tools_used": []}
        findings_g = parse_findings(g_raw["summary"])
        stages["gather"] = findings_g
        trace.add("gather", success=g_raw["success"], timed_out=g_timeout,
                  tools=g_raw["tools_used"], confidence=findings_g.confidence)

        # ---- 2. Research（联网补充，失败容忍） ----------------------- #
        research_task = f"研究主题：{topic}\n已知线索：{json.dumps(findings_g.model_dump(exclude={'raw'}), ensure_ascii=False)[:1500]}\n请联网补充外部资料并输出 Findings JSON；无法联网时在 conflicts 中说明。"
        r_raw, r_timeout = self._run_with_timeout(
            lambda: dispatch_subagent(self.agents.researcher, research_task, role="researcher"),
            "research")
        if r_timeout:
            r_raw = {"success": False, "summary": "（research 阶段执行超时）", "tools_used": []}
        findings_r = parse_findings(r_raw["summary"])
        stages["research"] = findings_r
        trace.add("research", success=r_raw["success"], timed_out=r_timeout,
                  tools=r_raw["tools_used"])

        # ---- 3. Write（GSSC 组装证据+记忆） --------------------------- #
        evidence, memory_block = self._gssc_blocks(topic)
        write_task = (
            f"研究主题：{topic}\n\n"
            f"## 本地检索发现\n{findings_g.model_dump_json()}\n"
            f"## 联网研究发现\n{findings_r.model_dump_json()}\n"
            + (f"## 知识库证据片段\n{evidence}\n" if evidence else "")
            + (f"## 记忆上下文\n{memory_block}\n" if memory_block else "")
            + "\n请撰写结构化中文 Markdown 研究报告（结论先行、引用标记 [doc_id/chunk_id]、"
              "未解问题收尾；正文控制在 1200 字以内，宁精勿长）。"
        )
        draft, w_timeout = self._run_with_timeout(
            lambda: self._run_agent_text(self.agents.writer, write_task, role="writer"), "write")
        if w_timeout:
            draft = f"# 报告生成失败\n\nwrite 阶段执行超时（>{int(self.stage_timeout)}s）"
        trace.add("write", chars=len(draft), timed_out=w_timeout)

        # ---- 4. Critique --------------------------------------------- #
        critique_task = f"审校以下研究报告（主题：{topic}）：\n\n{draft[:6000]}\n\n给出修订清单或回复'{_NO_REVISION_MARK}'。"
        critique, c_timeout = self._run_with_timeout(
            lambda: self._run_agent_text(self.agents.critic, critique_task, role="critic"),
            "critique")
        if c_timeout:
            critique = "（critique 阶段执行超时，跳过修订）"
        trace.add("critique", needs_revision=_NO_REVISION_MARK not in critique,
                  timed_out=c_timeout)

        # ---- 5. Revise（≤1 次） -------------------------------------- #
        revised = False
        if not c_timeout and _NO_REVISION_MARK not in critique:
            revise_task = (
                f"根据审校意见修订报告（主题：{topic}）。保持原有引用标记，修正指出的问题。\n\n"
                f"## 原稿\n{draft[:5000]}\n\n## 审校意见\n{critique[:2000]}"
            )
            draft2, v_timeout = self._run_with_timeout(
                lambda: self._run_agent_text(self.agents.writer, revise_task, role="writer"),
                "revise")
            if not v_timeout and draft2.strip() and not draft2.startswith("# 报告生成失败"):
                draft, revised = draft2, True  # 修订成功才采纳（异常兜底文本不覆盖原稿）
            trace.add("revise", adopted=revised, timed_out=v_timeout)

        self._maybe_repair_registry()
        duration = time.time() - t0
        self._flush_trace(trace, topic)
        return ResearchResult(topic=topic, mode="pipeline", report_md=draft,
                              stages={"gather": stages["gather"], "research": stages["research"]},
                              revised=revised, duration_seconds=duration)

    # ---------------------------------------------------------------- #
    def _run_agent_text(self, agent, task: str, role: str) -> str:
        """以子代理模式跑文本阶段（writer/critic），套角色过滤器取正文。

        必须走 run_as_subagent + tool_filter：直接 .run() 会看到共享注册表
        全部工具（含 Task），免费档模型可能自行派发嵌套子代理导致失控。
        免费档偶发传输截断（响应体解析失败）→ 整次重试一次再放弃；
        writer 末条回复若不是报告正文（无 Markdown 标题）→ 带纠正指令重问一次。
        """
        last_err = "unknown"
        for attempt in range(2):
            try:
                out = agent.run_as_subagent(task, tool_filter=filter_for_role(role),
                                            return_summary=False)
                if out.get("success"):
                    text = str(out.get("result", ""))
                    if role != "writer" or attempt == 1 or text.lstrip().startswith("#"):
                        return text
                    last_err = "回复是过程叙述而非报告正文"
                    task += (
                        "\n\n【纠正】你上一条回复是过程叙述，不是最终产出。"
                        "请只输出完整报告 Markdown 正文（以 '# ' 标题开头），"
                        "不要任何工作汇报或过渡语。"
                    )
                    continue
                last_err = str((out.get("metadata") or {}).get("error") or "执行失败")
            except Exception as exc:  # noqa: BLE001 —— 单阶段失败不拖垮流水线
                last_err = str(exc)
        return f"# 报告生成失败\n\n代理执行异常：{last_err}"

    def _flush_trace(self, trace: PipelineTrace, topic: str) -> None:
        if not self.traces_dir:
            return
        try:
            self.traces_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d-%H%M%S")
            path = self.traces_dir / f"pipeline-{ts}-{uuid.uuid4().hex[:6]}.jsonl"
            path.write_text(json.dumps({"topic": topic}, ensure_ascii=False) + "\n" + trace.to_jsonl(),
                            encoding="utf-8")
        except OSError:
            pass


class ExplorationMode:
    """orchestrator 自主规划模式（PlanSolve + TaskTool 派发）。"""

    def __init__(self, orchestrator) -> None:
        self.orchestrator = orchestrator

    def run(self, topic: str) -> ResearchResult:
        t0 = time.time()
        try:
            text = str(self.orchestrator.run(
                f"请完成以下研究任务并输出结构化中文 Markdown 报告（带 [doc_id/chunk_id] 引用）：\n{topic}"
            ))
        except Exception as exc:
            text = f"# 研究失败\n\n编排器异常：{exc}"
        return ResearchResult(topic=topic, mode="explore", report_md=text,
                              stages={}, revised=False, duration_seconds=time.time() - t0)


def run_research(topic: str, *, mode: str = "pipeline", agent_set=None, **pipeline_kw) -> ResearchResult:
    """统一入口：mode=pipeline | explore。"""
    if mode == "explore":
        if agent_set is None:
            raise ValueError("explore 模式需要 agent_set")
        return ExplorationMode(agent_set.orchestrator).run(topic)
    pipe = ResearchPipeline(agent_set, **pipeline_kw)
    return pipe.run(topic)
