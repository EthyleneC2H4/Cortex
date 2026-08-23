"""P5 单元测试：Findings 解析 + 流水线编排（全 mock 代理）。"""

from __future__ import annotations

import json
import time

import pytest

from cortex.agents.dispatch import ExplorationMode, ResearchPipeline
from cortex.agents.findings import parse_findings

pytestmark = pytest.mark.unit


class TestParseFindings:
    def test_plain_json(self):
        raw = json.dumps({"conclusions": ["A"], "supporting": ["s"], "confidence": 0.9,
                          "citations": [{"doc_id": "d", "chunk_id": "d:0"}]})
        f = parse_findings(raw)
        assert f.conclusions == ["A"] and f.confidence == 0.9
        assert f.citations[0].chunk_id == "d:0"

    def test_fenced_json_with_prose(self):
        raw = '结果如下：\n```json\n{"conclusions": "单条转列表", "confidence": 2}\n```'
        f = parse_findings(raw)
        assert f.conclusions == ["单条转列表"]
        assert f.confidence == 1.0  # clamp 到 [0,1]

    def test_string_citation_coerced(self):
        raw = '{"citations": ["d:3"]}'
        f = parse_findings(raw)
        assert f.citations[0].chunk_id == "d:3"

    def test_garbage_falls_back_to_raw(self):
        f = parse_findings("完全不是 JSON 的一段结论文字。")
        assert f.raw.startswith("完全不是")
        assert f.conclusions and f.confidence == 0.3

    def test_empty(self):
        assert parse_findings("").raw == "(空回复)"

    def test_summary_wrapped_json(self):
        # 框架 summary 外壳（任务/结果前缀 + 后置元信息）不应破坏解析
        raw = '任务: 研究主题X\n\n请输出 Findings JSON。\n结果: {"conclusions": ["RRF 无需归一化"], "confidence": 0.9}\n步数: 3\n耗时: 12秒'
        f = parse_findings(raw)
        assert f.conclusions == ["RRF 无需归一化"] and f.confidence == 0.9

    def test_truncated_json_salvage(self):
        # 生成上限截断的 JSON：按字段正则抢救
        f = parse_findings('前言。{"conclusions": ["完整的第一条", "截断的第')
        assert f.conclusions[0] == "完整的第一条"
        assert f.raw == "(截断抢救)"

    def test_salvage_preserves_cjk(self):
        f = parse_findings('{"conclusions": ["中文结论不受转义影响"], "confidence": 0.7')
        assert f.conclusions == ["中文结论不受转义影响"]
        assert abs(f.confidence - 0.7) < 1e-9

    def test_clamp_long(self):
        big = {"conclusions": ["x" * 3000] * 5, "supporting": ["y" * 3000] * 5}
        f = parse_findings(json.dumps(big))
        total = sum(len(x) for x in f.conclusions + f.supporting)
        assert total <= 2000 * 2.5 + 2000


# --------------------------------------------------------------------- #
class MockSub:
    """run_as_subagent 型代理（retriever/researcher）；return_summary=False 契约。"""

    def __init__(self, reply):
        self.reply = reply

    def run_as_subagent(self, task, tool_filter=None, return_summary=False, max_steps_override=None):
        return {"success": True, "result": self.reply, "metadata": {"tools_used": ["knowledge_search"]}}


class MockText:
    """文本型代理（writer/critic 走 run_as_subagent；orchestrator 走 run）。"""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def _next(self):
        self.calls.append(self._current_task)
        return self.replies.pop(0) if self.replies else "默认回复"

    def run(self, task):
        self._current_task = task
        return self._next()

    def run_as_subagent(self, task, tool_filter=None, return_summary=False,
                        max_steps_override=None):
        assert tool_filter is not None, "文本阶段必须套角色过滤器（防 Task 逃逸）"
        self._current_task = task
        return {"success": True, "result": self._next(),
                "metadata": {"tools_used": [], "error": None}}


@pytest.fixture
def agent_set():
    g = MockSub('{"conclusions":["RRF 是默认融合方案"],"confidence":0.85,'
                '"citations":[{"doc_id":"abc","chunk_id":"abc:0"}]}')
    r = MockSub('{"conclusions":[],"conflicts":["无法联网"],"confidence":0.2}')
    w = MockText(["# 报告草稿\nRRF 融合两路信号 [abc/abc:0]。",
                  "# 修订版报告\n更严谨的内容 [abc/abc:0]。"])  # 第 2 次调用=按审校意见修订
    c = MockText(["存在未标注引用的问题，请修改。"])
    return type("S", (), {"retriever": g, "researcher": r, "writer": w, "critic": c,
                          "orchestrator": None})()


class TestPipeline:
    def test_full_pipeline_with_revision(self, agent_set, tmp_path):
        pipe = ResearchPipeline(agent_set, traces_dir=tmp_path / "traces")
        result = pipe.run("混合检索策略对比")

        assert result.mode == "pipeline"
        assert result.revised is True                      # critic 挑了问题 → 触发修订
        assert result.report_md.startswith("# 修订版报告")
        assert "[abc/abc:0]" in result.report_md           # 引用保留
        assert result.stages["gather"].conclusions == ["RRF 是默认融合方案"]
        # 调用顺序：gather → research → write → critique → revise
        assert len(agent_set.writer.calls) == 2
        assert "研究主题：混合检索策略对比" in agent_set.writer.calls[0]
        assert "审校意见" in agent_set.writer.calls[1]

    def test_no_revision_when_critic_approves(self, tmp_path):
        w = MockText(["# 草稿"])
        c = MockText(["无需改进，报告质量良好。"])
        s = type("S", (), {
            "retriever": MockSub('{"conclusions":[],"confidence":0.1}'),
            "researcher": MockSub('{"conclusions":[]}'),
            "writer": w, "critic": c, "orchestrator": None,
        })()
        result = ResearchPipeline(s).run("任意主题")
        assert result.revised is False
        assert result.report_md == "# 草稿"
        assert len(w.calls) == 1

    def test_subagent_failure_does_not_crash(self, tmp_path):
        class Boom:
            def run_as_subagent(self, *a, **k):
                raise RuntimeError("429 rate limited")

        s = type("S", (), {
            "retriever": Boom(), "researcher": Boom(),
            "writer": MockText(["# 兜底报告"]),
            "critic": MockText(["无需改进"]), "orchestrator": None,
        })()
        result = ResearchPipeline(s).run("主题")
        assert "兜底报告" in result.report_md  # 阶段失败降级为占位 Findings，流水线继续

    def test_trace_flushed(self, agent_set, tmp_path):
        ResearchPipeline(agent_set, traces_dir=tmp_path / "traces").run("主题T")
        files = list((tmp_path / "traces").glob("pipeline-*.jsonl"))
        assert files
        lines = files[0].read_text(encoding="utf-8").splitlines()
        stages = [json.loads(ln)["stage"] for ln in lines[1:]]
        assert stages[0] == "gather"


class TestExplore:
    def test_explore_uses_orchestrator(self):
        orch = MockText(["# 探索式报告"])
        result = ExplorationMode(orch).run("主题")
        assert result.mode == "explore" and "探索式" in result.report_md
        assert orch.calls and "结构化中文 Markdown" in orch.calls[0]


# --------------------------------------------------------------------- #
class HangSub(MockSub):
    """可指定挂死（超过看门狗窗口）的子代理桩。"""

    def __init__(self, reply, hang=False):
        super().__init__(reply)
        self.hang = hang

    def run_as_subagent(self, task, tool_filter=None, return_summary=False, max_steps_override=None):
        if self.hang:
            time.sleep(5)  # 远超测试看门狗窗口；守护线程被放弃
        return super().run_as_subagent(task, tool_filter, return_summary, max_steps_override)


class TestStageWatchdog:
    def _agent_set(self, retriever, researcher):
        w = MockText(["# 报告"])
        c = MockText(["无需改进"])
        return type("S", (), {"retriever": retriever, "researcher": researcher,
                              "writer": w, "critic": c, "orchestrator": None})()

    def test_timeout_placeholder_and_registry_repair(self):
        # gather/research 均挂死 → 占位 Findings + 流水线继续 + 修复只调一次
        repairs = []
        s = self._agent_set(
            HangSub('{"conclusions":["g"],"confidence":0.9}', hang=True),
            HangSub('{"conclusions":["r"],"confidence":0.9}', hang=True))
        pipe = ResearchPipeline(s, stage_timeout=0.2, registry_repair=lambda: repairs.append(1))
        result = pipe.run("主题")

        assert "超时" in result.stages["gather"].conclusions[0]
        assert "超时" in result.stages["research"].conclusions[0]
        assert result.report_md == "# 报告"          # 写作阶段不受挂死阶段影响
        assert len(repairs) == 1                      # 一次 run 只修复一次

    def test_no_timeout_no_repair(self):
        repairs = []
        s = self._agent_set(
            HangSub('{"conclusions":["g"],"confidence":0.9}'),
            HangSub('{"conclusions":["r"],"confidence":0.9}'))
        pipe = ResearchPipeline(s, stage_timeout=5, registry_repair=lambda: repairs.append(1))
        pipe.run("主题")
        assert repairs == []

    def test_write_timeout_yields_failure_text(self):
        class HangWriter:
            def run_as_subagent(self, task, tool_filter=None, return_summary=False,
                                max_steps_override=None):
                time.sleep(5)  # 远超看门狗窗口；守护线程被放弃
                return {"success": True, "result": "", "metadata": {}}

        repairs = []
        s = self._agent_set(
            HangSub('{"conclusions":["g"],"confidence":0.9}'),
            HangSub('{"conclusions":[],"confidence":0.1}'))
        s.writer, s.critic = HangWriter(), MockText(["无需改进"])
        pipe = ResearchPipeline(s, stage_timeout=0.2, registry_repair=lambda: repairs.append(1))
        result = pipe.run("主题")

        assert "超时" in result.report_md
        assert len(repairs) == 1

    def test_stage_timeout_zero_disables_watchdog(self):
        s = self._agent_set(
            HangSub('{"conclusions":["g"],"confidence":0.9}'),
            HangSub('{"conclusions":[],"confidence":0.1}'))
        pipe = ResearchPipeline(s, stage_timeout=0)   # 0 = 不限时
        result = pipe.run("主题")
        assert result.report_md == "# 报告"

    def test_writer_narration_gets_corrective_reask(self):
        # writer 首条回复是过程叙述（无 Markdown 标题）→ 带纠正指令重问一次
        w = MockText(["全部片段核验完毕，现在将记录写入笔记。",
                      "# 正式报告\n带引用的正文 [abc/abc:0]。"])
        c = MockText(["无需改进"])
        s = type("S", (), {
            "retriever": MockSub('{"conclusions":[],"confidence":0.1}'),
            "researcher": MockSub('{"conclusions":[]}'),
            "writer": w, "critic": c, "orchestrator": None,
        })()
        result = ResearchPipeline(s).run("主题")

        assert result.report_md.startswith("# 正式报告")
        assert len(w.calls) == 2
        assert "【纠正】" in w.calls[1]          # 第二次任务带纠正指令

    def test_writer_narration_second_try_still_accepted(self):
        # 纠正后仍是叙述 → 不无限循环，原样采纳（宁要内容不要死循环）
        w = MockText(["还在核验引用……", "仍在整理素材……"])
        c = MockText(["无需改进"])
        s = type("S", (), {
            "retriever": MockSub('{"conclusions":[],"confidence":0.1}'),
            "researcher": MockSub('{"conclusions":[]}'),
            "writer": w, "critic": c, "orchestrator": None,
        })()
        result = ResearchPipeline(s).run("主题")
        assert result.report_md == "仍在整理素材……"
        assert len(w.calls) == 2
