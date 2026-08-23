"""P5 单元测试：GSSC 上下文引擎（预算/配额/压缩/分区序）。"""

from __future__ import annotations

import pytest

from cortex.context.engine import TYPE_QUOTAS, ContextEngine, Packet

pytestmark = pytest.mark.unit


class FakeCounter:
    """把 token 计数近似为字符数//4，稳定可断言。"""

    def count_text(self, s):
        return max(1, len(s) // 4)


@pytest.fixture
def engine(cfg_factory):
    cfg = cfg_factory()
    eng = ContextEngine(cfg)
    eng.counter = FakeCounter()
    eng._count = eng.counter.count_text
    return eng


class TestBudget:
    def test_budget_formula(self, engine):
        # 128000 × 0.7 × (1-0.15)
        assert engine.budget == int(128000 * 0.7 * 0.85)


class TestSelect:
    def test_type_quota_enforced(self, engine):
        big = Packet(type="knowledge_base", text="x" * 40000, weight=1.0)
        small = Packet(type="task_state", text="task", weight=1.0)
        out = engine.select([big, small])
        kb = next(p for p in out if p.type == "knowledge_base")
        assert engine._count(kb.text) <= TYPE_QUOTAS["knowledge_base"] + 50
        assert any(p.type == "task_state" for p in out)

    def test_skip_and_continue_keeps_others(self, engine):
        huge = Packet(type="related_memory", text="m" * 9000, weight=1.0)
        tiny = Packet(type="related_memory", text="tiny", weight=0.9)
        out = engine.select([huge, tiny])
        types = [p.type for p in out]
        assert "related_memory" in types  # 大包被跳过后小包仍入选


class TestCompress:
    def test_short_history_untouched(self, engine):
        out, compressed = engine.compress_history("短历史", budget_tokens=1500)
        assert out == "短历史" and compressed is False

    def test_long_history_compressed_with_keeps(self, engine):
        class LLM:
            def invoke(self, messages):
                assert "对话历史" in messages[0]["content"]
                return type("R", (), {"content": "[PLAN] 已完成检索\n[结论] RRF 最优"})()

        engine.weak_llm = LLM()
        hist = "\n".join(f"line {i}: " + "y" * 200 for i in range(40))
        hist += "\n[结论] 必须保留的行"
        out, compressed = engine.compress_history(hist, budget_tokens=100)
        assert compressed is True
        assert "[PLAN]" in out
        assert "必须保留的行" in out

    def test_llm_failure_hard_truncate(self, engine):
        class Bad:
            def invoke(self, m):
                raise RuntimeError("429")

        engine.weak_llm = Bad()
        hist = "z" * 5000
        out, compressed = engine.compress_history(hist, budget_tokens=200)
        assert compressed is True and len(out) < len(hist)


class TestStructure:
    def test_section_order(self, engine):
        selected = [
            Packet(type="related_memory", text="MEM"),
            Packet(type="knowledge_base", text="EVID"),
        ]
        prompt = engine.structure("任务T", selected, compressed_history="HIST")
        i_task = prompt.index("# Task")
        i_mem = prompt.index("# Relevant Memory")
        i_evid = prompt.index("# Evidence")
        i_hist = prompt.index("# Conversation History")
        assert i_task < i_mem < i_evid < i_hist
        assert "只依据提供的证据" in prompt.split("\n")[1]


class TestGatherAndBuild:
    def test_gather_pulls_memory_and_knowledge(self, cfg_factory):
        class Mem:
            def recall_for_context(self, limit=10, min_importance=0.3):
                return [{"type": "semantic", "content": "用户偏好中文"}]

        class KB:
            def search_for_context(self, query, limit=5):
                return [{"text": "证据", "source": "s", "chunk_id": "d:0", "score": 0.8}]

        eng = ContextEngine(cfg_factory(), memory=Mem(), knowledge=KB())
        packets = eng.gather("主题")
        types = {p.type for p in packets}
        assert {"related_memory", "knowledge_base", "task_state"} <= types

    def test_build_system_prompt_end_to_end(self, engine):
        built = engine.build_system_prompt(
            "对比混合检索策略",
            history=[{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}],
        )
        assert built.system_prompt.startswith("# Role & Policies")
        assert "对比混合检索策略" in built.system_prompt
        assert built.budget_tokens > 0
