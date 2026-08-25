"""评测运行器离线测试：全 mock LLM；套件逻辑用行内数据，编排用 monkeypatch。"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import cortex.eval.runner as runner_mod
from cortex.eval.runner import Checkpoint, RunOptions, run_bfcl, run_gaia, run_judge

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------- 假 LLM -- #
class _Resp:
    def __init__(self, content: str = "") -> None:
        self.content = content


class _ToolCall:
    def __init__(self, name: str, args: dict) -> None:
        self.name = name
        self.arguments = json.dumps(args, ensure_ascii=False)


class _ToolResp:
    def __init__(self, calls) -> None:
        self.tool_calls = calls


class _FakeToolLLM:
    """恒定返回同一工具调用（invoke_with_tools 形态，同框架 LLM）。"""

    def __init__(self, name: str, args: dict) -> None:
        self.name, self.args = name, args
        self.calls = 0

    def invoke_with_tools(self, messages, tools, tool_choice="auto", **kwargs):
        self.calls += 1
        return _ToolResp([_ToolCall(self.name, self.args)])


class _FakeMultiCallLLM:
    """恒定返回一组工具调用（多调用行用）。"""

    def __init__(self, calls) -> None:
        self.calls_spec = calls  # [(name, args)]
        self.calls = 0

    def invoke_with_tools(self, messages, tools, tool_choice="auto", **kwargs):
        self.calls += 1
        return _ToolResp([_ToolCall(n, a) for n, a in self.calls_spec])


class _FakeOpenAIShapeLLM:
    """模拟框架适配器的**原生 OpenAI 响应形态**：
    choices[0].message.tool_calls[*].function.name/.arguments（live 实际走这条路径）。"""

    def __init__(self, calls) -> None:
        self.calls_spec = calls

    def invoke_with_tools(self, messages, tools, tool_choice="auto", **kwargs):
        fn_calls = [
            SimpleNamespace(function=SimpleNamespace(name=n, arguments=json.dumps(a, ensure_ascii=False)))
            for n, a in self.calls_spec
        ]
        message = SimpleNamespace(tool_calls=fn_calls)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeTextLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0

    def invoke(self, messages, **kwargs):
        self.calls += 1
        return _Resp(self.content)


# ---------------------------------------------------------------- bfcl ---- #
def test_run_bfcl_correct_call(tmp_path):
    rows = [{"name": "get_weather", "args": {"city": "北京"}, "instruction": "查天气"}]
    llm = _FakeToolLLM("get_weather", {"city": "北京"})
    res = run_bfcl(rows, llm, Checkpoint(tmp_path / "c.json"))
    assert res["accuracy"] == 1.0
    assert llm.calls == 1


def test_run_bfcl_wrong_args_scores_zero(tmp_path):
    rows = [{"name": "get_weather", "args": {"city": "北京"}, "instruction": "查天气"}]
    llm = _FakeToolLLM("get_weather", {"city": "上海"})  # 金标参数不符
    res = run_bfcl(rows, llm, Checkpoint(tmp_path / "c.json"))
    assert res["accuracy"] == 0.0


def test_run_bfcl_multi_call_row(tmp_path):
    # 数据集行 6/8 形态：name/args 为平行列表（一次请求应发多个工具调用）
    rows = [{
        "name": ["knowledge_add_text", "knowledge_search"],
        "args": [{"title": "GSSC 管线", "content": "四阶段。"},
                 {"query": "GSSC 上下文管线", "top_k": 3}],
    }]
    llm = _FakeMultiCallLLM([
        ("knowledge_add_text", {"title": "GSSC 管线", "content": "四阶段。"}),
        ("knowledge_search", {"query": "GSSC 上下文管线", "top_k": 3}),
    ])
    res = run_bfcl(rows, llm, Checkpoint(tmp_path / "c.json"))
    assert res["accuracy"] == 1.0
    assert len(res["detail"]["bfcl/0"]["actual"]) == 2


def test_run_bfcl_multi_call_wrong_count_scores_zero(tmp_path):
    rows = [{"name": ["a_tool", "b_tool"], "args": [{"x": 1}, {"y": 2}]}]
    llm = _FakeMultiCallLLM([("a_tool", {"x": 1})])  # 只调了一个 → 数量不等
    res = run_bfcl(rows, llm, Checkpoint(tmp_path / "c.json"))
    assert res["accuracy"] == 0.0


def test_run_bfcl_mixed_single_and_multi_rows(tmp_path):
    rows = [{"name": "solo", "args": {"q": 1}},
            {"name": ["m1", "m2"], "args": [{"p": 2}, {"r": 3}]}]
    llm = _FakeMultiCallLLM([("m1", {"p": 2}), ("m2", {"r": 3})])
    res = run_bfcl(rows, llm, Checkpoint(tmp_path / "c.json"))
    # 第一行（单调用金标）收到两个调用只取第一个 → 不匹配；第二行匹配
    assert res["detail"]["bfcl/0"]["correct"] is False
    assert res["detail"]["bfcl/1"]["correct"] is True


def test_run_bfcl_raw_openai_response_shape(tmp_path):
    # live 回归：框架适配器返回原生 OpenAI 形态，tool_calls 在 choices[0].message 下
    rows = [{"name": "get_weather", "args": {"city": "北京"}, "instruction": "查天气"},
            {"name": ["a_tool", "b_tool"], "args": [{"x": 1}, {"y": 2}]}]
    llm = _FakeOpenAIShapeLLM([("get_weather", {"city": "北京"})])
    res = run_bfcl(rows[:1], llm, Checkpoint(tmp_path / "c.json"))
    assert res["accuracy"] == 1.0  # 修复前此形态恒为 actual=null → accuracy=0.0

    llm2 = _FakeOpenAIShapeLLM([("a_tool", {"x": 1}), ("b_tool", {"y": 2})])
    res2 = run_bfcl(rows[1:], llm2, Checkpoint(tmp_path / "c2.json"))
    assert res2["accuracy"] == 1.0


def test_extract_tool_calls_marks_bad_arguments_as_unmatchable():
    """参数 JSON 解析失败 → None 哨兵（非静默 {}）；合法调用不受影响。"""
    from cortex.eval.runner import _extract_tool_calls

    resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[
        SimpleNamespace(function=SimpleNamespace(name="t1", arguments="{bad json")),
        SimpleNamespace(function=SimpleNamespace(name="t2", arguments='{"ok": 1}')),
    ]))])
    calls = _extract_tool_calls(resp)
    assert calls == [("t1", None), ("t2", {"ok": 1})]


class _BrokenArgsLLM:
    """arguments 为非法 JSON 字符串（模拟免费档生成截断）。"""

    def __init__(self, name: str) -> None:
        self.name = name

    def invoke_with_tools(self, messages, tools, tool_choice="auto", **kwargs):
        return _ToolResp([SimpleNamespace(name=self.name,
                                          arguments='{"query": "x"')])  # 缺右括号


def test_run_bfcl_unparsable_args_never_match_empty_gold(tmp_path):
    """回归：损坏的参数绝不能静默变 {} 与「无参金标」假匹配（修复前 accuracy=1.0）。"""
    rows = [{"name": "no_arg_tool", "args": {}, "instruction": "x"}]
    res = run_bfcl(rows, _BrokenArgsLLM("no_arg_tool"), Checkpoint(tmp_path / "c.json"))
    assert res["accuracy"] == 0.0
    assert res["detail"]["bfcl/0"]["actual"] == [("no_arg_tool", None)]


def test_extract_tool_calls_empty_and_none():
    from cortex.eval.runner import _extract_tool_calls

    assert _extract_tool_calls(SimpleNamespace(content="纯文本")) == []
    assert _extract_tool_calls(SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None))])) == []


def test_checkpoint_persists_and_resumes(tmp_path):
    path = tmp_path / "c.json"
    rows = [{"name": "t", "args": {}, "instruction": "x"}]

    ck = Checkpoint(path)
    ck.mark("bfcl/0", {"correct": True})
    # 重启后新实例读回断点，不再调用 LLM
    llm = _FakeToolLLM("t", {})
    res = run_bfcl(rows, llm, Checkpoint(path))
    assert llm.calls == 0
    assert res["accuracy"] == 1.0


# ---------------------------------------------------------------- gaia ---- #
def test_run_gaia_quasi_exact_match(tmp_path):
    rows = [{"id": "g1", "question": "1+1等于几？", "answer": "2", "level": "L1"}]
    llm = _FakeTextLLM("2")
    res = run_gaia(rows, llm, Checkpoint(tmp_path / "c.json"))
    assert res["quasi_exact_match"] == 1.0
    assert res["levels"]["L1"]["correct"] == 1
    assert llm.calls == 1


def test_run_gaia_wrong_answer_per_level(tmp_path):
    rows = [
        {"id": "g1", "question": "q1", "answer": "paris", "level": "L1"},
        {"id": "g2", "question": "q2", "answer": "3", "level": "L2"},
    ]
    llm = _FakeTextLLM("错误答案")
    res = run_gaia(rows, llm, Checkpoint(tmp_path / "c.json"))
    assert res["quasi_exact_match"] == 0.0
    assert res["levels"]["L1"]["n"] == 1 and res["levels"]["L2"]["n"] == 1


# ---------------------------------------------------------------- judge --- #
def test_run_judge_agreement_and_kappa(tmp_path):
    # 裁判恒答 "a"：与金标一半一致 → agreement=0.5，kappa 可计算（非 NaN）
    rows = [
        {"id": "j1", "question": "q", "answer_a": "长而好的回答", "answer_b": "短回答",
         "better": "a"},
        {"id": "j2", "question": "q", "answer_a": "x", "answer_b": "y", "better": "tie"},
    ]
    judge = _FakeTextLLM("我认为回答 A 更好")
    res = run_judge(rows, judge, Checkpoint(tmp_path / "c.json"))
    assert res["agreement_with_gold"] == 0.5
    assert res["win_rate_a"] == 1.0
    assert -1.0 <= res["kappa_vs_gold"] <= 1.0


# ---------------------------------------------------------------- 编排 ---- #
def test_run_eval_writes_report_and_checkpoint(tmp_path, monkeypatch):
    def _gaia(rows, llm, ckpt):  # 模拟真实套件：逐条 mark 断点
        ckpt.mark("gaia/0", {"pred": "x"})
        return {"n": 2, "quasi_exact_match": 0.5, "levels": {"L1": {"n": 1, "correct": 1}}}

    monkeypatch.setattr(runner_mod, "run_bfcl",
                        lambda rows, llm, ckpt: {"n": 1, "accuracy": 1.0})
    monkeypatch.setattr(runner_mod, "run_gaia", _gaia)
    monkeypatch.setattr(runner_mod, "run_judge",
                        lambda rows, llm, ckpt: {"n": 1, "agreement_with_gold": 0.8,
                                                 "kappa_vs_gold": 0.6})
    msgs: list[str] = []
    cfg = SimpleNamespace(data_dir=tmp_path)
    out = runner_mod.run_eval(cfg, strong_llm=object(), weak_llm=object(),
                              options=RunOptions(),
                              progress=msgs.append)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "BFCL" in text and "GAIA" in text and "Judge" in text
    assert "no_rerank=False" in text
    assert (out.parent / "checkpoint.json").exists()
    assert any("报告已写入" in m for m in msgs)


def test_render_report_lists_ablation_flags():
    text = runner_mod._render_report(
        RunOptions(no_rerank=True, no_memory=True), [("X", {"accuracy": 0.9})])
    assert "no_rerank=True" in text and "no_memory=True" in text
