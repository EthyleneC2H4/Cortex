"""P6 单元测试：评测匹配器（BFCL/GAIA/Judge，纯函数零网络）与冻结数据集校验。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from cortex.eval import (
    JUDGE_DIMS,
    ast_match,
    cohen_kappa,
    judge_answer,
    level_metrics,
    multi_call_match,
    normalize,
    pairwise_winrate,
    parse_trace_tool_calls,
    quasi_exact_match,
    tool_call_accuracy,
)

pytestmark = pytest.mark.unit

DATA_DIR = Path(__file__).resolve().parents[1] / "cortex" / "eval" / "datasets"


def _load_jsonl(name: str) -> list:
    text = (DATA_DIR / name).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


class FakeLLM:
    """mock LLM 惯例：invoke 返回带 content 的对象；可脚本化多轮回复。"""

    def __init__(self, replies):
        self._replies = list(replies)
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages[-1]["content"])
        content = self._replies.pop(0) if self._replies else ""
        return type("R", (), {"content": content})()


# ---------------------------------------------------------------- normalize


class TestNormalize:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("  Hello World  ", "hello world"),          # strip + lower
            ("The RAG Pipeline", "rag pipeline"),        # 去冠词 the
            ("an Apple and the egg", "apple and egg"),   # 去冠词 a/an/the，不误伤 and
            ("$1,234", "1234"),                          # 去货币符号 + 千位逗号合并
            ("约 ¥3,000", "约 3000"),                     # 中文语境货币符号
            ("正确率 95%", "正确率 95"),                   # 去百分号
            ("1,234,567", "1234567"),                    # 多段千位逗号
            ("foo \t bar   baz ", "foo bar baz"),        # 空白折叠（含制表符）
            ("512。", "512"),                            # 去中文尾部标点
            ("Done!", "done"),                           # 去英文尾部标点
            ("end;", "end"),
        ],
    )
    def test_rules(self, raw, expected):
        assert normalize(raw) == expected

    def test_combo(self):
        # 全规则组合：strip/lower→去冠词→去 $ →千位逗号→折叠空白→去尾部 !
        assert normalize("  The Score: $1,234,567 !  ") == "score: 1234567"


# -------------------------------------------------------- quasi_exact_match


class TestQuasiExactMatch:
    @pytest.mark.parametrize(
        "pred,gold",
        [
            ("512", "512"),                       # 精确
            ("The Answer Is 42", "answer is 42"),  # 大小写 + 冠词
            ("1,234", "$1234"),                   # 千位逗号 + 货币符号
            ("香草, 巧克力", "巧克力,香草"),           # 逗号列表乱序相等
            ("香草，巧克力", "巧克力，香草"),           # 全角逗号列表乱序相等
            ("a, b, c", "c, b, a"),
            ("512。", "512"),                     # 尾部标点
        ],
    )
    def test_equal(self, pred, gold):
        assert quasi_exact_match(pred, gold) is True

    @pytest.mark.parametrize(
        "pred,gold",
        [
            ("512", "1024"),                      # 不等
            ("apple, banana", "apple, cherry"),   # 列表内容不同
            ("448", "448 tokens"),                # 多出 token
        ],
    )
    def test_not_equal(self, pred, gold):
        assert quasi_exact_match(pred, gold) is False


# ----------------------------------------------------------------- bfcl_like


class TestAstMatch:
    @pytest.mark.parametrize(
        "ename,eargs,aname,aargs,expected",
        [
            ("f", {"a": 1}, "f", {"a": 1}, True),                              # 同名同参
            ("f", {"a": 1, "b": 2}, "f", {"b": 2, "a": 1}, True),              # 参数顺序无关
            ("f", {"a": 1}, "f", {"a": 1, "b": 2}, False),                     # 实参多余 ✗
            ("f", {"a": 1, "b": 2}, "f", {"a": 1}, False),                     # 金标多余同样 ✗
            ("f", {"k": 1}, "f", {"k": 1.0}, True),                            # 1 == 1.0
            ("f", {"flag": True}, "f", {"flag": 1}, False),                    # bool vs int ✗
            ("f", {"flag": 0}, "f", {"flag": False}, False),                   # int vs bool ✗
            ("f", {"w": {"dense": 0.6}, "tags": ["a", "b"]},
             "f", {"tags": ["a", "b"], "w": {"dense": 0.6}}, True),             # 嵌套 dict/list
            ("f", {"tags": ["a", "b"]}, "f", {"tags": ["b", "a"]}, False),      # 列表有序
            ("search", {"q": "x"}, "retrieve", {"q": "x"}, False),              # 名字不同 ✗
        ],
    )
    def test_matrix(self, ename, eargs, aname, aargs, expected):
        assert ast_match(ename, eargs, aname, aargs) is expected

    def test_unparsable_args_sentinel_never_matches(self):
        """args=None（解析失败哨兵）与任何金标——含空参金标——都不得匹配。"""
        assert ast_match("f", {}, "f", None) is False
        assert ast_match("f", None, "f", {}) is False
        assert multi_call_match([("f", {})], [("f", None)]) is False


class TestMultiCallMatch:
    def test_two_calls_out_of_order_all_hit(self):
        exp = [("knowledge_search", {"query": "RRF", "top_k": 5}),
               ("note_create", {"title": "t", "content": "c"})]
        act = [("note_create", {"content": "c", "title": "t"}),
               ("knowledge_search", {"top_k": 5.0, "query": "RRF"})]
        assert multi_call_match(exp, act) is True

    def test_missing_one_call(self):
        exp = [("f", {}), ("g", {})]
        assert multi_call_match(exp, [("f", {})]) is False

    def test_extra_actual_call(self):
        assert multi_call_match([("f", {})], [("f", {}), ("g", {})]) is False

    def test_same_name_distinct_args_swapped(self):
        exp = [("f", {"v": 1}), ("f", {"v": 2})]
        act = [("f", {"v": 2.0}), ("f", {"v": 1})]
        assert multi_call_match(exp, act) is True

    def test_empty_both(self):
        assert multi_call_match([], []) is True


class TestToolCallAccuracy:
    def test_mixed_shapes(self):
        cases = [
            {"expected": ("f", {"a": 1}), "actual": ("f", {"a": 1.0})},            # ✓
            {"expected": {"name": "g", "args": {"b": True}},
             "actual": {"name": "g", "args": {"b": 1}}},                           # ✗ bool 严格
            {"expected": [("f", {}), ("g", {})],
             "actual": [("g", {}), ("f", {})]},                                    # ✓ 集合乱序
            {"expected": {"name": ["h", "i"], "args": [{"x": 1}, {"y": 2}]},
             "actual": [("h", {"x": 1}), ("i", {"y": 2})]},                        # ✓ name 列表形态
            {"expected": ("h", {"x": 1}), "actual": ("h", {"x": 2})},              # ✗
        ]
        assert tool_call_accuracy(cases) == pytest.approx(3 / 5)

    def test_empty_cases(self):
        assert tool_call_accuracy([]) == 0.0


class TestParseTraceToolCalls:
    def _write(self, tmp_path, lines):
        p = tmp_path / "trace.jsonl"
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return p

    def test_mixed_lines(self, tmp_path):
        p = self._write(tmp_path, [
            '{"event": "llm_end", "payload": {"text": "hi"}}',                  # 非 tool_call 事件
            ('{"event": "tool_call", "payload": '
             '{"tool_name": "knowledge_search", "parameters": {"query": "RRF", "top_k": 5}}}'),
            '这不是 JSON 的坏行 {{{',
            ('{"event": "tool_call", "name": "note_search",'
             ' "arguments": "{\\"query\\": \\"记忆\\", \\"limit\\": 3}"}'),       # 平铺键 + 字符串 arguments
            '{"event": "tool_call", "payload": {"tool_name": "broken", "arguments": "{oops"}}',  # 坏 arguments
            '{"event": "tool_start", "payload": {"tool_name": "ignored"}}',      # 非 tool_call 事件
            "",                                                                  # 空行
        ])
        calls = parse_trace_tool_calls(p)
        assert calls == [
            ("knowledge_search", {"query": "RRF", "top_k": 5}),
            ("note_search", {"query": "记忆", "limit": 3}),
        ]

    def test_bad_json_line_skipped_and_rest_parsed(self, tmp_path):
        p = self._write(tmp_path, ['{"event": "tool_call"',                  # 截断 JSON
                                   '{"event": "tool_call", "payload": {"tool_name": "mcp_list_tools"}}'])
        assert parse_trace_tool_calls(p) == [("mcp_list_tools", {})]


# -------------------------------------------------------------------- judge


class TestJudgeAnswer:
    def test_fenced_json_scores(self):
        llm = FakeLLM([('好的，评分如下：\n```json\n{"准确性": 5, "相关性": 4, '
                        '"完整性": 4, "引用可靠性": 3, "rationale": "基本忠实"}\n```')])
        result = judge_answer("Q", "A", "REF", llm)
        assert result["准确性"] == 5 and result["相关性"] == 4
        assert result["完整性"] == 4 and result["引用可靠性"] == 3
        assert result["rationale"] == "基本忠实"
        assert set(JUDGE_DIMS).issubset(result.keys())

    def test_plain_json_with_surrounding_text(self):
        llm = FakeLLM(['结论：{"准确性": 2, "相关性": 2, "完整性": 5, "引用可靠性": 5, "rationale": "有编造"}'])
        assert judge_answer("Q", "A", "REF", llm)["准确性"] == 2

    def test_bad_output_all_three(self):
        llm = FakeLLM(["我觉得这个回答相当不错，值得肯定。"])
        result = judge_answer("Q", "A", "REF", llm)
        assert all(result[d] == 3 for d in JUDGE_DIMS)
        assert result["rationale"] == "judge_failed"

    def test_llm_exception_all_three(self):
        class BoomLLM:
            def invoke(self, messages):
                raise RuntimeError("down")

        result = judge_answer("Q", "A", "REF", BoomLLM())
        assert [result[d] for d in JUDGE_DIMS] == [3, 3, 3, 3]
        assert result["rationale"] == "judge_failed"

    def test_clamp_and_missing_dim(self):
        llm = FakeLLM(['{"准确性": 9, "相关性": 0, "完整性": "4", "rationale": null}'])
        result = judge_answer("Q", "A", "REF", llm)
        assert result["准确性"] == 5          # 越上界 clamp
        assert result["相关性"] == 1          # 越下界 clamp
        assert result["完整性"] == 4          # 字符串数值收敛
        assert result["引用可靠性"] == 3       # 缺失补 3 分


class TestPairwiseWinrate:
    Q, A, B = "什么是混合检索？", "答案甲", "答案乙"

    def test_disagreement_is_tie(self):
        llm = FakeLLM(["A", "B"])  # 第一次回 A、第二次回 B
        assert pairwise_winrate(self.Q, self.A, self.B, llm) == "tie"
        assert len(llm.prompts) == 2
        # 第二次调用展示顺序交换：B 的块在前（用 rindex 跳过模板头部的标签说明）
        assert llm.prompts[0].rindex("【回答 A】") < llm.prompts[0].rindex("【回答 B】")
        assert llm.prompts[1].rindex("【回答 B】") < llm.prompts[1].rindex("【回答 A】")

    def test_consistent_b_wins(self):
        llm = FakeLLM(["B", "B"])
        assert pairwise_winrate(self.Q, self.A, self.B, llm) == "b"

    def test_consistent_a_wins_with_noise_words(self):
        llm = FakeLLM(["我认为 Answer A 更好", "答：A"])
        assert pairwise_winrate(self.Q, self.A, self.B, llm) == "a"

    def test_invalid_verdicts_tie(self):
        llm = FakeLLM(["无法比较", "C"])
        assert pairwise_winrate(self.Q, self.A, self.B, llm) == "tie"

    def test_no_swap_single_call(self):
        llm = FakeLLM(["tie"])
        assert pairwise_winrate(self.Q, self.A, self.B, llm, swap=False) == "tie"
        assert len(llm.prompts) == 1

    def test_llm_exception_tie(self):
        class BoomLLM:
            def invoke(self, messages):
                raise RuntimeError("boom")

        assert pairwise_winrate(self.Q, self.A, self.B, BoomLLM()) == "tie"


class TestCohenKappa:
    def test_perfect_agreement(self):
        votes = ["a", "b", "tie"]
        assert cohen_kappa(votes, list(votes)) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "va,vb,expected",
        [
            (["x", "x", "y", "y"], ["x", "y", "x", "y"], 0.0),    # po=pe=0.5 → κ=0
            (["x", "x", "y", "y"], ["x", "x", "y", "x"], 0.5),    # 手算 κ=0.5
        ],
    )
    def test_known_values(self, va, vb, expected):
        assert cohen_kappa(va, vb) == pytest.approx(expected)

    def test_degenerate_single_label(self):
        assert cohen_kappa(["z", "z", "z"], ["z", "z", "z"]) == pytest.approx(1.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            cohen_kappa(["a"], ["a", "b"])

    def test_empty(self):
        assert cohen_kappa([], []) == 0.0


class TestLevelMetrics:
    def test_grouping_and_ordering(self):
        rows = [
            {"level": "L2", "correct": True},
            {"level": "L1", "correct": True},
            {"level": "L1", "correct": False},
            {"level": "L3", "correct": False},
        ]
        metrics = level_metrics(rows)
        assert list(metrics.keys()) == ["L1", "L2", "L3"]
        assert metrics["L1"] == {"n": 2, "correct": 1}
        assert metrics["L2"] == {"n": 1, "correct": 1}
        assert metrics["L3"] == {"n": 1, "correct": 0}

    def test_empty_rows(self):
        assert level_metrics([]) == {}


# ------------------------------------------------------- 冻结数据集加载校验


class TestFrozenDatasets:
    def test_bfcl_dataset(self):
        rows = _load_jsonl("bfcl.jsonl")
        assert len(rows) == 10
        multi = 0
        for row in rows:
            assert "name" in row and "args" in row
            if isinstance(row["name"], list):  # 多调用场景：name 列表 + args 列表并行
                assert isinstance(row["args"], list) and len(row["args"]) == len(row["name"])
                for args in row["args"]:
                    assert isinstance(args, dict)
                multi += 1
            else:
                assert isinstance(row["name"], str) and row["name"]
                assert isinstance(row["args"], dict)
        assert multi >= 1  # 至少覆盖一个多调用场景
        flat_args = [a for r in rows if isinstance(r["args"], dict) for a in r["args"].values()]
        assert any(isinstance(v, bool) for v in flat_args)          # 布尔型参数
        assert any(isinstance(v, (int, float)) and not isinstance(v, bool)
                   for v in flat_args)                               # 数值型参数
        assert any(isinstance(v, (list, dict)) for v in flat_args)   # 嵌套 list/dict 参数

    def test_gaia_dataset(self):
        rows = _load_jsonl("gaia.jsonl")
        assert len(rows) == 14
        levels = Counter(r["level"] for r in rows)
        assert levels == Counter({"L1": 5, "L2": 5, "L3": 4})
        ids = [r["id"] for r in rows]
        assert len(set(ids)) == len(ids)
        for row in rows:
            assert {"id", "level", "question", "answer"} <= set(row)
            assert row["question"].strip()
            assert normalize(row["answer"]), f"答案规范化后为空: {row['id']}"

    def test_gaia_gold_self_consistency(self):
        """金标与其自身变体必须 quasi-exact 相等（防 normalize 规则误伤冻结答案）。"""
        for row in _load_jsonl("gaia.jsonl"):
            gold = row["answer"]
            padded = f"  {gold.capitalize()} 。" if gold.isascii() else f" The {gold}。 "
            assert quasi_exact_match(padded, gold), f"自一致失败: {row['id']}"

    def test_judge_pairs_dataset(self):
        rows = _load_jsonl("judge_pairs.jsonl")
        assert len(rows) == 6
        labels = set()
        ids = []
        for row in rows:
            assert {"id", "question", "reference", "answer_a", "answer_b", "better"} <= set(row)
            assert row["better"] in {"A", "B", "tie"}
            assert row["question"].strip() and row["reference"].strip()
            assert row["answer_a"].strip() and row["answer_b"].strip()
            labels.add(row["better"])
            ids.append(row["id"])
        assert len(set(ids)) == len(ids)
        assert labels == {"A", "B", "tie"}  # 三种金标齐全，供 runner 校准与 κ 计算

    def test_readme_freeze_date(self):
        text = (DATA_DIR / "README.md").read_text(encoding="utf-8")
        assert "2026-08-22" in text
