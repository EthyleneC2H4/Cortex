"""P2 单元测试：NoteTool 展开名、乐观锁、ToolResponse 形状。"""

from __future__ import annotations

import pytest
from hello_agents.tools import ToolRegistry

from cortex.tools.note_tool import NoteTool

pytestmark = pytest.mark.unit


@pytest.fixture
def note(tmp_path) -> NoteTool:
    return NoteTool(notes_dir=tmp_path / "notes")


def _status(resp):
    """ToolResponse.status 是 ToolStatus 枚举（值 'success'/'error'）。"""
    return resp.status.value if hasattr(resp.status, "value") else resp.status


def ok(resp):
    assert _status(resp) == "success", f"期望成功，实际 {resp.status}: {resp.error_info}"
    return resp


def _code(resp):
    """error_info 是 dict（{"code": "CONFLICT", ...}），兼容对象形态。"""
    info = resp.error_info
    if info is None:
        return None
    return info.get("code") if isinstance(info, dict) else getattr(info, "code", None)


def err_code(resp, code: str):
    e = err(resp)
    assert _code(e) == code, f"期望 {code}，实际 {_code(e)}"
    return e


def err(resp):
    assert _status(resp) != "success", f"期望失败，实际 success: {resp.text}"
    assert _code(resp) is not None, "错误响应必须带 error_info.code"
    return resp


class TestExpansion:
    def test_expanded_names_exact_set(self):
        tool = NoteTool(notes_dir="/tmp/cortex-test-notes-unused")
        names = {t.name for t in tool.get_expanded_tools()}
        assert names == {
            "note_create",
            "note_read",
            "note_update",
            "note_search",
            "note_list",
            "note_delete",
        }

    def test_register_expands_into_registry(self, tmp_path):
        reg = ToolRegistry()
        tool = NoteTool(notes_dir=tmp_path / "notes")
        with __import__("cortex.logging_setup", fromlist=["quiet"]).quiet():
            reg.register_tool(tool)  # expandable 工具自动展开（无 expandable kwarg）
        names = set(reg.list_tools())
        assert {"note_create", "note_read", "note_update", "note_search", "note_list", "note_delete"} <= names
        assert "note" not in names, "expandable 注册后父工具不应存在"


class TestCrud:
    def test_create_read_roundtrip(self, note):
        ok(note.create(title="RESEARCH_PLAN", content="# 计划\n1. 检索"))
        r = ok(note.read(title="RESEARCH_PLAN"))
        assert "# 计划" in r.data["content"]
        assert r.data["version"] == 1
        assert "v1" in r.text

    def test_duplicate_create_conflict(self, note):
        ok(note.create(title="dup", content="a"))
        err_code(note.create(title="dup", content="b"), "CONFLICT")

    def test_update_optimistic_lock(self, note):
        ok(note.create(title="FINDINGS", content="v1 内容"))
        err_code(note.update(title="FINDINGS", content="过期写入", expected_version=99), "CONFLICT")
        # 读到的版本仍是最初内容 —— 过期写入未生效
        assert ok(note.read(title="FINDINGS")).data["content"] == "v1 内容"
        r = ok(note.update(title="FINDINGS", content="v2 内容", expected_version=1))
        assert r.data["version"] == 2
        assert ok(note.read(title="FINDINGS")).data["content"] == "v2 内容"

    def test_read_missing_not_found(self, note):
        e = err(note.read(title="不存在"))
        assert _code(e) == "NOT_FOUND"

    def test_empty_title_invalid_param(self, note):
        e = err(note.create(title="  ", content="x"))
        assert _code(e) == "INVALID_PARAM"

    def test_delete_then_read_fails(self, note):
        ok(note.create(title="tmp", content="x"))
        ok(note.delete(title="tmp"))
        err(note.read(title="tmp"))

    def test_search_and_list(self, note):
        ok(note.create(title="检索策略笔记", content="混合检索 RRF 融合"))
        ok(note.create(title="记忆设计", content="0.6/0.25/0.15 权重"))
        hits = ok(note.search(query="RRF")).data["hits"]
        assert [h["title"] for h in hits] == ["检索策略笔记"]
        titles = {n["title"] for n in ok(note.list()).data["notes"]}
        assert titles == {"检索策略笔记", "记忆设计"}

    def test_never_raises_run_paths(self, note):
        # 所有公开动作在异常输入下都返回 ToolResponse 而非抛异常
        for call in (
            lambda: note.read(title="ghost"),
            lambda: note.update(title="ghost", content="x", expected_version=1),
            lambda: note.delete(title="ghost"),
            lambda: note.search(query=""),
        ):
            resp = call()
            assert hasattr(resp, "status")

    def test_cjk_slug_collision_distinct_titles(self, note):
        ok(note.create(title="结论：A/B 测试", content="1"))
        ok(note.create(title="结论：A_B 测试", content="2"))  # slug 同为 结论-a-b-测试
        # 标题精确查找仍各自可达（meta.title 为唯一键）
        assert ok(note.read(title="结论：A/B 测试")).data["content"] == "1"
        assert ok(note.read(title="结论：A_B 测试")).data["content"] == "2"

    def test_bind_registry_cache_called(self, note):
        class FakeReg:
            def __init__(self):
                self.calls = []

            def cache_read_metadata(self, path, meta):
                self.calls.append((path, meta))

        fake = FakeReg()
        note.bind_registry(fake)
        ok(note.create(title="cached", content="x"))
        ok(note.read(title="cached"))
        assert len(fake.calls) == 1
        assert fake.calls[0][1] == {"version": 1}
