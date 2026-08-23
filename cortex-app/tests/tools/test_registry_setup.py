"""P2 单元测试：注册表装配（builtin + cortex 工具、quiet 无打印）。"""

from __future__ import annotations

import contextlib
import io

import pytest
from hello_agents.tools import ToolRegistry

from cortex.tools.registry_setup import register_builtin_tools, register_cortex_tools

pytestmark = pytest.mark.unit


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry()


def test_builtin_tools_registered(registry, tmp_path):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        register_builtin_tools(registry, project_root=tmp_path)
    names = set(registry.list_tools())
    assert "python_calculator" in names
    assert "Read" in names
    assert "Write" not in names and "Edit" not in names  # 默认不给写权限
    assert "✅" not in buf.getvalue(), "注册过程不应向 stdout 打印"


def test_builtin_write_opt_in(registry, tmp_path):
    with contextlib.redirect_stdout(io.StringIO()):
        register_builtin_tools(registry, project_root=tmp_path, allow_write=True)
    names = set(registry.list_tools())
    assert {"Write", "Edit"} <= names


def test_cortex_note_tools_registered(cfg_factory, registry):
    cfg = cfg_factory()
    with contextlib.redirect_stdout(io.StringIO()):
        tools = register_cortex_tools(registry, cfg)
    names = set(registry.list_tools())
    assert {"note_create", "note_read", "note_update", "note_search", "note_list", "note_delete"} <= names
    assert "note" not in names
    assert set(tools) == {"note"}


def test_full_setup_quiet(cfg_factory, registry, tmp_path):
    """builtin + cortex 一起装，stdout 保持干净（框架噪声被 quiet 吞掉）。"""
    cfg = cfg_factory()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        register_builtin_tools(registry, project_root=tmp_path)
        register_cortex_tools(registry, cfg)
    assert buf.getvalue() == "", f"stdout 泄漏: {buf.getvalue()!r}"
