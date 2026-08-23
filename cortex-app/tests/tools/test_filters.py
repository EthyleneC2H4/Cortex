"""P2 单元测试：角色过滤器矩阵。"""

from __future__ import annotations

import pytest

from cortex.tools.filters import ROLE_FILTERS, filter_for_role

pytestmark = pytest.mark.unit

# 各角色的最小可见集（展开后的子工具名）
EXPECT = {
    "retriever": {"knowledge_search", "memory_search"},
    "researcher": {"mcp_list_tools", "mcp_call_tool", "note_create", "note_update", "Skill"},
    "writer": {"memory_add", "memory_search", "memory_summary", "note_read", "note_update", "citation_verify", "Read"},
    "critic": {"citation_verify", "knowledge_search"},
}

# 绝对不可见的危险/越权工具
FORBIDDEN_ALL = {"Write", "Edit", "MultiEdit"}
FORBIDDEN_PER_ROLE = {
    "retriever": {"memory_add", "memory_forget", "note_delete"},
    "writer": {"mcp_call_tool", "knowledge_add_document"},
    "critic": {"memory_add", "mcp_call_tool", "note_delete"},
}


def allowed_names(flt) -> set:
    """ReadOnlyFilter 用 .allowed_tools，CustomFilter 白名单用 .allowed。"""
    return set(getattr(flt, "allowed_tools", None) or getattr(flt, "allowed", ()))


@pytest.mark.parametrize("role", sorted(ROLE_FILTERS))
def test_role_visibility(role):
    flt = filter_for_role(role)
    assert flt is not None
    allowed = allowed_names(flt)
    assert EXPECT[role] <= allowed, f"{role} 缺少工具: {EXPECT[role] - allowed}"
    assert not (allowed & FORBIDDEN_ALL), f"{role} 可见写文件工具"
    for bad in FORBIDDEN_PER_ROLE.get(role, set()):
        assert bad not in allowed, f"{role} 不应可见 {bad}"


def test_unknown_role_returns_none():
    assert filter_for_role("ghost") is None


def test_retriever_is_readonly_base():
    flt = filter_for_role("retriever")
    assert "Read" in allowed_names(flt), "ReadOnlyFilter 底座应含 Read 等只读工具"
