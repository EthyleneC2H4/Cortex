"""P1 单元测试：配置加载与框架 Config 桥接。"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_paths_absolute_and_dirs_created(cfg_factory, tmp_path):
    cfg = cfg_factory(data_dir="data", qdrant_local_path="data/qdrant_local")
    assert cfg.project_root == tmp_path
    assert cfg.data_dir == tmp_path / "data" and cfg.data_dir.is_dir()
    assert cfg.qdrant_local_path == tmp_path / "data" / "qdrant_local"
    for sub in ("notes", "bm25", "traces", "sessions", "todos", "devlogs", "tool-output", "eval"):
        assert (cfg.data_dir / sub).is_dir(), sub


def test_framework_config_dirs_absolute_under_data(cfg_factory):
    from cortex.config import framework_config

    cfg = cfg_factory()
    ha = framework_config(cfg)
    assert str(cfg.data_dir) in ha.trace_dir and Path(ha.trace_dir).is_absolute()
    assert str(cfg.data_dir) in ha.session_dir
    assert str(cfg.data_dir) in ha.todowrite_persistence_dir
    assert str(cfg.data_dir) in ha.devlog_persistence_dir
    assert str(cfg.data_dir) in ha.tool_output_dir
    # 主 agent 保留子代理/TodoWrite 能力（TaskTool 由 app 显式注册）
    assert ha.subagent_enabled is True


def test_framework_config_subagent_flags_off(cfg_factory):
    from cortex.config import framework_config

    ha = framework_config(cfg_factory(), subagent=True)
    assert ha.subagent_enabled is False
    assert ha.todowrite_enabled is False
    assert ha.devlog_enabled is False


def test_mcp_servers_json_parsing(cfg_factory):
    cfg = cfg_factory(mcp_servers_json='[{"name":"fetch","command":"uvx"}]')
    assert cfg.mcp_servers() == [{"name": "fetch", "command": "uvx"}]
    empty = cfg_factory()
    assert empty.mcp_servers() == []
    bad = cfg_factory(mcp_servers_json="{}")
    with pytest.raises(Exception):
        bad.mcp_servers()


def test_rebase_moves_derived_paths(cfg_factory, tmp_path):
    cfg = cfg_factory()
    old_data = cfg.data_dir
    assert old_data.exists()
    new_root = tmp_path / "elsewhere"
    new_root.mkdir()
    cfg.rebase(new_root)
    assert cfg.project_root == new_root.resolve()
    assert cfg.data_dir == new_root / "data"
    assert cfg.qdrant_local_path == new_root / "data" / "qdrant_local"
    assert cfg.notes_dir == new_root / "data" / "notes"
    cfg.ensure_dirs()
    assert (new_root / "data" / "notes").exists()


def test_rebase_keeps_explicit_external_path(cfg_factory, tmp_path):
    external_root = tmp_path / "ext-root"
    cfg = cfg_factory(project_root=external_root, data_dir=tmp_path / "explicit-data")
    external = (tmp_path / "explicit-data").resolve()
    cfg.rebase(tmp_path / "other-root")
    assert Path(cfg.data_dir) == external  # 显式外部路径不被迁移


# ------------------------------------------------ 启动期配置校验 ---------- #
def test_memory_weights_must_sum_to_one(cfg_factory):
    from cortex.config import CortexConfigError

    with pytest.raises(CortexConfigError, match="记忆权重"):
        cfg_factory(memory_w_similarity=0.5, memory_w_importance=0.25,
                    memory_w_recency=0.15)  # 和=0.9


def test_negative_memory_weight_rejected_even_if_sum_ok(cfg_factory):
    from cortex.config import CortexConfigError

    with pytest.raises(CortexConfigError, match="记忆权重"):
        cfg_factory(memory_w_similarity=-0.1, memory_w_importance=1.0,
                    memory_w_recency=0.1)


def test_chunk_overlap_must_be_lt_size(cfg_factory):
    from cortex.config import CortexConfigError

    with pytest.raises(CortexConfigError, match="chunk_overlap"):
        cfg_factory(chunk_size=64, chunk_overlap=64)
    with pytest.raises(CortexConfigError, match="chunk_overlap"):
        cfg_factory(chunk_size=64, chunk_overlap=-1)


def test_negative_timeouts_rejected(cfg_factory):
    from cortex.config import CortexConfigError

    with pytest.raises(CortexConfigError, match="CORTEX_STAGE_TIMEOUT"):
        cfg_factory(stage_timeout=-5)
    with pytest.raises(CortexConfigError, match="WEAK_TIMEOUT"):
        cfg_factory(weak_timeout=-1)


def test_zero_stage_timeout_allowed_but_not_llm_timeouts(cfg_factory):
    # stage 0 = 不限时（合法）；LLM 超时为 0 也放行（不校验上限，只挡负数）
    cfg = cfg_factory(stage_timeout=0, stage_timeout_write=0, llm_timeout=0,
                      strong_timeout=0, weak_timeout=0)
    assert cfg.stage_timeout == 0
