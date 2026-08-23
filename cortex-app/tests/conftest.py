"""共享测试夹具。

- 会话启动时加载项目根 .env（提供 LLM_API_KEY / HAS_LLM_KEY 等）；
- ``cfg_factory`` 构造数据目录落在 tmp_path 的 CortexConfig，测试间零共享状态。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


@pytest.fixture
def cfg_factory(tmp_path: Path):
    """返回构造函数：make_cfg(**overrides) -> CortexConfig（数据目录在 tmp_path）。"""
    from cortex.config import CortexConfig

    def _make(**overrides):
        defaults = dict(
            llm_base_url="https://openrouter.ai/api/v1",
            llm_api_key=os.getenv("LLM_API_KEY", "test-key"),
            llm_model_id="test/model-x",
            project_root=tmp_path,
            store_mode="local",
        )
        defaults.update(overrides)
        cfg = CortexConfig(**defaults)
        cfg.ensure_dirs()
        return cfg

    return _make
