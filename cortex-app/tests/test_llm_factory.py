"""P1 单元测试：双档 LLM 构造（MagicMock 替身，不发真实请求）。"""

from __future__ import annotations

import pytest

from cortex.config import CortexConfigError

pytestmark = pytest.mark.unit


@pytest.fixture
def patched_llm(monkeypatch):
    from cortex.llm import factory as mod

    fake = pytest.importorskip("unittest.mock").MagicMock
    instances = []

    def _ctor(**kwargs):
        inst = fake(**kwargs)
        instances.append(inst)
        return inst

    monkeypatch.setattr(mod, "HelloAgentsLLM", _ctor)
    return mod, instances


def test_build_llms_dual_tier_kwargs(cfg_factory, patched_llm):
    mod, instances = patched_llm
    cfg = cfg_factory(strong_temperature=0.3, strong_timeout=120, weak_temperature=0.0, weak_timeout=60)
    strong, weak = mod.build_llms(cfg)

    assert len(instances) == 2
    assert strong.model == "test/model-x" and weak.model == "test/model-x"  # 同模型起步
    assert strong.temperature == 0.3 and strong.timeout == 120
    assert weak.temperature == 0.0 and weak.timeout == 60
    # 全部显式传参：不依赖框架内部读 env
    assert strong.api_key == cfg.llm_api_key and strong.base_url == cfg.llm_base_url


def test_build_llms_separate_model_ids(cfg_factory, patched_llm):
    mod, _ = patched_llm
    cfg = cfg_factory(strong_model_id="vendor/big", weak_model_id="vendor/small")
    strong, weak = mod.build_llms(cfg, model_id="ignored")
    assert strong.model == "vendor/big" and weak.model == "vendor/small"


def test_missing_api_key_raises_before_construction(cfg_factory, patched_llm):
    mod, instances = patched_llm
    cfg = cfg_factory(llm_api_key="")
    with pytest.raises(CortexConfigError, match="LLM_API_KEY"):
        mod.build_llms(cfg)
    assert instances == []


def test_ping_wraps_errors(cfg_factory, patched_llm):
    mod, _instances = patched_llm
    from unittest.mock import MagicMock

    good = MagicMock()
    bad = MagicMock()
    bad.invoke.side_effect = RuntimeError("boom")  # side_effect 须挂在 invoke 子 mock 上
    from cortex.llm.factory import ping

    assert ping(good) is True
    assert ping(bad) is False
