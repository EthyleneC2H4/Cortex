"""P1 单元测试：模型 ID 解析（纯离线，httpx MockTransport）。"""

from __future__ import annotations

import json
import time

import httpx
import pytest

from cortex.config import CortexConfigError
from cortex.llm.model_resolver import pick_model_id, resolve_model_id

pytestmark = pytest.mark.unit

MODELS = [
    {"id": "z/gpt"},
    {"id": "openrouter/a:free"},
    {"id": "stealth/ox-alpha"},
    {"id": "stealth/ox-alpha:free"},
]


def _transport(status=200, models=MODELS, seen=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        # base_url 带路径时（…/api/v1），完整 path 为 /api/v1/models
        if not request.url.path.endswith("/models"):
            return httpx.Response(404)
        if status == 200:
            return httpx.Response(200, json={"data": models})
        return httpx.Response(status, json={"error": "denied"})

    return httpx.MockTransport(handler)


class TestPickModelId:
    def test_prefers_alpha_free(self):
        assert pick_model_id([m["id"] for m in MODELS]) == "stealth/ox-alpha:free"

    def test_alpha_any_over_plain_free(self):
        assert pick_model_id(["openrouter/a:free", "vendor/x-alpha"]) == "vendor/x-alpha"

    def test_free_fallback(self):
        assert pick_model_id(["b:free", "a:free", "z/gpt"]) == "a:free"

    def test_last_resort_first_sorted(self):
        assert pick_model_id(["z/gpt", "a/model"]) == "a/model"

    def test_empty_returns_none(self):
        assert pick_model_id([]) is None


class TestResolve:
    def test_explicit_short_circuits_no_http(self):
        calls = []
        cfg = _cfg(llm_model_id="manual/model")
        got = resolve_model_id(cfg, transport=_transport(seen=calls))
        assert got == "manual/model"
        assert calls == []

    def test_live_resolution_writes_cache(self, tmp_path):
        cfg = _cfg(tmp_path, llm_model_id="")
        got = resolve_model_id(cfg, transport=_transport())
        assert got == "stealth/ox-alpha:free"
        cache = json.loads((cfg.data_dir / "model_cache.json").read_text())
        assert cache["model_id"] == "stealth/ox-alpha:free"

    def test_fresh_cache_skips_http(self, tmp_path):
        cfg = _cfg(tmp_path, llm_model_id="")
        cfg.ensure_dirs()
        (cfg.data_dir / "model_cache.json").write_text(
            json.dumps(
                {
                    "model_id": "cached/m",
                    "base_url": cfg.llm_base_url,
                    "resolved_at": time.time(),
                }
            )
        )
        calls = []

        class Boom(httpx.MockTransport):
            pass

        def handler(request):  # pragma: no cover —— 命中缓存则不应触网
            calls.append(request)
            return httpx.Response(500)

        got = resolve_model_id(cfg, transport=httpx.MockTransport(handler))
        assert got == "cached/m" and calls == []

    def test_stale_base_url_ignores_cache(self, tmp_path):
        cfg = _cfg(tmp_path, llm_model_id="", llm_base_url="https://other.example/v1")
        cfg.ensure_dirs()
        (cfg.data_dir / "model_cache.json").write_text(
            json.dumps({"model_id": "old/m", "base_url": "https://openrouter.ai/api/v1", "resolved_at": time.time()})
        )
        got = resolve_model_id(cfg, transport=_transport())
        assert got == "stealth/ox-alpha:free"

    def test_auth_failure_actionable_message(self, tmp_path):
        cfg = _cfg(tmp_path, llm_model_id="", llm_api_key="bad")
        with pytest.raises(CortexConfigError, match="LLM_API_KEY"):
            resolve_model_id(cfg, transport=_transport(status=401))

    def test_no_models_raises(self, tmp_path):
        cfg = _cfg(tmp_path, llm_model_id="")
        with pytest.raises(CortexConfigError, match="LLM_MODEL_ID"):
            resolve_model_id(cfg, transport=_transport(models=[]))


def _cfg(tmp_path=None, **overrides):
    from cortex.config import CortexConfig

    defaults = dict(
        llm_api_key="good",
        project_root=tmp_path or "/tmp/cortex-resolver-test",
    )
    defaults.update(overrides)
    cfg = CortexConfig(**defaults)
    cfg.ensure_dirs()
    return cfg
