"""P1 live 冒烟：真实 OpenRouter 连通性。仅 HAS_LLM_KEY=1 时运行（pytest -m e2e）。"""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.e2e]

requires_key = pytest.mark.skipif(
    not os.getenv("HAS_LLM_KEY"), reason="需要 HAS_LLM_KEY=1 且 .env 配好 LLM_API_KEY"
)


@requires_key
def test_dual_tier_ping_live():
    from cortex.app import CortexApp
    from cortex.llm.factory import ping

    app = CortexApp.build()
    assert app.agent is not None
    assert ping(app.strong_llm), "strong LLM ping 失败"
    assert ping(app.weak_llm), "weak LLM ping 失败"
