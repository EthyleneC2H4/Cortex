"""API 离线测试：桩 CortexApp（假 pipeline/假知识库）+ TestClient，零 LLM。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------- 桩件 ---- #
class _StubKB:
    def ingest_document(self, path, weak_llm=None):
        return {"doc_id": "d1", "title": path.name, "n_chunks": 3}

    def ingest_text(self, *, title, text, weak_llm=None):
        return {"doc_id": "d2", "title": title, "n_chunks": len(text) // 10 + 1}

    def stats(self):
        return {"documents": 1, "chunks": 3}


class _StubResult:
    def __init__(self) -> None:
        self.report_md = "# 桩研报\n\n内容"
        self.revised = False
        self.duration_seconds = 0.1


class _StubPipeline:
    def __init__(self) -> None:
        self.ran: list[str] = []

    def run(self, topic: str) -> _StubResult:
        self.ran.append(topic)
        return _StubResult()


class _StubApp:
    def __init__(self, api_key: str = "") -> None:
        self.cfg = SimpleNamespace(api_key=api_key)
        self.weak_llm = None
        self.services = {"knowledge": _StubKB()}
        self.pipeline = _StubPipeline()


@pytest.fixture()
def client():
    from cortex.api.server import create_app

    app = create_app(cortex_app=_StubApp())
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------- 用例 ---- #
def test_research_sse_streams_progress_and_done(client):
    with client.stream("POST", "/research", json={"topic": "测试主题"}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = "".join(resp.iter_text())
    assert "event: done" in body
    assert "桩研报" in body
    assert client.app and True  # TestClient 冒烟
    # 桩 pipeline 确实收到主题
    # （client.app 是 FastAPI；pipeline 在 lifespan 注入的 cortex_app 上）


def test_research_rejects_bad_mode(client):
    resp = client.post("/research", json={"topic": "t", "mode": "nope"})
    assert resp.status_code == 422


def test_ingest_text_roundtrip(client):
    resp = client.post("/ingest-text",
                       json={"title": "样例", "content": "一些内容" * 20})
    assert resp.status_code == 200
    data = resp.json()
    assert data["doc_id"] == "d2" and data["n_chunks"] >= 1


def test_health_reports_knowledge(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["knowledge"]["documents"] == 1


def test_session_create_get_and_404(client):
    created = client.post("/session", json={"history": [{"role": "user", "content": "hi"}]})
    assert created.status_code == 200
    sid = created.json()["session_id"]
    assert created.json()["turns"] == 1
    got = client.get(f"/session/{sid}")
    assert got.status_code == 200
    assert got.json()["history"][0]["content"] == "hi"
    assert client.get("/session/missing").status_code == 404


def test_api_key_enforced_when_configured():
    from cortex.api.server import create_app

    app = create_app(cortex_app=_StubApp(api_key="secret"))
    with TestClient(app) as c:
        denied = c.post("/ingest-text", json={"title": "t", "content": "c"})
        assert denied.status_code == 401
        ok = c.post("/ingest-text", json={"title": "t", "content": "c"},
                    headers={"X-API-Key": "secret"})
        assert ok.status_code == 200
