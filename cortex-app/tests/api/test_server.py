"""API 离线测试：桩 CortexApp（假 pipeline/假知识库/假编排器）+ TestClient，零 LLM。"""

from __future__ import annotations

import json
import re
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

    def list_documents(self):
        return [{"doc_id": "d1", "title": "样例文档", "source": None,
                 "n_chunks": 3, "created_at": "2026-01-01T00:00:00"}]

    def delete(self, doc_id):
        return doc_id == "d1"


class _StubMemory:
    def __init__(self) -> None:
        self.remembered: list[dict] = []

    def remember(self, *, content, type="episodic", source="", metadata=None, **kw):
        self.remembered.append({"content": content, "type": type,
                                "source": source, "metadata": metadata})


class _StubHistory:
    def __init__(self, outer: "_StubOrchestrator") -> None:
        self._outer = outer
        self.appended: list[tuple[str, str]] = []

    def clear(self) -> None:
        self._outer.cleared += 1

    def append(self, message) -> None:  # hello_agents Message 鸭子类型
        self.appended.append((message.role, message.content))


class _StubOrchestrator:
    def __init__(self) -> None:
        self.cleared = 0
        self.calls: list[str] = []
        self.history_manager = _StubHistory(self)

    def run(self, text: str) -> str:
        self.calls.append(text)
        return f"回答：{text}"


class _StubResult:
    def __init__(self) -> None:
        self.report_md = "# 桩研报\n\n内容"
        self.revised = False
        self.duration_seconds = 0.1
        self.mode = "pipeline"


class _StubPipeline:
    def __init__(self) -> None:
        self.ran: list[str] = []
        self.progress_cb = None

    def run(self, topic: str) -> _StubResult:
        if self.progress_cb is not None:
            self.progress_cb({"type": "stage_start", "stage": "gather", "run_elapsed": 0.0})
            self.progress_cb({"type": "stage_end", "stage": "gather", "ok": True,
                              "timed_out": False, "elapsed": 0.1, "run_elapsed": 0.1})
            self.progress_cb({"type": "note", "code": "revise_skipped", "msg": "跳过"})
        self.ran.append(topic)
        return _StubResult()


class _StubApp:
    def __init__(self, api_key: str = "", **cfg_extra) -> None:
        self.cfg = SimpleNamespace(api_key=api_key, **cfg_extra)
        self.weak_llm = None
        self.services = {"knowledge": _StubKB(), "memory": _StubMemory()}
        self.agents = SimpleNamespace(orchestrator=_StubOrchestrator())
        self.pipeline = _StubPipeline()

    def build_pipeline(self, progress_cb=None) -> _StubPipeline:
        pipe = _StubPipeline()
        pipe.progress_cb = progress_cb
        return pipe


def _make_client(stub=None):
    from cortex.api.server import create_app

    app = create_app(cortex_app=stub if stub is not None else _StubApp())
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def client():
    yield from _make_client()


def _sse_data(body: str, event: str) -> list[dict]:
    """从 SSE 文本中取某事件的全部 data 载荷（JSON 解析失败即断言失败）。"""
    out = []
    for m in re.finditer(rf"event: {event}\s*\ndata: (.*)", body):
        out.append(json.loads(m.group(1).strip()))
    return out


# ---------------------------------------------------------------- 用例 ---- #
def test_research_sse_streams_progress_and_done(client):
    with client.stream("POST", "/research", json={"topic": "测试主题"}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = "".join(resp.iter_text())
    done = _sse_data(body, "done")
    assert done and "桩研报" in done[0]["report_md"]
    assert client.app and True  # TestClient 冒烟
    # 桩 pipeline 确实收到主题
    # （client.app 是 FastAPI；pipeline 在 lifespan 注入的 cortex_app 上）


def test_research_emits_structured_progress_json(client):
    with client.stream("POST", "/research", json={"topic": "t"}) as resp:
        body = "".join(resp.iter_text())
    # 全部事件 data 均为合法 JSON；progress 含结构化阶段事件
    starts = _sse_data(body, "progress")
    assert any(p.get("type") == "stage_start" and p.get("stage") == "gather" for p in starts)
    assert any(p.get("type") == "stage_end" and p.get("stage") == "gather"
               and p.get("ok") is True for p in starts)
    done = _sse_data(body, "done")
    assert {"report_md", "revised", "duration_seconds", "mode"} <= set(done[0])


def test_research_done_persists_report_file(tmp_path):
    from cortex.api.server import create_app

    stub = _StubApp(data_dir=tmp_path)
    with TestClient(create_app(cortex_app=stub)) as c:
        with c.stream("POST", "/research", json={"topic": "持久化检查"}) as resp:
            body = "".join(resp.iter_text())
    reports = list((tmp_path / "reports").glob("*.md"))
    assert reports, "done 后报告应落盘 data/reports/"
    assert "桩研报" in reports[0].read_text(encoding="utf-8")
    assert _sse_data(body, "done")[0]["report_path"] == str(reports[0])


def test_research_busy_returns_409(client):
    from cortex.api.server import _research_running

    _research_running.set()
    try:
        resp = client.post("/research", json={"topic": "t"})
        assert resp.status_code == 409
    finally:
        _research_running.clear()


def test_research_rejects_bad_mode(client):
    resp = client.post("/research", json={"topic": "t", "mode": "nope"})
    assert resp.status_code == 422


def test_ingest_text_roundtrip(client):
    resp = client.post("/ingest-text",
                       json={"title": "样例", "content": "一些内容" * 20})
    assert resp.status_code == 200
    data = resp.json()
    assert data["doc_id"] == "d2" and data["n_chunks"] >= 1


def test_documents_list_and_delete_flow(client):
    lst = client.get("/documents").json()
    assert lst["total"] == 1
    doc = lst["documents"][0]
    assert doc["doc_id"] == "d1" and doc["n_chunks"] == 3
    assert doc["source"] == ""  # 库里 NULL 归一为空串
    assert client.delete("/documents/d1").json() == {"deleted": True, "doc_id": "d1"}
    assert client.delete("/documents/missing").status_code == 404


def test_chat_stream_message_and_session_roundtrip(client):
    with client.stream("POST", "/chat/stream", json={"message": "你好"}) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())
    sessions_evt = _sse_data(body, "session")
    messages = _sse_data(body, "message")
    dones = _sse_data(body, "done")
    assert sessions_evt and sessions_evt[0]["session_id"]
    assert messages and "回答：你好" in messages[0]["content"]
    assert dones and dones[0]["turns"] >= 1
    sid = sessions_evt[0]["session_id"]
    got = client.get(f"/session/{sid}")
    assert got.status_code == 200
    history = got.json()["history"]
    assert history[0]["role"] == "user" and history[0]["content"] == "你好"
    assert history[1]["role"] == "assistant"


def test_chat_reset_clears_orchestrator_history():
    from cortex.api.server import create_app

    stub = _StubApp()
    with TestClient(create_app(cortex_app=stub)) as c:
        with c.stream("POST", "/chat/stream",
                      json={"message": "重置我", "reset": True}) as resp:
            "".join(resp.iter_text())
    orch = stub.agents.orchestrator
    assert orch.calls == ["重置我"]
    assert orch.cleared == 1


def test_chat_multi_turn_injects_transcript_prefix():
    """同会话第二轮必须带上首轮问答转录（PlanSolveAgent 不读 history_manager）。"""
    from cortex.api.server import create_app

    stub = _StubApp()
    with TestClient(create_app(cortex_app=stub)) as c:
        # 第一轮（新会话）
        sid = ""
        with c.stream("POST", "/chat/stream", json={"message": "我叫小乙烯"}) as resp:
            for line in resp.iter_lines():
                if line.startswith("data:") and "session_id" in line:
                    sid = line.split(":", 1)[1].strip()
        import json as _json

        sid = _json.loads(sid)["session_id"]
        # 第二轮（同会话，不 reset）
        with c.stream("POST", "/chat/stream",
                      json={"message": "我叫什么？", "session_id": sid}) as resp:
            "".join(resp.iter_text())
    orch = stub.agents.orchestrator
    assert orch.calls[0] == "我叫小乙烯"
    second = orch.calls[1]
    assert "【对话历史】" in second
    assert "user: 我叫小乙烯" in second or "用户: 我叫小乙烯" in second
    assert second.endswith("【当前消息】\n我叫什么？")


def test_chat_reset_clears_session_history_too():
    """reset 除清编排器历史外，还须清会话转录——防旧 sid 复活后粘合两段历史。"""
    import json as _json

    from cortex.api.server import create_app

    stub = _StubApp()
    with TestClient(create_app(cortex_app=stub)) as c:
        sid = ""
        with c.stream("POST", "/chat/stream", json={"message": "第一轮"}) as resp:
            for line in resp.iter_lines():
                if line.startswith("data:") and "session_id" in line:
                    sid = _json.loads(line.split(":", 1)[1].strip())["session_id"]
        with c.stream("POST", "/chat/stream",
                      json={"message": "重来", "session_id": sid, "reset": True}) as resp:
            body = "".join(resp.iter_text())
    assert '"turns": 1' in body  # reset 后从 0 计数
    orch = stub.agents.orchestrator
    assert orch.calls[0] == "第一轮"
    assert orch.calls[1] == "重来"  # reset → 无转录前缀


def test_chat_busy_when_orchestrator_locked():
    """编排器被研究（或另一轮对话）占用时，chat 快速失败而非排队。"""
    import cortex.api.server as server_mod
    from cortex.api.server import create_app

    stub = _StubApp()
    app = create_app(cortex_app=stub)
    server_mod._agent_gate.acquire()
    try:
        with TestClient(app) as c:
            with c.stream("POST", "/chat/stream", json={"message": "你好"}) as resp:
                body = "".join(resp.iter_text())
        assert "研究/探索任务进行中" in body
        assert stub.agents.orchestrator.calls == []  # 未进入编排器
    finally:
        server_mod._agent_gate.release()


def test_chat_memory_write_through_called():
    from cortex.api.server import create_app

    stub = _StubApp()
    with TestClient(create_app(cortex_app=stub)) as c:
        with c.stream("POST", "/chat/stream", json={"message": "记住我"}) as resp:
            "".join(resp.iter_text())
    mem = stub.services["memory"]
    assert len(mem.remembered) == 1
    assert "记住我" in mem.remembered[0]["content"]
    assert mem.remembered[0]["type"] == "episodic"
    assert mem.remembered[0]["source"] == "web-chat"


def test_health_reports_services_auth_uptime(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["knowledge"]["documents"] == 1
    assert body["services"]["knowledge"] is True
    assert body["services"]["memory"] is True
    assert body["auth_required"] is False
    assert isinstance(body["uptime_seconds"], (int, float))


def test_session_create_get_and_404(client):
    created = client.post("/session", json={"history": [{"role": "user", "content": "hi"}]})
    assert created.status_code == 200
    sid = created.json()["session_id"]
    assert created.json()["turns"] == 1
    got = client.get(f"/session/{sid}")
    assert got.status_code == 200
    assert got.json()["history"][0]["content"] == "hi"
    assert client.get("/session/missing").status_code == 404


def test_cors_headers_when_configured():
    from cortex.api.server import create_app

    app = create_app(cortex_app=_StubApp(cors_origins="http://localhost:5173"))
    with TestClient(app) as c:
        preflight = c.options(
            "/ingest-text",
            headers={"Origin": "http://localhost:5173",
                     "Access-Control-Request-Method": "POST"},
        )
        assert preflight.headers["access-control-allow-origin"] == "http://localhost:5173"
        # 未列入白名单的来源不带放行头
        other = c.options(
            "/ingest-text",
            headers={"Origin": "http://evil.example",
                     "Access-Control-Request-Method": "POST"},
        )
        assert "access-control-allow-origin" not in other.headers


def test_cors_disabled_by_default(client):
    preflight = client.options(
        "/ingest-text",
        headers={"Origin": "http://localhost:5173",
                 "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in preflight.headers


def test_spa_fallback_serves_index(tmp_path, monkeypatch):
    from cortex.api.server import create_app

    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>spa-index</html>", encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log(1);", encoding="utf-8")
    monkeypatch.setenv("CORTEX_DIST_DIR", str(dist))
    with TestClient(create_app(cortex_app=_StubApp())) as c:
        assert "spa-index" in c.get("/").text
        assert "spa-index" in c.get("/chat").text  # BrowserRouter 子路由回退 index
        assert c.get("/assets/app.js").text == "console.log(1);"
        assert c.get("/health").json()["status"] == "ok"  # API 路由不被兜底遮蔽


def test_api_key_enforced_when_configured():
    from cortex.api.server import create_app

    app = create_app(cortex_app=_StubApp(api_key="secret"))
    with TestClient(app) as c:
        denied = c.post("/ingest-text", json={"title": "t", "content": "c"})
        assert denied.status_code == 401
        ok = c.post("/ingest-text", json={"title": "t", "content": "c"},
                    headers={"X-API-Key": "secret"})
        assert ok.status_code == 200
