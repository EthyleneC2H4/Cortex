"""FastAPI + SSE 服务。

- POST /research：SSE 流式进度 + 完整研报（worker 线程跑 pipeline →
  queue.Queue → asyncio.to_thread 排水桥接）
- POST /ingest：上传文件入知识库（也支持 JSON 文本摄取 /ingest-text）
- POST /session + GET /session/{id}：会话历史存取（进程内 dict，v1 无 Redis）
- X-API-Key 鉴权（cfg.api_key 为空则开放）

参考框架 examples/fastapi_sse_server.py 的 lifespan 优雅关闭模式。
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from sse_starlette.sse import EventSourceResponse

from .schemas import IngestTextRequest, ResearchRequest, SessionCreateRequest, SessionInfo


def create_app(cortex_app=None):
    """构建 FastAPI 应用；cortex_app 缺省时在 lifespan 中现场构建。"""
    sessions: dict[str, SessionInfo] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal cortex_app
        if cortex_app is None:
            from ..app import CortexApp

            cortex_app = CortexApp.build()
        mem = cortex_app.services.get("memory")
        yield
        # 优雅关闭：flush 记忆队列 → 关服务 → 关 MCP
        if mem is not None:
            flush = getattr(mem, "flush_writes", None)
            try:
                if flush is not None:
                    flush(timeout=10)
            finally:
                close = getattr(mem, "close", None)
                if close:
                    close()
        kb = cortex_app.services.get("knowledge")
        if kb is not None and hasattr(kb, "close"):
            kb.close()
        mcp = cortex_app.services.get("mcp")
        if mcp is not None and hasattr(mcp, "close"):
            mcp.close()

    app = FastAPI(title="Cortex API", version="0.1.0", lifespan=lifespan)

    # ---- 鉴权 ------------------------------------------------------- #
    def require_key(request: Request) -> None:
        cfg = cortex_app.cfg if cortex_app else None
        expected = getattr(cfg, "api_key", "") if cfg else ""
        if not expected:
            return
        got = request.headers.get("X-API-Key", "")
        if got != expected:
            raise HTTPException(status_code=401, detail="无效 X-API-Key")

    # ---- POST /research（SSE） -------------------------------------- #
    @app.post("/research")
    async def research(req: ResearchRequest, _: None = Depends(require_key)):
        app_ref = cortex_app
        if app_ref is None:
            raise HTTPException(status_code=503, detail="应用尚未初始化")

        events: queue.Queue = queue.Queue()

        def worker() -> None:
            try:
                if req.mode == "explore":
                    from ..agents.dispatch import ExplorationMode

                    events.put({"stage": "plan", "msg": "编排器规划中"})
                    result = ExplorationMode(app_ref.agents.orchestrator).run(req.topic)
                else:
                    pipe = app_ref.pipeline
                    result = pipe.run(req.topic)
                events.put({"done": True, "report_md": result.report_md,
                            "revised": result.revised,
                            "duration_seconds": result.duration_seconds})
            except Exception as exc:
                events.put({"error": str(exc)})

        threading.Thread(target=worker, daemon=True).start()

        async def stream():
            while True:
                item = await asyncio.to_thread(events.get)
                if "error" in item:
                    yield {"event": "error", "data": item["error"]}
                    break
                if item.get("done"):
                    yield {"event": "done", "data": item["report_md"]}
                    break
                yield {"event": "progress", "data": f"{item['stage']}: {item['msg']}"}

        return EventSourceResponse(stream())

    # ---- POST /ingest ----------------------------------------------- #
    @app.post("/ingest")
    async def ingest(file: UploadFile = File(...), _: None = Depends(require_key)):
        kb = cortex_app.services.get("knowledge") if cortex_app else None
        if kb is None:
            raise HTTPException(status_code=503, detail="知识库服务未初始化")
        upload_dir = Path("knowledge_base/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        dest = upload_dir / f"{uuid.uuid4().hex[:8]}-{file.filename}"
        dest.write_bytes(await file.read())
        try:
            info = kb.ingest_document(dest, weak_llm=cortex_app.weak_llm)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"摄取失败: {exc}") from exc
        return {"doc_id": info["doc_id"], "title": info["title"], "n_chunks": info["n_chunks"]}

    @app.post("/ingest-text")
    async def ingest_text(req: IngestTextRequest, _: None = Depends(require_key)):
        kb = cortex_app.services.get("knowledge") if cortex_app else None
        if kb is None:
            raise HTTPException(status_code=503, detail="知识库服务未初始化")
        info = kb.ingest_text(title=req.title, text=req.content, weak_llm=cortex_app.weak_llm)
        return {"doc_id": info["doc_id"], "n_chunks": info["n_chunks"]}

    # ---- 会话 ------------------------------------------------------- #
    @app.post("/session", response_model=SessionInfo)
    async def create_session(req: SessionCreateRequest, _: None = Depends(require_key)):
        sid = uuid.uuid4().hex[:12]
        info = SessionInfo(session_id=sid, created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                           turns=len(req.history), history=req.history)
        sessions[sid] = info
        return info

    @app.get("/session/{session_id}", response_model=SessionInfo)
    async def get_session(session_id: str, _: None = Depends(require_key)):
        info = sessions.get(session_id)
        if info is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        return info

    @app.get("/health")
    async def health():
        stats = {}
        kb = cortex_app.services.get("knowledge") if cortex_app else None
        if kb is not None and hasattr(kb, "stats"):
            try:
                stats["knowledge"] = kb.stats()
            except Exception:
                pass
        return {"status": "ok", **stats}

    return app


def main() -> None:
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
