"""FastAPI + SSE 服务。

- POST /research：SSE 结构化进度（stage_start/stage_end/note）+ 完整研报
  （worker 线程跑 pipeline → queue.Queue → asyncio.to_thread 排水桥接）；
  全部事件 data 为 JSON；done 附带落盘路径，断连后可从 data/reports/ 找回
- POST /chat/stream：对话 SSE（session → message → done/error），复用
  /research 的 worker+queue 桥；整轮返回（框架同步 run 无 token 流）
- POST /ingest / /ingest-text：文件与文本摄取入知识库（大小上限 CORTEX_MAX_UPLOAD_MB，
  默认 50MB）
- GET /documents + DELETE /documents/{doc_id}：知识库文档列表与删除
- GET /reports + GET /reports/{filename}：研究报告列表与取回（断连后从 data/reports/ 找回）
- POST /session + GET /session/{id}：会话历史存取（进程内 LRU+TTL 存储，v1 无 Redis；
  容量 CORTEX_SESSION_MAX 默认 1000、TTL CORTEX_SESSION_TTL_HOURS 默认 24h，重启即失）
- GET /health：服务健康 + 版本/运行时长/各服务在位状态/知识库统计
- X-API-Key 鉴权（cfg.api_key 为空则开放）；CORTEX_CORS_ORIGINS 非空时启用 CORS；
  frontend/dist 存在时托管 SPA（生产单命令部署）

参考框架 examples/fastapi_sse_server.py 的 lifespan 优雅关闭模式。
"""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import queue
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from .schemas import (
    ChatRequest,
    DocumentList,
    DocumentOut,
    IngestTextRequest,
    ResearchRequest,
    SessionCreateRequest,
    SessionInfo,
)

# 免费档限流保护：同进程同一时刻只允许一个研究任务（运行中提交 → 409）
_research_running = threading.Event()
# 编排器互斥：explore 研究 / chat 共享同一 orchestrator 实例，框架的
# run_as_subagent 会对 history_manager 与共享 ToolRegistry 做非线程安全的
# 临时摘除——并发调用互相破坏状态，必须应用层串行化（禁改框架源码）。
_agent_gate = threading.Lock()


def _humanize_error(exc: Exception) -> str:
    """把底层异常翻译成可操作的中文短句（原始串放 detail 供排查）。"""
    text = str(exc).lower()
    if any(k in text for k in ("quota", "rate limit", "429", "限流")):
        return "模型配额或限流受限，请稍后重试"
    if any(k in text for k in ("timeout", "timed out", "超时")):
        return "执行超时，请稍后重试或缩小任务范围"
    if any(k in text for k in ("connection", "network", "unreachable", "proxy")):
        return "网络连接失败，请检查网络或代理设置"
    if any(k in text for k in ("auth", "api key", "401", "unauthorized")):
        return "鉴权失败，请检查 LLM_API_KEY 配置"
    return "执行出错，请稍后重试"


def _save_report(cfg, topic: str, report_md: str) -> str:
    """报告落盘到 data/reports/：客户端断连后仍可找回结果。失败静默（尽力而为）。"""
    data_dir = getattr(cfg, "data_dir", None)
    if not data_dir:
        return ""
    try:
        reports_dir = Path(data_dir) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        slug = "".join(ch if ch.isalnum() else "-" for ch in topic)[:32].strip("-") or "topic"
        out = reports_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{slug}.md"
        out.write_text(report_md, encoding="utf-8")
        return str(out)
    except OSError:
        return ""


class _SessionStore:
    """进程内会话存储：LRU 容量上限 + TTL 过期（v1 无 Redis 的防无界增长折中）。

    - get 命中即续期（滑动 TTL）：活跃会话不会因持续使用被过期清掉
    - 超过容量上限时按最久未活跃淘汰
    - 淘汰/过期留有界墓碑：GET /session 可区分「曾存在但已失效」(410) 与「从未存在」(404)
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 24 * 3600.0,
                 time_fn=time.time) -> None:
        self._data: OrderedDict[str, tuple[SessionInfo, float]] = OrderedDict()
        self._evicted: OrderedDict[str, None] = OrderedDict()  # 有界墓碑，容量同 max
        self._lock = threading.Lock()
        self._max = max(1, int(max_size))
        self._ttl = float(ttl_seconds)
        self._time_fn = time_fn

    def get(self, session_id: str) -> SessionInfo | None:
        now = self._time_fn()
        with self._lock:
            entry = self._data.get(session_id)
            if entry is None:
                return None
            info, ts = entry
            if now - ts > self._ttl:
                del self._data[session_id]
                self._mark_evicted(session_id)
                return None
            self._data.move_to_end(session_id)
            self._data[session_id] = (info, now)  # 滑动续期
            return info

    def put(self, session_id: str, info: SessionInfo) -> None:
        now = self._time_fn()
        with self._lock:
            self._data[session_id] = (info, now)
            self._data.move_to_end(session_id)
            while len(self._data) > self._max:
                stale, _ = self._data.popitem(last=False)
                self._mark_evicted(stale)

    def evicted_recently(self, session_id: str) -> bool:
        """该会话是否曾被本进程持有后因 TTL/容量被清理（区别于从未存在）。"""
        with self._lock:
            return session_id in self._evicted

    def _mark_evicted(self, session_id: str) -> None:
        self._evicted[session_id] = None
        self._evicted.move_to_end(session_id)
        while len(self._evicted) > self._max:
            self._evicted.popitem(last=False)


def _mount_spa(fastapi_app: FastAPI, dist_dir: Path) -> bool:
    """dist 存在时挂载 SPA：/assets 静态目录 + 兜底路由回退 index.html。

    不用 StaticFiles(html=True) 直挂根——BrowserRouter 子路由（如 /chat）会 404。
    必须在全部 API 路由注册之后调用（FastAPI 按注册顺序匹配）。
    """
    index_html = dist_dir / "index.html"
    if not index_html.exists():
        return False

    assets_dir = dist_dir / "assets"
    if assets_dir.exists():
        fastapi_app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    dist_root = str(dist_dir.resolve())

    @fastapi_app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        candidate = (dist_dir / full_path).resolve()
        # 防目录穿越：仅当解析后仍在 dist 内且是真实文件时直出
        if full_path and str(candidate).startswith(dist_root) and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index_html)

    return True


def create_app(cortex_app=None):
    """构建 FastAPI 应用；cortex_app 缺省时在 lifespan 中现场构建。"""
    # 会话存储：LRU+TTL 防无界增长（env 可调；测试可注入更小阈值）
    session_max = int(os.environ.get("CORTEX_SESSION_MAX", "1000") or "1000")
    session_ttl_h = float(os.environ.get("CORTEX_SESSION_TTL_HOURS", "24") or "24")
    sessions = _SessionStore(max_size=session_max, ttl_seconds=session_ttl_h * 3600.0)
    started_at = time.time()

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

    app = FastAPI(title="Cortex API", version="0.2.0", lifespan=lifespan)

    # ---- CORS（默认关闭：开发走 vite 代理、生产同源；跨域部署才配置） -- #
    cfg = getattr(cortex_app, "cfg", None)
    cors_raw = str(getattr(cfg, "cors_origins", "") or "")
    cors_origins = [o.strip() for o in cors_raw.split(",") if o.strip()]
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=False,  # 头部认证（X-API-Key），非 cookie
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _svc(name: str):
        return cortex_app.services.get(name) if cortex_app else None

    # ---- 鉴权 ------------------------------------------------------- #
    def require_key(request: Request) -> None:
        expected = getattr(cfg, "api_key", "") if cfg else ""
        if not expected:
            return
        got = request.headers.get("X-API-Key", "")
        # 常数时间比较，防时序侧信道逐字节探测密钥
        if not hmac.compare_digest(got.encode("utf-8"), expected.encode("utf-8")):
            raise HTTPException(status_code=401, detail="无效 X-API-Key")

    # ---- POST /research（SSE） -------------------------------------- #
    @app.post("/research")
    async def research(req: ResearchRequest, _: None = Depends(require_key)):
        app_ref = cortex_app
        if app_ref is None:
            raise HTTPException(status_code=503, detail="应用尚未初始化")
        # 409 检查与置位必须在 handler 内同步完成（async handler 无 await 的同步段
        # 相对其它请求是原子的），消除旧实现“handler 只查、worker 线程才置位”留下的
        # 并发窗口（两个请求都能通过检查）；worker 只负责 finally 清位。
        if _research_running.is_set():
            raise HTTPException(status_code=409, detail="已有研究任务在执行中，请等待完成后再提交")
        _research_running.set()

        events: queue.Queue = queue.Queue()

        def _on_progress(payload: dict) -> None:
            events.put({"progress": payload})

        def worker() -> None:
            _agent_gate.acquire()  # 研究全程持锁：与 chat 互斥（409 闸已在 handler 置位）
            try:
                if req.mode == "explore":
                    from ..agents.dispatch import ExplorationMode

                    result = ExplorationMode(app_ref.agents.orchestrator,
                                             progress_cb=_on_progress).run(req.topic)
                else:
                    pipe = app_ref.build_pipeline(progress_cb=_on_progress)
                    result = pipe.run(req.topic)
                report_path = _save_report(app_ref.cfg, req.topic, result.report_md)
                events.put({"done": {
                    "report_md": result.report_md,
                    "revised": result.revised,
                    "duration_seconds": round(result.duration_seconds, 1),
                    "mode": result.mode,
                    "topic": req.topic,
                    "report_path": report_path,
                }})
            except Exception as exc:  # noqa: BLE001 —— 异常经 SSE error 事件送达
                events.put({"error": {"message": _humanize_error(exc), "detail": str(exc)}})
            finally:
                _research_running.clear()
                _agent_gate.release()

        threading.Thread(target=worker, daemon=True).start()

        async def stream():
            while True:
                item = await asyncio.to_thread(events.get)
                if "error" in item:
                    yield {"event": "error",
                           "data": json.dumps(item["error"], ensure_ascii=False)}
                    break
                if "done" in item:
                    yield {"event": "done",
                           "data": json.dumps(item["done"], ensure_ascii=False)}
                    break
                yield {"event": "progress",
                       "data": json.dumps(item["progress"], ensure_ascii=False)}

        return EventSourceResponse(stream())

    # ---- POST /chat/stream（SSE 对话） ------------------------------- #
    @app.post("/chat/stream")
    async def chat_stream(req: ChatRequest, _: None = Depends(require_key)):
        app_ref = cortex_app
        if app_ref is None:
            raise HTTPException(status_code=503, detail="应用尚未初始化")

        sid = req.session_id or uuid.uuid4().hex[:12]
        info = sessions.get(sid)
        if info is None:
            info = SessionInfo(session_id=sid,
                               created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                               turns=0, history=[])
            sessions.put(sid, info)

        events: queue.Queue = queue.Queue()

        def worker() -> None:
            agents = getattr(app_ref, "agents", None)
            orch = getattr(agents, "orchestrator", None) if agents else None
            if orch is None:
                events.put({"error": {"message": "编排器未初始化", "detail": "orchestrator missing"}})
                return
            # 编排器一次只服务一个调用方：研究进行中（或另一轮对话未结束）快速失败，
            # 不把本轮对话静默排队到长达数十分钟的研究之后。
            if not _agent_gate.acquire(blocking=False):
                events.put({"error": {"message": "研究/探索任务进行中，请稍后再发消息",
                                      "detail": "orchestrator busy"}})
                return
            try:
                # 读己之写：上一轮经异步队列入记忆的问答先落盘，本轮 recall 才可见
                mem0 = _svc("memory")
                if mem0 is not None and hasattr(mem0, "flush_writes"):
                    try:
                        mem0.flush_writes(timeout=2)
                    except Exception:  # noqa: BLE001 —— flush 失败不阻断对话
                        pass
                if req.reset:
                    hm = getattr(orch, "history_manager", None)
                    if hm is not None and hasattr(hm, "clear"):
                        hm.clear()
                    # 会话级同步清空：防止迟到的 session 事件复活旧 sid 后
                    # 把新旧两段转录粘进同一份 info.history
                    info.history.clear()
                    info.turns = 0
                t0 = time.time()
                # 多轮上下文：PlanSolveAgent 不消费 history_manager（框架设计如此），
                # 历史以转录前缀进入任务文本；reset 或首轮则不加。
                # 注意：框架 run() 自身会把本轮问答追加进 history_manager，这里不再手写。
                prefix = ""
                if not req.reset and info.history:
                    lines = [
                        f"{'用户' if h['role'] == 'user' else '助手'}: {h['content'][:400]}"
                        for h in info.history[-6:]
                    ]
                    prefix = "【对话历史】\n" + "\n".join(lines) + "\n【当前消息】\n"
                answer = str(orch.run(prefix + req.message))
                duration = round(time.time() - t0, 1)
                # 记忆写穿（默认异步队列，廉价；失败不阻断对话）
                mem = _svc("memory")
                if mem is not None and hasattr(mem, "remember"):
                    try:
                        mem.remember(content=f"Q: {req.message}\nA: {answer}",
                                     type="episodic", source="web-chat",
                                     metadata={"session_id": sid})
                    except Exception:  # noqa: BLE001
                        pass
                info.turns += 1
                info.history.extend([
                    {"role": "user", "content": req.message},
                    {"role": "assistant", "content": answer},
                ])
                events.put({"message": {"role": "assistant", "content": answer,
                                        "duration_seconds": duration}})
                events.put({"done": {"session_id": sid, "turns": info.turns}})
            except Exception as exc:  # noqa: BLE001
                events.put({"error": {"message": _humanize_error(exc), "detail": str(exc)}})
            finally:
                _agent_gate.release()

        threading.Thread(target=worker, daemon=True).start()

        async def stream():
            # 先告知会话 id：前端可立即渲染用户消息并持久化 sid
            yield {"event": "session",
                   "data": json.dumps({"session_id": sid}, ensure_ascii=False)}
            while True:
                item = await asyncio.to_thread(events.get)
                if "error" in item:
                    yield {"event": "error",
                           "data": json.dumps(item["error"], ensure_ascii=False)}
                    break
                if "message" in item:
                    yield {"event": "message",
                           "data": json.dumps(item["message"], ensure_ascii=False)}
                elif "done" in item:
                    yield {"event": "done",
                           "data": json.dumps(item["done"], ensure_ascii=False)}
                    break

        return EventSourceResponse(stream())

    # ---- POST /ingest ----------------------------------------------- #
    @app.post("/ingest")
    async def ingest(file: UploadFile = File(...), _: None = Depends(require_key)):
        kb = _svc("knowledge")
        if kb is None:
            raise HTTPException(status_code=503, detail="知识库服务未初始化")
        upload_dir = Path("knowledge_base/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        # 上传大小上限：先读入内存再校验（个人助手定位下文件量级小，可接受）
        max_mb = int(os.environ.get("CORTEX_MAX_UPLOAD_MB", "50") or "50")
        payload = await file.read()
        if len(payload) > max_mb * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"文件过大：超过 {max_mb}MB 上限（可用 CORTEX_MAX_UPLOAD_MB 调整）")
        dest = upload_dir / f"{uuid.uuid4().hex[:8]}-{file.filename}"
        dest.write_bytes(payload)
        try:
            info = kb.ingest_document(dest, weak_llm=cortex_app.weak_llm)
        except Exception as exc:  # noqa: BLE001
            # 422 不回显原始异常串：可能携带本地路径等敏感信息
            raise HTTPException(
                status_code=422, detail=f"摄取失败: {_humanize_error(exc)}") from exc
        return {"doc_id": info["doc_id"], "title": info["title"], "n_chunks": info["n_chunks"]}

    @app.post("/ingest-text")
    async def ingest_text(req: IngestTextRequest, _: None = Depends(require_key)):
        kb = _svc("knowledge")
        if kb is None:
            raise HTTPException(status_code=503, detail="知识库服务未初始化")
        info = kb.ingest_text(title=req.title, text=req.content, weak_llm=cortex_app.weak_llm)
        return {"doc_id": info["doc_id"], "n_chunks": info["n_chunks"]}

    # ---- 知识库文档管理 ---------------------------------------------- #
    @app.get("/documents", response_model=DocumentList)
    async def list_documents(_: None = Depends(require_key)):
        kb = _svc("knowledge")
        if kb is None or not hasattr(kb, "list_documents"):
            raise HTTPException(status_code=503, detail="知识库服务未初始化")
        raw = kb.list_documents()
        docs = [
            # 显式归一：库里字段可能为 NULL，pydantic 严格类型不收 None
            DocumentOut(
                doc_id=str(d.get("doc_id") or ""),
                title=str(d.get("title") or ""),
                source=str(d.get("source") or ""),
                n_chunks=int(d.get("n_chunks") or 0),
                created_at=str(d.get("created_at") or ""),
            )
            for d in raw
        ]
        return DocumentList(documents=docs, total=len(docs))

    @app.delete("/documents/{doc_id}")
    async def delete_document(doc_id: str, _: None = Depends(require_key)):
        kb = _svc("knowledge")
        if kb is None:
            raise HTTPException(status_code=503, detail="知识库服务未初始化")
        deleted = kb.delete(doc_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="文档不存在")
        return {"deleted": True, "doc_id": doc_id}

    # ---- 研究报告取回 ------------------------------------------------ #
    def _reports_dir() -> Path:
        data_dir = getattr(cfg, "data_dir", "") if cfg else ""
        return Path(data_dir) / "reports" if data_dir else Path("data/reports")

    @app.get("/reports")
    async def list_reports(_: None = Depends(require_key)):
        reports_dir = _reports_dir()
        if not reports_dir.exists():
            return {"reports": [], "total": 0}
        items = [
            {
                "filename": p.name,
                "size_bytes": p.stat().st_size,
                "modified_at": time.strftime("%Y-%m-%dT%H:%M:%S",
                                             time.localtime(p.stat().st_mtime)),
            }
            for p in sorted(reports_dir.glob("*.md"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        ]
        return {"reports": items, "total": len(items)}

    @app.get("/reports/{filename}")
    async def get_report(filename: str, _: None = Depends(require_key)):
        reports_dir = _reports_dir().resolve()
        candidate = (reports_dir / filename).resolve()
        # 防目录穿越：解析后必须仍位于 reports 目录内的真实文件
        if not candidate.is_relative_to(reports_dir) or not candidate.is_file():
            raise HTTPException(status_code=404, detail="报告不存在")
        return FileResponse(candidate, media_type="text/markdown", filename=candidate.name)

    # ---- 会话 ------------------------------------------------------- #
    @app.post("/session", response_model=SessionInfo)
    async def create_session(req: SessionCreateRequest, _: None = Depends(require_key)):
        sid = uuid.uuid4().hex[:12]
        info = SessionInfo(session_id=sid, created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                           turns=len(req.history), history=req.history)
        sessions.put(sid, info)
        return info

    @app.get("/session/{session_id}", response_model=SessionInfo)
    async def get_session(session_id: str, _: None = Depends(require_key)):
        info = sessions.get(session_id)
        if info is None:
            if sessions.evicted_recently(session_id):
                raise HTTPException(status_code=410, detail="会话已过期或被清理，请开启新对话")
            raise HTTPException(status_code=404, detail="会话不存在")
        return info

    # ---- 健康检查 ---------------------------------------------------- #
    @app.get("/health")
    async def health():
        payload = {
            "status": "ok",
            "version": app.version,
            "uptime_seconds": round(time.time() - started_at, 1),
            "auth_required": bool(getattr(cfg, "api_key", "") if cfg else False),
            "research_running": _research_running.is_set(),
            "services": {
                name: (_svc(name) is not None)
                for name in ("knowledge", "memory", "context", "mcp", "a2a")
            },
        }
        kb = _svc("knowledge")
        if kb is not None and hasattr(kb, "stats"):
            try:
                payload["knowledge"] = kb.stats()
            except Exception:  # noqa: BLE001
                pass
        return payload

    # ---- SPA 托管（最后注册，避免遮蔽 API 路由） ---------------------- #
    dist_override = os.environ.get("CORTEX_DIST_DIR")
    dist_dir = Path(dist_override) if dist_override \
        else Path(__file__).resolve().parents[2] / "frontend" / "dist"
    _mount_spa(app, dist_dir)

    return app


def main() -> None:
    import uvicorn

    host = os.environ.get("CORTEX_API_HOST", "127.0.0.1")
    port = int(os.environ.get("CORTEX_API_PORT", "8000"))
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
