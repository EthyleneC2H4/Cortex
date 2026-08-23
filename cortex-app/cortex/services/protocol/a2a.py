"""A2A（Agent-to-Agent）极简桥接（P6）。

- :class:`A2AClient`：把研究任务 POST 给远端 researcher 服务的
  ``/a2a/message`` 端点，取回 ``{"task_id","status","artifact"}``；
- :class:`A2AServer`：把本机 researcher 流水线（任意 ``task -> str``
  回调）暴露为同形 HTTP 端点，供其它 agent 委派。

v1 只做"委派-等待-取结果"；ANP 发现/路由属展望能力，显式 NotImplementedError。
"""

from __future__ import annotations

import uuid
from typing import Any, Callable, Dict

from fastapi import APIRouter
from pydantic import BaseModel, Field


class _DelegateRequest(BaseModel):
    """POST /a2a/message 的请求体；task 缺失或空串由 pydantic 校验为 422。"""

    task: str = Field(min_length=1)


class A2AClient:
    """远端 researcher 的 A2A 客户端。

    Args:
        base_url: 远端服务基址，如 ``http://localhost:8900``
        timeout: 委派超时秒数（researcher 流水线可能较慢，默认 300）
        transport: 可注入 httpx transport（如 MockTransport），仅测试用
    """

    def __init__(self, base_url: str, timeout: float = 300, transport: Any = None) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.timeout = timeout
        self._transport = transport

    def delegate(self, task: str) -> Dict[str, Any]:
        """委派任务并同步等待结果。

        Returns:
            归一化后的 ``{"task_id", "status", "artifact"}``（失败响应额外带 error）

        Raises:
            RuntimeError: HTTP 状态非 2xx（信息含状态码）
        """
        import httpx

        with httpx.Client(timeout=self.timeout, transport=self._transport) as client:
            resp = client.post(f"{self.base_url}/a2a/message", json={"task": task})
        if not (200 <= resp.status_code < 300):
            raise RuntimeError(
                f"A2A 委派失败: HTTP {resp.status_code}: {resp.text[:200]}"
            )
        data = resp.json()
        out: Dict[str, Any] = {
            "task_id": data.get("task_id"),
            "status": data.get("status"),
            "artifact": data.get("artifact"),
        }
        if data.get("error"):
            out["error"] = data["error"]
        return out

    def discover(self) -> None:  # pragma: no cover - 展望能力
        """ANP 能力发现；v1 未实现。"""
        raise NotImplementedError("ANP 发现/路由属展望能力，v1 未实现")

    def route(self) -> None:  # pragma: no cover - 展望能力
        """ANP 路由决策；v1 未实现。"""
        raise NotImplementedError("ANP 发现/路由属展望能力，v1 未实现")


class A2AServer:
    """把本机 runner（researcher 流水线）暴露为 A2A HTTP 端点。

    Args:
        runner: 同步回调 ``task:str -> artifact:str``，在 FastAPI 线程池中执行；
                抛异常时端点仍返回 200，status 置为 failed。
    """

    def __init__(self, runner: Callable[[str], str]) -> None:
        self.runner = runner

    def router(self) -> APIRouter:
        """构造可挂载到任意 FastAPI app 的路由（POST /a2a/message）。"""
        router = APIRouter()
        server = self

        @router.post("/a2a/message")
        def delegate(req: _DelegateRequest) -> Dict[str, Any]:
            # sync def → Starlette 自动放入线程池执行，不阻塞事件循环
            try:
                artifact = server.runner(req.task)
            except Exception as exc:
                return {
                    "task_id": uuid.uuid4().hex,
                    "status": "failed",
                    "artifact": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            return {
                "task_id": uuid.uuid4().hex,
                "status": "completed",
                "artifact": artifact,
            }

        return router


def build_a2a_app(runner: Callable[[str], str]) -> Any:
    """便捷函数：runner → 挂好 /a2a/message 的 FastAPI 应用（uvicorn 直起）。"""
    from fastapi import FastAPI

    app = FastAPI(title="Cortex A2A Researcher")
    app.include_router(A2AServer(runner).router())
    return app
