"""MCPClientManager — MCP（Model Context Protocol）客户端管理器（P6）。

职责：
- 持有若干 MCP 服务器配置并做形状校验（构造时不连接）；
- 每个 server 一个后台线程 + 独立 asyncio loop（懒启动），同步桥
  :meth:`run_coro` 把协程投递到对应 loop 并阻塞取结果；
- :meth:`list_tools` / :meth:`call_tool` 建会话列工具 / 调工具，
  异常统一归一为 ``TimeoutError`` / ``ConnectionError``，供上层映射错误码；
- ``mcp`` 包为可选依赖：未安装时网络操作抛带安装指引的 RuntimeError。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import threading
from typing import Any, Dict, Iterator, List

try:  # mcp 为可选依赖（pyproject [mcp] extra）
    import mcp  # noqa: F401

    HAS_MCP = True
except ImportError:  # pragma: no cover - 环境相关
    HAS_MCP = False

_SUPPORTED_TRANSPORTS = ("stdio", "sse", "http")
_MCP_NOT_INSTALLED = "mcp 未安装：pip install -e '.[mcp]'"


def _require_mcp() -> None:
    """未安装 mcp 包时抛可操作的报错（读模块级 HAS_MCP，便于测试 monkeypatch）。"""
    if not HAS_MCP:
        raise RuntimeError(_MCP_NOT_INSTALLED)


def _loop_worker(loop: asyncio.AbstractEventLoop) -> None:
    """线程主体：在该线程里常驻跑一个事件循环。"""
    asyncio.set_event_loop(loop)
    loop.run_forever()


class _LoopHandle:
    """一个 server 对应的后台 loop 句柄（懒启动、幂等关闭）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: threading.Thread | None = None

    def ensure(self) -> asyncio.AbstractEventLoop:
        """返回存活的事件循环；没有或已停则新起线程。"""
        with self._lock:
            if self.loop is not None and self.loop.is_running():
                return self.loop
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=_loop_worker, args=(loop,), name="cortex-mcp-loop", daemon=True
            )
            thread.start()
            self.loop, self.thread = loop, thread
            return loop

    def stop(self) -> None:
        """停掉 loop 线程；幂等（重复调用安全）。"""
        with self._lock:
            loop, thread = self.loop, self.thread
            self.loop = self.thread = None
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5)


class MCPClientManager:
    """MCP 服务器连接管理器：配置校验 + 每 server 独立事件循环 + 会话桥接。

    server 配置形如::

        {"name": "web-search", "transport": "stdio",
         "command": "npx", "args": ["-y", "server"], "env": {...}}
        {"name": "remote", "transport": "http" /* 或 "sse" */, "url": "https://..."}

    构造时只校验配置形状，不做任何网络/进程操作；真正的会话在
    :meth:`list_tools` / :meth:`call_tool` 首次使用时建立。
    """

    def __init__(self, servers: List[Dict[str, Any]] | None = None) -> None:
        if servers is None:
            servers = []
        if not isinstance(servers, list):
            raise ValueError("MCP 服务器配置必须是 JSON 数组（list[dict]）")
        seen: set[str] = set()
        for raw in servers:
            self._validate_server(raw)
            if raw["name"] in seen:
                raise ValueError(f"MCP 服务器名重复: '{raw['name']}'")
            seen.add(raw["name"])
        # 校验通过后浅拷贝固化，避免外部后续改动影响运行时
        self.servers: List[Dict[str, Any]] = [dict(s) for s in servers]
        self._handles: Dict[str, _LoopHandle] = {}

    # ------------------------------------------------------------------ #
    # 配置校验                                                             #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_server(server: Any) -> None:
        """单条 server 配置的形状校验；不合法抛 ValueError（可读中文信息）。"""
        if not isinstance(server, dict):
            raise ValueError(f"MCP 服务器配置必须是对象(dict)，得到 {type(server).__name__}")
        name = server.get("name")
        if not name or not isinstance(name, str):
            raise ValueError(f"MCP 服务器配置缺少非空 'name' 字段: {server!r}")
        transport = server.get("transport")
        if not transport or not isinstance(transport, str):
            raise ValueError(f"MCP 服务器 '{name}' 缺少 'transport' 字段（可选值: stdio/http/sse）")
        if transport not in _SUPPORTED_TRANSPORTS:
            raise ValueError(
                f"MCP 服务器 '{name}' 的 transport '{transport}' 不支持，"
                f"可选值: {'/'.join(_SUPPORTED_TRANSPORTS)}"
            )
        command = server.get("command")
        if transport == "stdio" and (not isinstance(command, str) or not command.strip()):
            raise ValueError(f"MCP 服务器 '{name}'（stdio）缺少非空 'command' 字段")
        url = server.get("url")
        if transport in ("http", "sse") and (not isinstance(url, str) or not url.strip()):
            raise ValueError(f"MCP 服务器 '{name}'（{transport}）缺少非空 'url' 字段")

    def _handle(self, server: Dict[str, Any]) -> _LoopHandle:
        """按 server 名取（必要时新建）后台 loop 句柄。"""
        handle = self._handles.get(server["name"])
        if handle is None:
            handle = _LoopHandle()
            self._handles[server["name"]] = handle
        return handle

    # ------------------------------------------------------------------ #
    # 同步桥                                                               #
    # ------------------------------------------------------------------ #
    def _ensure_loop(self, server: Dict[str, Any]) -> asyncio.AbstractEventLoop:
        """确保该 server 的后台 loop 线程已启动并返回它（懒启动）。"""
        return self._handle(server).ensure()

    def run_coro(self, server: Dict[str, Any], coro: Any, timeout: float = 60) -> Any:
        """把协程投递到该 server 的后台 loop，同步等待结果。

        - 等待超时 → ``TimeoutError``（并尝试取消远端任务）；
        - 连接类失败（ConnectionError/OSError/MCP 错误等其余异常）
          统一包装为 ``ConnectionError``，上层据此映射 NETWORK_ERROR。
        """
        loop = self._ensure_loop(server)
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            return future.result(timeout)
        except (asyncio.TimeoutError, concurrent.futures.TimeoutError):
            future.cancel()
            raise TimeoutError(
                f"MCP 操作超时（{timeout}s）：server={server['name']}"
            ) from None
        except (ConnectionError, OSError) as exc:
            raise ConnectionError(f"MCP 连接失败（server={server['name']}）: {exc}") from exc
        except BaseException as exc:  # MCP SDK 错误/其它异常 → 归一为 ConnectionError
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise ConnectionError(
                f"MCP 调用出错（server={server['name']}）: {exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    # 会话建立（测试可 monkeypatch 本方法离线）                             #
    # ------------------------------------------------------------------ #
    def _open_session(self, server: Dict[str, Any]):
        """按 transport 打开一个 async 上下文，进入即 yield 已初始化的 ClientSession。"""
        transport = server["transport"]
        if transport == "stdio":
            return self._open_stdio_session(server)
        if transport == "sse":
            return self._open_sse_session(server)
        if transport == "http":
            return self._open_http_session(server)
        raise ValueError(  # pragma: no cover - 构造期已拦截
            f"MCP 服务器 '{server['name']}' 的 transport '{transport}' 不支持"
        )

    @staticmethod
    @contextlib.asynccontextmanager
    async def _open_stdio_session(server: Dict[str, Any]) -> Iterator[Any]:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=server["command"],
            args=list(server.get("args") or []),
            env=dict(server.get("env") or {}) or None,
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    @staticmethod
    @contextlib.asynccontextmanager
    async def _open_sse_session(server: Dict[str, Any]) -> Iterator[Any]:
        from mcp import ClientSession
        from mcp.client.sse import sse_client

        async with sse_client(url=server["url"]) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    @staticmethod
    @contextlib.asynccontextmanager
    async def _open_http_session(server: Dict[str, Any]) -> Iterator[Any]:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        async with streamablehttp_client(server["url"]) as (read, write, *_rest):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    # ------------------------------------------------------------------ #
    # 对外能力                                                             #
    # ------------------------------------------------------------------ #
    def list_tools(self) -> List[Dict[str, str]]:
        """遍历所有 server 列出远端工具：[{"server","name","description"}, ...]。"""
        if not self.servers:
            return []
        _require_mcp()
        out: List[Dict[str, str]] = []
        for server in self.servers:

            async def _list_one(srv: Dict[str, Any]) -> List[Dict[str, str]]:
                async with self._open_session(srv) as session:
                    result = await session.list_tools()
                    return [
                        {
                            "server": srv["name"],
                            "name": getattr(t, "name", ""),
                            "description": getattr(t, "description", "") or "",
                        }
                        for t in getattr(result, "tools", []) or []
                    ]

            out.extend(self.run_coro(server, _list_one(server), timeout=60))
        return out

    def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any], timeout: float = 60) -> str:
        """调用指定 server 的指定工具，返回拼接后的文本结果。

        未配置任何 server 时抛 ``ValueError("未配置 MCP 服务器")``；
        server 名不存在同样抛 ValueError。
        """
        if not self.servers:
            raise ValueError("未配置 MCP 服务器")
        server = next((s for s in self.servers if s.get("name") == server_name), None)
        if server is None:
            known = ", ".join(s["name"] for s in self.servers)
            raise ValueError(f"未知的 MCP 服务器 '{server_name}'（已配置: {known}）")
        _require_mcp()

        async def _call_one(srv: Dict[str, Any]) -> str:
            async with self._open_session(srv) as session:
                result = await session.call_tool(tool_name, dict(arguments or {}))
                text = _result_text(result)
                if getattr(result, "isError", False):
                    raise ConnectionError(f"远端工具报错: {text[:300]}")
                return text

        return self.run_coro(server, _call_one(server), timeout=timeout)

    def close(self) -> None:
        """停掉全部后台 loop 线程；幂等，可安全重复调用。"""
        handles, self._handles = list(self._handles.values()), {}
        for handle in handles:
            handle.stop()


def _result_text(result: Any) -> str:
    """拼接 MCP 结果的文本内容块（TextContent.text），无文本回退 repr。"""
    parts: List[str] = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        parts.append(text if isinstance(text, str) else repr(block))
    return "\n".join(p for p in parts if p)
