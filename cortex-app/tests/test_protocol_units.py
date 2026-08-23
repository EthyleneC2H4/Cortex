"""P6 单元测试：MCP 客户端管理器 / MCP 桥工具 / A2A 客户端与服务器 / A2A 工具壳。

零网络零 docker：
- MCP 网络路径通过 monkeypatch ``manager._open_session`` 用假会话离线跑；
- ``mcp`` 包未安装（HAS_MCP=False）路径显式 monkeypatch 模块常量验证；
- A2A 客户端用 httpx.MockTransport；A2A 服务器用 fastapi TestClient（ASGI 内存）。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import types

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import cortex.services.protocol.mcp_client as mcp_client_module
from cortex.services.protocol.a2a import A2AClient, A2AServer
from cortex.services.protocol.mcp_client import MCPClientManager
from cortex.tools.a2a_tool import A2ATool
from cortex.tools.mcp_tool import MCPBridgeTool

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------- #
# 断言辅助                                                                 #
# ---------------------------------------------------------------------- #
def _status(resp):
    """ToolResponse.status 是枚举，用 .value 比较。"""
    return resp.status.value if hasattr(resp.status, "value") else resp.status


def ok(resp):
    assert _status(resp) == "success", f"期望成功，实际 {resp.status}: {resp.error_info}"
    return resp


def err_code(resp, code: str):
    """错误码用字符串比较（ToolErrorCode 常量即字符串）。"""
    info = resp.error_info
    got = info.get("code") if isinstance(info, dict) else getattr(info, "code", None)
    assert _status(resp) == "error", f"期望 {code}，实际 success: {resp.text}"
    assert got == code, f"期望 {code}，实际 {got}: {info}"
    return resp


# ---------------------------------------------------------------------- #
# MCPClientManager：配置形状校验                                            #
# ---------------------------------------------------------------------- #
def _stdio_cfg(**over):
    cfg = {"name": "web-search", "transport": "stdio", "command": "npx", "args": ["-y", "srv"]}
    cfg.update(over)
    return cfg


class TestManagerValidation:
    def test_missing_name(self):
        with pytest.raises(ValueError, match="name"):
            MCPClientManager([{"transport": "stdio", "command": "npx"}])

    @pytest.mark.parametrize(
        "cfg",
        [
            {"name": "a"},  # 缺 transport
            {"name": "a", "transport": "carrier-pigeon", "command": "npx"},  # 不支持的 transport
        ],
    )
    def test_bad_transport(self, cfg):
        with pytest.raises(ValueError, match="transport"):
            MCPClientManager([cfg])

    def test_stdio_requires_command(self):
        with pytest.raises(ValueError, match="command"):
            MCPClientManager([{"name": "a", "transport": "stdio"}])

    def test_http_requires_url(self):
        with pytest.raises(ValueError, match="url"):
            MCPClientManager([{"name": "b", "transport": "http"}])

    def test_duplicate_names(self):
        with pytest.raises(ValueError, match="重复"):
            MCPClientManager([_stdio_cfg(), _stdio_cfg()])

    def test_non_list_servers(self):
        with pytest.raises(ValueError):
            MCPClientManager({"name": "a", "transport": "stdio"})  # type: ignore[arg-type]

    def test_empty_config_constructs_fine(self):
        assert MCPClientManager([]).servers == []
        assert MCPClientManager().servers == []


# ---------------------------------------------------------------------- #
# MCPClientManager：空配置行为                                              #
# ---------------------------------------------------------------------- #
class TestManagerEmpty:
    def test_list_tools_empty(self):
        assert MCPClientManager([]).list_tools() == []

    def test_call_tool_raises_value_error(self):
        with pytest.raises(ValueError, match="未配置 MCP 服务器"):
            MCPClientManager([]).call_tool("any", "tool", {})

    def test_unknown_server_raises_value_error(self):
        mgr = MCPClientManager([_stdio_cfg()])
        with pytest.raises(ValueError, match="未知的 MCP 服务器"):
            mgr.call_tool("nope", "tool", {})


# ---------------------------------------------------------------------- #
# MCPClientManager：离线假会话（monkeypatch _open_session）                  #
# ---------------------------------------------------------------------- #
class _FakeSession:
    """固定返回结构的假 MCP 会话。"""

    async def list_tools(self):
        tool = types.SimpleNamespace(name="web_search", description="搜索网页")
        return types.SimpleNamespace(tools=[tool])

    async def call_tool(self, name, arguments):
        block = types.SimpleNamespace(text=f"结果:{name}:{sorted(arguments.items())}")
        return types.SimpleNamespace(content=[block], isError=False)


@contextlib.asynccontextmanager
async def _fake_open(server):  # 作为实例属性补丁时只接收 server
    yield _FakeSession()


def _patched_manager(monkeypatch, session=_FakeSession):
    """带一个合法 stdio 配置、_open_session 已替换为假会话的管理器。

    同时强制 HAS_MCP=True：会话已被假货顶替，不会真正 import mcp，
    这样在未安装 mcp 包的环境里也能走通正常路径。
    """

    @contextlib.asynccontextmanager
    async def _open(server):
        yield session()

    mgr = MCPClientManager([_stdio_cfg()])
    monkeypatch.setattr(mgr, "_open_session", _open)
    monkeypatch.setattr(mcp_client_module, "HAS_MCP", True)
    return mgr


class TestManagerOffline:
    def test_list_tools_happy_path(self, monkeypatch):
        mgr = _patched_manager(monkeypatch)
        tools = mgr.list_tools()
        assert tools == [{"server": "web-search", "name": "web_search", "description": "搜索网页"}]

    def test_call_tool_happy_path(self, monkeypatch):
        mgr = _patched_manager(monkeypatch)
        out = mgr.call_tool("web-search", "web_search", {"q": "RAG"}, timeout=10)
        assert out.startswith("结果:web_search:")
        assert "'q', 'RAG'" in out

    def test_timeout_maps_to_timeout_error(self, monkeypatch):
        class Boom(_FakeSession):
            async def call_tool(self, name, arguments):
                raise TimeoutError("模拟超时")

        mgr = _patched_manager(monkeypatch, session=Boom)
        with pytest.raises(TimeoutError, match="超时"):
            mgr.call_tool("web-search", "web_search", {})

    def test_oserror_maps_to_connection_error(self, monkeypatch):
        class Boom(_FakeSession):
            async def call_tool(self, name, arguments):
                raise OSError("连接被拒")

        mgr = _patched_manager(monkeypatch, session=Boom)
        with pytest.raises(ConnectionError, match="连接失败"):
            mgr.call_tool("web-search", "web_search", {})

    def test_is_error_result_maps_to_connection_error(self, monkeypatch):
        class ErrSession(_FakeSession):
            async def call_tool(self, name, arguments):
                block = types.SimpleNamespace(text="远端炸了")
                return types.SimpleNamespace(content=[block], isError=True)

        mgr = _patched_manager(monkeypatch, session=ErrSession)
        with pytest.raises(ConnectionError, match="远端报错|远端炸了"):
            mgr.call_tool("web-search", "web_search", {})

    def test_has_mcp_false_call_tool_runtime_error(self, monkeypatch):
        mgr = MCPClientManager([_stdio_cfg()])
        monkeypatch.setattr(mcp_client_module, "HAS_MCP", False)
        with pytest.raises(RuntimeError, match=r"pip install"):
            mgr.call_tool("web-search", "web_search", {})

    def test_has_mcp_false_list_tools_runtime_error(self, monkeypatch):
        mgr = MCPClientManager([_stdio_cfg()])
        monkeypatch.setattr(mcp_client_module, "HAS_MCP", False)
        with pytest.raises(RuntimeError, match=r"pip install"):
            mgr.list_tools()

    def test_run_coro_and_close_idempotent(self):
        """后台 loop 线程懒启动可用；close 幂等且可重复调用。"""
        mgr = MCPClientManager([_stdio_cfg()])

        async def answer():
            await asyncio.sleep(0)
            return 42

        assert mgr.run_coro(mgr.servers[0], answer(), timeout=5) == 42
        mgr.close()
        mgr.close()  # 幂等


# ---------------------------------------------------------------------- #
# MCPBridgeTool                                                            #
# ---------------------------------------------------------------------- #
class FakeManager:
    """工具层测试用的假管理器。"""

    def __init__(self, servers=None):
        self.servers = servers if servers is not None else [_stdio_cfg()]

    def list_tools(self):
        if not self.servers:
            return []
        return [{"server": "web-search", "name": "web_search", "description": "搜索网页"}]

    def call_tool(self, server_name, tool_name, arguments, timeout=60):
        if not self.servers:
            raise ValueError("未配置 MCP 服务器")
        return f"{server_name}.{tool_name} 的结果文本"


class BoomManager(FakeManager):
    def call_tool(self, server_name, tool_name, arguments, timeout=60):
        raise TimeoutError("慢")


class NetFailManager(FakeManager):
    def call_tool(self, server_name, tool_name, arguments, timeout=60):
        raise ConnectionError("连不上")


class TestMCPBridgeTool:
    def test_expanded_names_exact_set(self):
        tool = MCPBridgeTool(manager=FakeManager())
        assert {t.name for t in tool.get_expanded_tools()} == {
            "mcp_list_tools",
            "mcp_call_tool",
        }

    def test_bad_json_arguments_invalid_param(self):
        tool = MCPBridgeTool(manager=FakeManager())
        err_code(tool.call_tool(server="web-search", name="t", arguments="{不是json"), "INVALID_PARAM")

    def test_arguments_not_object_invalid_param(self):
        tool = MCPBridgeTool(manager=FakeManager())
        err_code(tool.call_tool(server="s", name="t", arguments='["数组"]'), "INVALID_PARAM")

    def test_empty_manager_call_invalid_param(self):
        tool = MCPBridgeTool(manager=MCPClientManager([]))
        err_code(tool.call_tool(server="x", name="t"), "INVALID_PARAM")

    def test_unknown_server_invalid_param(self):
        tool = MCPBridgeTool(manager=MCPClientManager([_stdio_cfg()]))
        err_code(tool.call_tool(server="ghost", name="t"), "INVALID_PARAM")

    def test_timeout_and_network_mapping(self):
        err_code(MCPBridgeTool(BoomManager()).call_tool(server="s", name="t"), "TIMEOUT")
        err_code(MCPBridgeTool(NetFailManager()).call_tool(server="s", name="t"), "NETWORK_ERROR")

    def test_call_success(self):
        tool = MCPBridgeTool(manager=FakeManager())
        resp = ok(tool.call_tool(server="web-search", name="web_search", arguments='{"q":"RAG"}'))
        assert "结果文本" in resp.text
        assert resp.data["server"] == "web-search"
        assert resp.data["name"] == "web_search"
        assert "raw" in resp.data

    def test_list_tools_success_format(self):
        tool = MCPBridgeTool(manager=FakeManager())
        resp = ok(tool.list_tools())
        assert "- [web-search] web_search: 搜索网页" in resp.text
        assert len(resp.data["tools"]) == 1

    def test_list_tools_empty_config_success_text(self):
        resp = ok(MCPBridgeTool(manager=MCPClientManager([])).list_tools())
        assert resp.text == "未配置 MCP 服务器"


# ---------------------------------------------------------------------- #
# A2AClient                                                                #
# ---------------------------------------------------------------------- #
class TestA2AClient:
    def _client(self, handler) -> A2AClient:
        return A2AClient("http://researcher.test/", transport=httpx.MockTransport(handler))

    def test_delegate_ok_structure(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["path"] = request.url.path
            captured["body"] = json.loads(request.content.decode())
            return httpx.Response(
                200,
                json={"task_id": "abc123", "status": "completed", "artifact": "# 报告"},
            )

        out = self._client(handler).delegate("调研 RAG")
        assert captured["path"] == "/a2a/message"
        assert captured["body"] == {"task": "调研 RAG"}
        assert out == {"task_id": "abc123", "status": "completed", "artifact": "# 报告"}

    def test_delegate_failure_passthrough_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"task_id": "e1", "status": "failed", "artifact": "", "error": "炸了"},
            )

        out = self._client(handler).delegate("x")
        assert out["status"] == "failed"
        assert out["error"] == "炸了"

    def test_delegate_http_500_runtime_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        with pytest.raises(RuntimeError, match="500"):
            self._client(handler).delegate("x")

    def test_discover_route_not_implemented(self):
        client = A2AClient("http://researcher.test")
        for fn in (client.discover, client.route):
            with pytest.raises(NotImplementedError, match="ANP"):
                fn()


# ---------------------------------------------------------------------- #
# A2AServer                                                                #
# ---------------------------------------------------------------------- #
def _test_app(runner) -> TestClient:
    app = FastAPI()
    app.include_router(A2AServer(runner).router())
    return TestClient(app)


class TestA2AServer:
    def test_completed_roundtrip(self):
        tc = _test_app(lambda task: f"done:{task}")
        r = tc.post("/a2a/message", json={"task": "调研 RAG"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "completed"
        assert body["artifact"] == "done:调研 RAG"
        assert isinstance(body["task_id"], str) and body["task_id"]

    def test_runner_exception_returns_failed_with_200(self):
        def boom(task):
            raise ValueError("流水线内部爆炸")

        tc = _test_app(boom)
        r = tc.post("/a2a/message", json={"task": "x"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "failed"
        assert body["artifact"] == ""
        assert "流水线内部爆炸" in body.get("error", "")

    @pytest.mark.parametrize("payload", [{}, {"task": ""}, {"other": "x"}])
    def test_missing_or_empty_task_is_422(self, payload):
        tc = _test_app(lambda t: t)
        assert tc.post("/a2a/message", json=payload).status_code == 422


# ---------------------------------------------------------------------- #
# A2ATool                                                                  #
# ---------------------------------------------------------------------- #
class FakeA2AClient:
    def __init__(self, result=None, exc: Exception | None = None):
        self.result = result or {"task_id": "t1", "status": "completed", "artifact": "长" * 800}
        self.exc = exc

    def delegate(self, task):
        if self.exc is not None:
            raise self.exc
        return self.result


class TestA2ATool:
    def test_expanded_names_exact_set(self):
        assert {t.name for t in A2ATool(None).get_expanded_tools()} == {"a2a_delegate"}

    def test_none_client_invalid_param_mentions_env(self):
        resp = A2ATool(None).delegate(task="调研")
        err = err_code(resp, "INVALID_PARAM")
        assert "CORTEX_A2A_RESEARCHER_URL" in err.text

    def test_success_truncated_text_full_data(self):
        tool = A2ATool(client=FakeA2AClient())
        resp = ok(tool.delegate(task="任务"))
        assert resp.data["task_id"] == "t1"
        assert resp.data["artifact"] == "长" * 800  # data 保留完整响应
        assert len(resp.text) <= 501  # text 截 500 字预览
        assert resp.text == "长" * 500

    def test_runtime_error_maps_to_network_error(self):
        tool = A2ATool(client=FakeA2AClient(exc=RuntimeError("HTTP 503")))
        err_code(tool.delegate(task="任务"), "NETWORK_ERROR")
