"""MCPBridgeTool — MCP 工具桥（P6）。

把 :class:`~cortex.services.protocol.mcp_client.MCPClientManager` 包装为
expandable ``Tool``，注册后展开为两个子工具：

- ``mcp_list_tools``：列出全部 MCP 远端工具；
- ``mcp_call_tool``：按 server 名 + 工具名调用远端工具（arguments 为
  JSON 字符串，对 LLM 友好）；异常统一映射结构化错误码。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from hello_agents.tools import Tool, ToolResponse, tool_action
from hello_agents.tools.errors import ToolErrorCode


class MCPBridgeTool(Tool):
    """MCP 远端工具的列出与调用；所有失败返回结构化 ToolResponse。"""

    def __init__(self, manager) -> None:
        self.manager = manager  # MCPClientManager
        super().__init__(
            name="mcp",
            description="MCP 协议桥：列出并调用外部 MCP 服务器的工具",
            expandable=True,
        )

    # 方法名 → 子工具名：mcp_list_tools / mcp_call_tool
    @tool_action(description="列出已配置 MCP 服务器提供的全部远端工具")
    def list_tools(self) -> ToolResponse:
        """列出全部 MCP 远端工具。

        Args:
            无参数
        """
        try:
            tools = self.manager.list_tools()
        except RuntimeError as exc:  # mcp 未安装等环境问题
            return ToolResponse.error(ToolErrorCode.EXECUTION_ERROR, str(exc))
        except TimeoutError:
            return ToolResponse.error(ToolErrorCode.TIMEOUT, "列取 MCP 工具超时")
        except ConnectionError as exc:
            return ToolResponse.error(ToolErrorCode.NETWORK_ERROR, f"连接 MCP 服务器失败: {exc}")
        except Exception as exc:
            return ToolResponse.error(ToolErrorCode.EXECUTION_ERROR, f"列取 MCP 工具失败: {exc}")
        if not tools:
            return ToolResponse.success(text="未配置 MCP 服务器", data={"tools": []})
        lines = [f"- [{t['server']}] {t['name']}: {t['description']}" for t in tools]
        return ToolResponse.success(text="\n".join(lines), data={"tools": tools})

    @tool_action(description="调用指定 MCP 服务器的指定工具；arguments 为 JSON 对象字符串")
    def call_tool(self, server: str, name: str, arguments: str = "{}") -> ToolResponse:
        """调用远端 MCP 工具。

        Args:
            server: MCP 服务器名（mcp_list_tools 结果方括号内的名字）
            name: 远端工具名
            arguments: 工具入参，JSON 对象字符串，如 '{"query": "RAG"}'
        """
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            return ToolResponse.error(
                ToolErrorCode.INVALID_PARAM,
                f"arguments 不是合法 JSON: {exc}；输入为 {arguments[:120]!r}",
            )
        if not isinstance(args, dict):
            return ToolResponse.error(
                ToolErrorCode.INVALID_PARAM, f"arguments 必须是 JSON 对象，得到 {type(args).__name__}"
            )
        try:
            raw = self.manager.call_tool(server, name, args)
        except ValueError as exc:  # 未配置服务器 / 未知 server 名
            return ToolResponse.error(ToolErrorCode.INVALID_PARAM, str(exc))
        except TimeoutError:
            return ToolResponse.error(
                ToolErrorCode.TIMEOUT, f"MCP 工具调用超时: {server}.{name}"
            )
        except ConnectionError as exc:
            return ToolResponse.error(
                ToolErrorCode.NETWORK_ERROR, f"MCP 连接失败: {exc}"
            )
        except Exception as exc:
            return ToolResponse.error(
                ToolErrorCode.EXECUTION_ERROR, f"MCP 调用失败: {exc}"
            )
        return ToolResponse.success(
            text=raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False),
            data={"server": server, "name": name, "raw": raw},
        )

    # 抽象契约（展开路径不会直调父工具）
    def get_parameters(self) -> List[Any]:  # pragma: no cover
        from hello_agents.tools import ToolParameter

        return [ToolParameter(name="action", type="string", description="动作名")]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:  # pragma: no cover
        return ToolResponse.error(
            ToolErrorCode.INVALID_PARAM, "MCPBridgeTool 应通过 mcp_* 子工具调用"
        )
