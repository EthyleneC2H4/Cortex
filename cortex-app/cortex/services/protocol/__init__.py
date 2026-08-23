"""协议桥接层（P6）：MCP 客户端管理器 + A2A 客户端/服务器。

- :class:`~cortex.services.protocol.mcp_client.MCPClientManager`：
  管理若干 MCP 服务器连接（stdio / sse / http），列出远端工具并代为调用；
- :class:`~cortex.services.protocol.a2a.A2AClient`：把研究任务委派给远端 researcher；
- :class:`~cortex.services.protocol.a2a.A2AServer`：把本机 researcher 以 HTTP 端点暴露。
"""

from cortex.services.protocol.a2a import A2AClient, A2AServer
from cortex.services.protocol.mcp_client import MCPClientManager

__all__ = ["MCPClientManager", "A2AClient", "A2AServer"]
