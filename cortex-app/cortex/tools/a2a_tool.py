"""A2ATool — 跨智能体委派工具壳（P6）。

把 :class:`~cortex.services.protocol.a2a.A2AClient` 包装为 expandable
``Tool``，注册后展开为 ``a2a_delegate`` 一个子工具；未配置
``CORTEX_A2A_RESEARCHER_URL`` 时（client 为 None）返回结构化错误提示。
"""

from __future__ import annotations

from typing import Any, Dict, List

from hello_agents.tools import Tool, ToolResponse, tool_action
from hello_agents.tools.errors import ToolErrorCode

_ARTIFACT_PREVIEW = 500  # text 给 LLM 的预览长度；data 保留完整响应


class A2ATool(Tool):
    """A2A 委派：把研究任务交给远端 researcher 服务并取回报告。"""

    def __init__(self, client) -> None:
        self.client = client  # A2AClient | None
        super().__init__(
            name="a2a",
            description="跨智能体委派（A2A）：将研究任务委派给远端 researcher 并取回报告",
            expandable=True,
        )

    # 方法名 → 子工具名：a2a_delegate
    @tool_action(description="委派研究任务给远端 researcher（A2A），返回报告文本")
    def delegate(self, task: str) -> ToolResponse:
        """委派任务。

        Args:
            task: 完整的研究任务描述（远端将独立执行并产出报告）
        """
        if self.client is None:
            return ToolResponse.error(
                ToolErrorCode.INVALID_PARAM,
                "未配置 CORTEX_A2A_RESEARCHER_URL，无法使用 A2A 委派",
            )
        try:
            resp = self.client.delegate(task)
        except RuntimeError as exc:
            return ToolResponse.error(ToolErrorCode.NETWORK_ERROR, f"A2A 委派失败: {exc}")
        except TimeoutError:
            return ToolResponse.error(ToolErrorCode.TIMEOUT, "A2A 委派超时")
        except Exception as exc:
            return ToolResponse.error(
                ToolErrorCode.EXECUTION_ERROR, f"A2A 委派出错: {exc}"
            )
        artifact = resp.get("artifact") or ""
        status = resp.get("status") or ""
        prefix = "" if status == "completed" else f"[status={status}] "
        return ToolResponse.success(
            text=prefix + artifact[:_ARTIFACT_PREVIEW],
            data=resp,
        )

    # 抽象契约（展开路径不会直调父工具）
    def get_parameters(self) -> List[Any]:  # pragma: no cover
        from hello_agents.tools import ToolParameter

        return [ToolParameter(name="action", type="string", description="动作名")]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:  # pragma: no cover
        return ToolResponse.error(
            ToolErrorCode.INVALID_PARAM, "A2ATool 应通过 a2a_delegate 子工具调用"
        )
