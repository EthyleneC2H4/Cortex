"""子代理派发桥：逻辑名 → 全新实例 + 角色过滤器。

- make_subagent_factory(agent_set)：给 TaskTool 用；每次派发**新建实例**
  （框架子代理机制要求隔离历史，共享实例会串上下文）
- dispatch_subagent：确定性流水线用；套角色 filter 调 run_as_subagent，
  返回归一化 dict {success, summary, tools_used}
"""

from __future__ import annotations

from typing import Callable

from ..tools.filters import filter_for_role

SUBAGENT_ROLES = ("retriever", "researcher", "writer", "critic")


def make_subagent_factory(agent_set) -> Callable[[str], object]:
    """返回 factory(role_name) -> 新建代理实例。未知角色抛 ValueError。

    新实例必须继承原型的 tool_registry：run_as_subagent 的工具过滤
    靠临时摘除注册表条目实现，没有注册表就无工具可用。
    """

    def factory(role: str):
        if role == "retriever":
            from hello_agents.agents import ReActAgent

            proto = agent_set.retriever
            return ReActAgent(name=proto.name, llm=proto.llm, system_prompt=proto.system_prompt,
                              config=proto.config, tool_registry=getattr(proto, "tool_registry", None),
                              max_steps=getattr(proto, "max_steps", 12))
        if role == "researcher":
            from hello_agents.agents import ReActAgent

            proto = agent_set.researcher
            return ReActAgent(name=proto.name, llm=proto.llm, system_prompt=proto.system_prompt,
                              config=proto.config, tool_registry=getattr(proto, "tool_registry", None),
                              max_steps=getattr(proto, "max_steps", 20))
        if role in ("writer", "critic"):
            from hello_agents.agents import ReflectionAgent

            proto = getattr(agent_set, role)
            return ReflectionAgent(name=proto.name, llm=proto.llm, system_prompt=proto.system_prompt,
                                   config=proto.config, tool_registry=getattr(proto, "tool_registry", None),
                                   max_iterations=getattr(proto, "max_iterations", 2))
        raise ValueError(f"未知子代理角色: {role}（可选 {SUBAGENT_ROLES}）")

    return factory


def dispatch_subagent(agent, task: str, role: str) -> dict:
    """套角色过滤器执行并归一化结果；异常转失败 dict（流水线不中断）。

    return_summary=False：取代理原始输出（框架 summary 会包上
    “任务: …\n结果: …”外壳，破坏 Findings JSON 解析）。
    """
    try:
        result = agent.run_as_subagent(task, tool_filter=filter_for_role(role),
                                       return_summary=False)
    except Exception as exc:  # noqa: BLE001 —— 单阶段失败不拖垮整条流水线
        return {"success": False, "summary": f"（{role} 执行异常: {exc}）", "tools_used": []}
    return {
        "success": bool(result.get("success", False)),
        "summary": result.get("result", ""),
        "tools_used": list((result.get("metadata") or {}).get("tools_used", [])),
    }
