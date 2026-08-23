"""AgentSet 装配：4 子代理先建，orchestrator 最后（TaskTool 解环在 subagent_factory）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from hello_agents.agents import PlanSolveAgent, ReActAgent, ReflectionAgent
from hello_agents.tools import ToolRegistry

from ..config import CortexConfig, framework_config
from . import prompts


@dataclass
class AgentSet:
    """全部装配好的代理；五代理共享同一注册表，角色可见性由过滤器裁剪

    （run_as_subagent 派发期临时摘除未授权工具，结束后还原）。
    """

    orchestrator: PlanSolveAgent
    retriever: ReActAgent
    researcher: ReActAgent
    writer: ReflectionAgent
    critic: ReflectionAgent

    def __iter__(self):
        return iter([self.orchestrator, self.retriever, self.researcher, self.writer, self.critic])


def build_agents(
    cfg: CortexConfig,
    strong_llm,
    weak_llm,
    registry: Optional[ToolRegistry],
) -> AgentSet:
    """构建五代理，全部挂同一 registry（可后补注册 TaskTool）。

    子代理 config 用 framework_config(subagent=True)：关闭其内部 Task/Todo/DevLog
    自动注册，避免与共享注册表冲突；工具隔离完全交给 run_as_subagent 的
    tool_filter —— 这要求子代理必须持有注册表，否则过滤器无从裁剪。
    """
    orch_cfg = framework_config(cfg)
    sub_cfg = framework_config(cfg, subagent=True)

    retriever = ReActAgent(
        name="retriever",
        llm=weak_llm,
        system_prompt=prompts.RETRIEVER,
        config=sub_cfg,
        tool_registry=registry,
        max_steps=12,
    )
    researcher = ReActAgent(
        name="researcher",
        llm=weak_llm,
        system_prompt=prompts.RESEARCHER,
        config=sub_cfg,
        tool_registry=registry,
        max_steps=20,
    )
    writer = ReflectionAgent(
        name="writer",
        llm=strong_llm,
        system_prompt=prompts.WRITER,
        config=sub_cfg,
        tool_registry=registry,
        max_iterations=2,
    )
    critic = ReflectionAgent(
        name="critic",
        llm=weak_llm,
        system_prompt=prompts.CRITIC,
        config=sub_cfg,
        tool_registry=registry,
        max_iterations=2,
    )
    orchestrator = PlanSolveAgent(
        name="orchestrator",
        llm=strong_llm,
        system_prompt=prompts.ORCHESTRATOR,
        planner_prompt=prompts.PLANNER_LIMIT,
        config=orch_cfg,
        tool_registry=registry,
        enable_tool_calling=True,
    )
    return AgentSet(orchestrator, retriever, researcher, writer, critic)
