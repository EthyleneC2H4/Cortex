"""MemoryTool — 长期记忆工具（P4）。

expandable → ``memory_add / memory_search / memory_summary / memory_forget /
memory_consolidate`` 五个子工具；type 大小写不敏感；procedural 未启用时
memory_add 返回 INVALID_PARAM 并说明原因。所有失败返回结构化
ToolResponse.error，绝不抛异常。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from hello_agents.tools import Tool, ToolResponse, tool_action
from hello_agents.tools.errors import ToolErrorCode

from ..services.memory.types import MemoryType


class MemoryTool(Tool):
    """长期记忆：写入、向量检索、摘要、遗忘与巩固。"""

    def __init__(self, service, *, procedural_enabled: Optional[bool] = None) -> None:
        self.service = service  # MemoryService
        # 工具层开关：显式传入可进一步收紧；cfg 关闭时服务层仍会拒绝（不可绕过）
        self.procedural_enabled = (
            bool(service.cfg.procedural_memory_enabled)
            if procedural_enabled is None
            else bool(procedural_enabled)
        )
        super().__init__(
            name="memory",
            description="个人长期记忆：写入/检索/摘要/遗忘/巩固",
            expandable=True,
        )

    # 方法名 → 子工具名：memory_add / memory_search / memory_summary /
    # memory_forget / memory_consolidate
    @tool_action(description="写入一条长期记忆；procedural 类型默认禁用")
    def add(self, content: str, type: str = "episodic", importance: Optional[float] = None) -> ToolResponse:
        """写入一条长期记忆。

        Args:
            content: 记忆内容文本（非空）
            type: 记忆类型 working/episodic/semantic/procedural，大小写不敏感
            importance: 重要性 0..1；缺省由弱模型打分或回退 0.5
        """
        if not str(content).strip():
            return ToolResponse.error(ToolErrorCode.INVALID_PARAM, "记忆内容不能为空")
        try:
            mt = MemoryType.parse(type)
        except ValueError as exc:
            return ToolResponse.error(ToolErrorCode.INVALID_PARAM, str(exc))
        if mt is MemoryType.PROCEDURAL and not self.procedural_enabled:
            return ToolResponse.error(
                ToolErrorCode.INVALID_PARAM,
                "procedural 记忆未启用（procedural_memory_enabled=False）：程序性技能"
                "记忆需要显式配置开启后才能写入",
            )
        try:
            item = self.service.remember(
                content=str(content),
                type=mt.value,
                importance=None if importance is None else float(importance),
            )
        except ValueError as exc:
            return ToolResponse.error(ToolErrorCode.INVALID_PARAM, str(exc))
        except Exception as exc:
            return ToolResponse.error(ToolErrorCode.EXECUTION_ERROR, f"写入记忆失败: {exc}")
        return ToolResponse.success(
            text=f"已记住 ({item.type}, importance={item.importance:.2f}) {item.content[:80]}",
            data={"id": item.id, "type": item.type, "importance": item.importance},
        )

    @tool_action(description="按语义检索长期记忆，返回相关条目列表")
    def search(self, query: str, top_k: int = 5) -> ToolResponse:
        """语义检索记忆。

        Args:
            query: 查询文本
            top_k: 返回条数上限
        """
        if not str(query).strip():
            return ToolResponse.error(ToolErrorCode.INVALID_PARAM, "查询词不能为空")
        try:
            hits = self.service.recall(str(query), top_k=int(top_k))
        except Exception as exc:
            return ToolResponse.error(ToolErrorCode.EXECUTION_ERROR, f"检索失败: {exc}")
        if not hits:
            return ToolResponse.success(text=f"没有与「{query}」相关的记忆", data={"hits": []})
        lines = [f"- [{h['score']:.3f}] ({h['type']}) {h['content'][:120]}" for h in hits]
        return ToolResponse.success(text="\n".join(lines), data={"hits": hits})

    @tool_action(description="把当前重要记忆总结成一段话")
    def summary(self) -> ToolResponse:
        """生成记忆摘要。"""
        try:
            text = self.service.summary()
        except Exception as exc:
            return ToolResponse.error(ToolErrorCode.EXECUTION_ERROR, f"生成摘要失败: {exc}")
        if not text:
            return ToolResponse.success(text="（暂无记忆）", data={"summary": ""})
        return ToolResponse.success(text=text, data={"summary": text})

    @tool_action(description="执行三策略软遗忘（低重要性/过旧/容量超额），绝不物理删除")
    def forget(self) -> ToolResponse:
        """运行遗忘策略。"""
        try:
            report = self.service.forget()
        except Exception as exc:
            return ToolResponse.error(ToolErrorCode.EXECUTION_ERROR, f"遗忘失败: {exc}")
        text = (
            f"遗忘完成：低重要性遗忘 {len(report['forgotten_low_importance'])} 条，"
            f"过旧遗忘 {len(report['forgotten_old'])} 条，"
            f"容量归档 {len(report['archived_for_capacity'])} 条，"
            f"剩余 active {report['active_after']} 条"
        )
        return ToolResponse.success(text=text, data=report)

    @tool_action(description="执行记忆巩固：工作→情景→语义升级与近重复合并")
    def consolidate(self) -> ToolResponse:
        """运行巩固流程。"""
        try:
            report = self.service.consolidate()
        except Exception as exc:
            return ToolResponse.error(ToolErrorCode.EXECUTION_ERROR, f"巩固失败: {exc}")
        text = (
            f"巩固完成：工作→情景 {len(report['working_to_episodic'])} 条，"
            f"情景→语义 {len(report['episodic_to_semantic'])} 条，"
            f"合并重复 {len(report['merged_duplicates'])} 组"
        )
        return ToolResponse.success(text=text, data=report)

    # 抽象契约（展开路径不会直调父工具）
    def get_parameters(self) -> List[Any]:  # pragma: no cover
        from hello_agents.tools import ToolParameter

        return [ToolParameter(name="action", type="string", description="动作名")]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:  # pragma: no cover
        return ToolResponse.error(
            ToolErrorCode.INVALID_PARAM, "MemoryTool 应通过 memory_* 子工具调用"
        )
