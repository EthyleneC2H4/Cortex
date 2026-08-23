"""GSSC 上下文引擎：Gather → Select → Compress → Structure。

包装框架 ``hello_agents.context``（ContextBuilder/TokenCounter），把
记忆/RAG 命中经 additional_packets 注入。预算 = 窗口×0.7×(1-reserve_ratio)；
历史超过预算 50% 时用弱模型滚动摘要压缩（必留计划/结论/未解问题/引用）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol

# 各 packet 类型的 token 配额（Select 阶段）
TYPE_QUOTAS = {
    "task_state": 800,
    "related_memory": 1200,
    "knowledge_base": 2000,
    "history": 1500,
}

# 历史压缩时必须保留的行前缀（计划/结论/未解问题/引用）
_KEEP_PREFIXES = ("[PLAN]", "[结论]", "[未解]", "[doc", "[")


class _WeakLLM(Protocol):
    def invoke(self, messages): ...


@dataclass
class Packet:
    """统一上下文包；type ∈ ContextConfig 支持的 metadata.type。"""

    type: str            # instructions / task_state / related_memory / knowledge_base / history / retrieval
    text: str
    weight: float = 0.5  # Select 阶段类型内归一化权重


@dataclass
class BuiltContext:
    system_prompt: str
    packets: List[Packet] = field(default_factory=list)
    budget_tokens: int = 0
    used_tokens: int = 0
    compressed: bool = False


_COMPRESS_PROMPT = """把下面的对话历史压缩为要点，保留：
1) 任务计划与已完成步骤 [PLAN]
2) 关键结论 [结论]
3) 未解问题 [未解]
4) 所有引用标记 [doc_id/chunk_id] 与来源
直接输出要点列表，不要解释。

对话历史：
{history}
"""


class ContextEngine:
    """GSSC 四段流水线；产出可直接注入 agent.system_prompt 的文本。"""

    def __init__(self, cfg, memory=None, knowledge=None, weak_llm=None) -> None:
        from hello_agents.context import TokenCounter

        self.cfg = cfg
        self.memory = memory          # MemoryService | None
        self.knowledge = knowledge    # KnowledgeService | None
        self.weak_llm = weak_llm
        try:
            self.counter = TokenCounter(model="gpt-4o-mini")
            self._count = self.counter.count_text
        except Exception:
            self._count = lambda s: max(1, len(s) // 3)

    # ---------------------------------------------------------------- #
    @property
    def budget(self) -> int:
        """总预算 = 窗口×0.7×(1-reserve_ratio)。"""
        window = int(getattr(self.cfg, "context_window", 128000) or 128000)
        reserve = 0.15
        return int(window * 0.7 * (1 - reserve))

    def gather(
        self,
        task: str,
        history: Optional[List[dict]] = None,
        *,
        session_id: str = "default",
        extra_packets: Optional[List[Packet]] = None,
    ) -> List[Packet]:
        """Gather：指令 + 任务 + 记忆 + 知识库 + 历史。"""
        packets: List[Packet] = []
        if self.memory is not None:
            for item in self.memory.recall_for_context(limit=10, min_importance=0.3):
                packets.append(Packet(type="related_memory",
                                      text=f"- ({item.get('type', '?')}) {item.get('content', '')}",
                                      weight=0.5))
        if self.knowledge is not None and task.strip():
            for hit in self.knowledge.search_for_context(task, limit=5):
                src = hit.get("source", "")
                cid = hit.get("chunk_id", "")
                packets.append(Packet(type="knowledge_base",
                                      text=f"[{cid}] {src}\n{hit.get('text', '')}",
                                      weight=0.9))
        if history:
            recent = list(history)[-5:]
            packets.append(Packet(
                type="history",
                text="\n".join(f"{m.get('role', 'user')}: {str(m.get('content', ''))[:400]}" for m in recent),
                weight=0.6,
            ))
        packets.append(Packet(type="task_state", text=f"当前任务：{task}", weight=1.0))
        if extra_packets:
            packets.extend(extra_packets)
        return packets

    # ---------------------------------------------------------------- #
    def select(self, packets: List[Packet]) -> List[Packet]:
        """Select：类型配额内按权重贪心装箱（skip-and-continue）。"""
        by_type: dict = {}
        for p in packets:
            by_type.setdefault(p.type, []).append(p)
        out: List[Packet] = []
        used_total = 0
        overhead = 800  # 结构骨架 + 指令预留
        for ptype, group in by_type.items():
            quota = TYPE_QUOTAS.get(ptype, 1000)
            quota = min(quota, max(200, self.budget - overhead - used_total))
            total_w = sum(max(p.weight, 0.01) for p in group) or 1.0
            spent = 0
            ranked = sorted(group, key=lambda p: p.weight / total_w, reverse=True)
            for p in ranked:
                n = self._count(p.text)
                share = int(quota * max(p.weight, 0.01) / total_w)
                take = min(n, max(share, 120))  # 保底 120 token 片段，截断而非整条丢弃
                if spent + take > quota:
                    continue
                out.append(Packet(type=ptype, text=_truncate(p.text, take), weight=p.weight))
                spent += take
            used_total += spent
        return out

    # ---------------------------------------------------------------- #
    def compress_history(self, history_text: str, budget_tokens: int) -> tuple[str, bool]:
        """历史 > 预算 50% 时弱模型滚动摘要；失败原样返回。"""
        n = self._count(history_text)
        if n <= budget_tokens * 0.5 or self.weak_llm is None:
            return history_text, False
        keep = [ln for ln in history_text.splitlines() if any(ln.startswith(k) for k in _KEEP_PREFIXES)]
        try:
            resp = self.weak_llm.invoke([{"role": "user", "content": _COMPRESS_PROMPT.format(history=history_text[:6000])}])
            summary = str(getattr(resp, "content", "")).strip()
        except Exception:
            summary = _truncate(history_text, budget_tokens // 2)
        merged = summary + ("\n" + "\n".join(keep) if keep else "")
        return merged, True

    # ---------------------------------------------------------------- #
    def structure(self, task: str, selected: List[Packet], compressed_history: str = "") -> str:
        """Structure：固定分区序，LLM 解析最稳的布局。"""
        sections: dict = {}
        for p in selected:
            sections.setdefault(p.type, []).append(p.text)

        parts = [
            "# Role & Policies\n你是 Cortex 研究助手：只依据提供的证据作答，"
            "事实性陈述必须带引用标记 [doc_id/chunk_id]，证据不足时明确声明。",
            "# Task\n" + task,
        ]
        if sections.get("related_memory"):
            parts.append("# Relevant Memory\n" + "\n".join(sections["related_memory"]))
        if sections.get("knowledge_base"):
            parts.append("# Evidence (Knowledge Base)\n" + "\n---\n".join(sections["knowledge_base"]))
        if compressed_history or sections.get("history"):
            hist = compressed_history or "\n".join(sections["history"])
            parts.append("# Conversation History\n" + hist)
        return "\n\n".join(parts)

    # ---------------------------------------------------------------- #
    def build_system_prompt(
        self,
        task: str,
        history: Optional[List[dict]] = None,
        *,
        session_id: str = "default",
        extra_packets: Optional[List[Packet]] = None,
    ) -> BuiltContext:
        packets = self.gather(task, history, session_id=session_id, extra_packets=extra_packets)
        selected = self.select(packets)
        hist_packet = next((p for p in selected if p.type == "history"), None)
        compressed = ""
        did_compress = False
        if hist_packet is not None:
            compressed, did_compress = self.compress_history(hist_packet.text, TYPE_QUOTAS["history"])
        prompt = self.structure(task, [p for p in selected if p.type != "history"], compressed)
        used = sum(self._count(p.text) for p in selected)
        return BuiltContext(system_prompt=prompt, packets=selected,
                            budget_tokens=self.budget, used_tokens=used, compressed=did_compress)


def _truncate(text: str, max_chars_hint: int) -> str:
    """按 token 预算近似截断（token≈字符数按计数器换算过，这里保守 1:2.5）。"""
    limit = max(80, int(max_chars_hint * 2.5))
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"
