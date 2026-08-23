# Cortex

基于 [hello-agents](https://github.com/jjyaoao/HelloAgents) 框架的多智能体个人知识管理与研究助手：私有知识库检索（RAG）、跨会话长期记忆、多智能体协作研究与三层评估。设计文档见 `../doc/`（规划文档为权威）。

## 快速开始

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # 填入 LLM_API_KEY（OpenRouter 兼容；只放这里）
python -m cortex.prewarm      # 解析模型 id / 预热 tiktoken / 下载 embedder（一次性）
pytest -m unit                # 离线测试
python -m cortex.app --chat   # CLI 对话
```

## 架构

```
load_config → 双档 LLM(strong/weak)
→ stores(qdrant-local|docker + graph + sqlite) → KnowledgeService/MemoryService
→ ToolRegistry(builtin + note/knowledge/memory/citation/mcp/a2a)
→ ContextEngine(GSSC) → agents(retriever/researcher/writer/critic → orchestrator)
→ ResearchPipeline(run_as_subagent) | Exploration(TaskTool)
→ FastAPI+SSE (/research /ingest /session)
```

- 默认**零 docker**：qdrant-client 本地模式 + networkx 图 + sqlite；`CORTEX_STORE_MODE=docker` 时 `docker compose up -d`。
- 联网能力 MCP-first（P6）；MCP 不可用时降级为仅知识库 + 拒答带。
- 测试分层：`unit`（全 mock）/ `integration`（本地真存储）/ `e2e`（`HAS_LLM_KEY=1` 才跑）。

## 阶段状态

| Phase | 内容 | 状态 |
|---|---|---|
| P1 | 配置 + 双档 LLM + 单 Agent CLI | ✅ |
| P2 | 工具系统（NoteTool/Registry/Filters） | ✅ |
| P3 | KnowledgeService（RAG） | ✅ |
| P4 | MemoryService | ✅ |
| P5 | ContextEngine + 多智能体编排 | ✅ |
| P6 | 协议 + 评估 + API | ✅ |

环境变量见 `.env.example`。
