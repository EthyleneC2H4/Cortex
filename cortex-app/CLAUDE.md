# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Cortex：基于 hello-agents 1.0.0 框架的多智能体个人知识管理与研究助手。P1–P6 全部实现（规划文档见 `../docs/代码实现规划文档.md`，为权威结构参考）。所有代码注释与文档为简体中文。

## Commands

```bash
python3 -m venv .venv && source .venv/bin/activate   # 独立 venv；hello-agents==1.0.0 锁版本
pytest -m unit                   # 全 mock 离线（默认无网络无 docker）
pytest -m integration            # 本地真存储（sqlite/qdrant-local 文件）
pytest -m e2e                    # live 测试，仅 HAS_LLM_KEY=1 时执行（需 .env 配好 key）
pytest tests/tools/test_note_tool.py::TestCrud::test_update_optimistic_lock -m unit   # 单测
python -m cortex.prewarm         # 一次性：解析模型 id / tiktoken 缓存 / 下载 bge embedder / 双模型 ping
python -m cortex.app --chat      # CLI 多轮对话
python -m cortex.app --ingest PATH [--research TOPIC --mode pipeline|explore]
python -m cortex.eval.run --suite all [--no-rerank|--no-memory|--single-agent]   # → data/eval/<ts>/report.md
python -m cortex.api.server      # FastAPI+SSE :8000；frontend/dist 存在时同端口托管 Web UI
```

主要端点：`POST /research`（结构化阶段进度 SSE + 报告落盘 data/reports/）、`POST /chat/stream`、`POST /ingest`（上限 `CORTEX_MAX_UPLOAD_MB`，默认 50，422 详情已脱敏）、`POST /ingest-text`、`GET|DELETE /documents[/{doc_id}]`、`GET /reports` 与 `GET /reports/{filename}`（路径穿越防护）、`GET|POST /session`（会话存储 LRU+TTL：过期/逐出返回 **410**，从未存在才是 404）、`GET /health`。运行中再提交 /research → 409（同步检查后置位，无竞态窗口）。

## Web 前端（frontend/）

React 19 + Vite + Tailwind v4 四页 SPA（研究工作台/智能对话/知识库/系统状态）。命令在 `frontend/` 下：

```bash
cd frontend && export npm_config_cache=/tmp/npm-cache-cortex   # 本机 ~/.npm 有 root 属主文件时的绕行
npm run dev        # :5173 热更新，API 路径代理到 :8000（零 CORS）
npx tsc -b         # 类型检查（project references，勿用 --noEmit）
npx vitest run     # SSE 解析器边界/组件键盘契约/rehype 引用 chip 等
npm run build      # 生产构建 → dist/（server 检测到即托管，SPA 子路由有兜底）
```

- **SSE 消费**：EventSource 不能 POST → 自写 fetch+ReadableStream 解析器（`src/api/sse.ts` 纯函数）。**不自动重连**（防孤儿 run）；断连提示结果已落盘。
- **多轮上下文**：`/chat/stream` 每轮必须把问答写回 `orchestrator.history_manager.append(Message(...))`——漏掉则下一轮看不见本轮（live 门禁抓到过的缺陷）。
- **主题**：令牌唯一入口 `src/theme.css`（Tailwind v4 @theme + `.dark` 类切换，index.html 内联脚本防 FOUC）。terracotta 对比度仅 ≈3.2:1，正文级强调一律 accent-strong。
- **引用 chip**：报告/回复中的 `[doc_id/chunk_id]` 由自写 rehype 插件渲染为点击复制按钮。

`.env` 存真实 key（已 gitignore）；`.env.example` 只有占位符。`CORTEX_STORE_MODE=docker` 才需要 `docker compose up -d`（Qdrant+Neo4j）；默认零 docker。

## Architecture

装配链（`cortex/app.py:CortexApp.build`）：load_config → resolve_model_id → build_llms(strong temp0.3/weak temp0.0) → 共享存储 → KnowledgeService + MemoryService → ContextEngine(GSSC) → ToolRegistry(builtin+cortex) → build_agents(4 子代理先建，orchestrator 后建) → 注册 TaskTool 解环。服务单个失败只降级不阻断启动。

**硬约束：qdrant 本地模式排他锁** —— 同一 storage path 全进程只能有一个 QdrantClient。知识库（collection `cortex_chunks`）与记忆（`cortex_memory`）必须共享 app.py 里 `make_qdrant_client(cfg)` 创建的单例，经 `client=` 参数注入两个 store；测试里并行打开同一 path 会 AlreadyLocked，重开前必须先 close()。

**工具系统**：自研工具继承框架 `Tool` 且 `expandable=True`，`@tool_action` 方法展开为全局唯一子工具名（`note_create`/`knowledge_search`/`memory_search`/`citation_verify`/`mcp_call_tool`/`a2a_delegate`），父工具不注册。角色可见性唯一来源在 `cortex/tools/filters.py`（过滤器匹配的是展开名）。

**多智能体**：确定性研究用 `ResearchPipeline`（gather→research→write→critique→≤1 修订，`run_as_subagent` + 角色 filter）；开放探索用 orchestrator PlanSolveAgent 经 TaskTool 派发。关键机制：
- 子代理**必须持有共享 ToolRegistry**——过滤靠 run_as_subagent 派发期临时摘除条目实现，没注册表就无工具可用（builders.py 给全部五个代理传 registry）
- **所有流水线阶段都走 run_as_subagent + 角色 filter**，包括 writer/critic 文本阶段——直接 `.run()` 会看到含 TaskTool 的完整注册表，模型可能自行派发嵌套子代理失控（实测发生过）
- **阶段看门狗**：每阶段套墙钟上限（`CORTEX_STAGE_TIMEOUT`，默认 900s；0=不限时）——免费档经本地代理偶发 HTTP 读挂死且框架流式读不触发超时。超时放弃守护线程、产出占位文本并继续；被放弃阶段没走 finally 还原会留下"被过滤"的注册表，由 `registry_repair` 回调用装配完成时的工具快照写回兜底
- **进度观测**：`build_pipeline(progress_cb=…)` / `ExplorationMode(progress_cb=…)` 在阶段边界发射结构化 dict（run_start/stage_start/stage_end/note/run_end），`_safe_emit` 吞异常保证进度故障不拖垮流水线；CLI/eval 不传回调零影响。服务器经 worker 线程 + `queue.Queue` + `asyncio.to_thread` 桥接为 SSE
- `dispatch_subagent` 用 `return_summary=False` 取原始输出：框架 summary 会包“任务:…\n结果:…”外壳破坏 Findings JSON 解析
- `parse_findings` 宽容解析：平衡括号扫描 → 截断 JSON 字段级抢救（免费档生成上限截断常见）

**双模型分工**：strong=规划/写作（orchestrator/writer），weak=路由/检索/摘要/HyDE/记忆打分/judge（retriever/researcher/critic）。服务层 LLM 调用走 `llm/resilience.py` 重试；框架 agent 内部调用不包重试，但 `_run_agent_text` 对整次失败重试一次。

**RAG**：MarkItDown 解析 → 512/64 token 分块（标题栈进 heading_path 前缀，代码块/表格原子）→ embed → qdrant + BM25(CJK bigram) 双路 → RRF k=60 融合 → 三带分类（≥0.3 answer / ≥0.15 uncertain / refuse 拒答）。复杂查询 MQE 3 变体+HyDE 并行扇出。rank_bm25 在极小语料 idf=0，bm25_index 有 epsilon 兜底。

**记忆**：四类（working/episodic/semantic/procedural，procedural 默认关）；权重 0.6*相似+0.25*重要+0.15*新近；episodic→semantic 巩固与三策略软遗忘（绝不物理 DELETE）；异步写经单守护队列（`memory_async_writes`，测试直通）。

**评测**：读 data/traces 观测数据；bfcl=AST 等价匹配（多调用行为 name/args 平行列表；框架适配器返回**原生 OpenAI 响应形态**，tool_calls 在 choices[0].message 下、参数在 function.name/.arguments——解析必须走 `_extract_tool_calls` 兼容两种形态；**数据集每行必须带 instruction 且完整决定金标参数**——只有 args 没有指令时模型无从命中）/ gaia=quasi-exact-match 分级 / judge=成对 win-rate+κ；按条目 checkpoint 断点续跑应对限流。

## Testing conventions

markers 必须三选一标注（unit/integration/e2e）。e2e 用 `cfg.rebase(tmp_path)` 迁移全部数据目录到临时区（qdrant 排他锁使并行 live 互斥）；离线测试用 HashEmbedder(dim=128) + mock LLM。`KnowledgeService.__new__` 构造的测试桩需手动补 `_qdrant_client = None`。
