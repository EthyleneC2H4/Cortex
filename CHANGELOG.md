# 更新日志

本项目的所有重要变更记录于此。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.3.0/)；
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 加固

- **API**：会话存储改为 LRU+TTL（容量/时限经 `CORTEX_SESSION_MAX` / `CORTEX_SESSION_TTL_H`
  可配），过期或被逐出的会话返回 410 而非 404；API key 比较改用
  `hmac.compare_digest` 常量时间比较；`POST /research` 并发提交的 409 判定改为
  同步检查后置位，消除竞态；`POST /ingest` 增加上传体积上限
  （`CORTEX_MAX_UPLOAD_MB`，默认 50）与脱敏后的 422 错误详情；新增报告历史端点
  `GET /reports` 与 `GET /reports/{filename}`（带路径穿越防护）。
- **评测**：checkpoint 写入改为「临时文件 + 原子改名」，限流中断不再作废全部进度；
  bfcl/gaia 新增固定种子自展 95% 置信区间；judge 引入位置交换双评，新增
  `swap_agreement_rate` 指标量化裁判位置偏置；BFCL 参数解析失败置 None 哨兵，
  绝不与「无参金标」假匹配；GAIA 匹配支持全角逗号列表。
- **核心**：配置快速失败校验（记忆权重和、分块重叠、阶段超时）；CLI/服务器聊天
  路径在召回前冲刷异步记忆写队列（读己之写）；上下文截断不再撕开未闭合的引用
  标记 `[doc/chunk`；token 计数器按配置模型逐级回退。

### 新增

- 可选 BGE 交叉编码器重排：`CORTEX_ENABLE_RERANK=1` + `pip install -e ".[rerank]"`，
  依赖缺失或加载失败时软降级回纯 RRF 融合，不阻断检索链路。
- Web 打字机动画支持键盘操作（Enter/Space 立即补全），尊重
  `prefers-reduced-motion`；播完自动摘除交互语义。
- 双语 README（英文为 GitHub 默认显示，中文版 `README.zh-CN.md`）与本更新日志。

### 工程化

- pyproject 注册控制台入口 `cortex` / `cortex-eval`；markitdown 依赖收上界 `<0.2`；
  开发状态分类器修正为 Beta。
- docker-compose 镜像 tag 固定（qdrant v1.19.0 / neo4j 5.26.30）；CI 增加并发组
  自动取消在途运行；接入 Dependabot（pip 周更、npm/docker 月更，hello-agents 锁定）。
- CI lint 步骤明确标注检查范围 `--select E4,E7,E9,F`。

## [1.0.0] - 2026-08-23

P1–P6 六阶段规划全部落地：工具系统与注册表、RAG 检索链（多查询扩展 + HyDE +
混合检索 RRF 融合）、四类记忆服务、GSSC 上下文引擎、多智能体研究流水线
（确定性 pipeline + 自主 explore 双模式）、FastAPI+SSE 服务与 React 前端、
BFCL/GAIA/Judge 三套评测与观测追踪。

[Unreleased]: https://github.com/EthyleneC2H4/Cortex/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/EthyleneC2H4/Cortex/releases/tag/v1.0.0
