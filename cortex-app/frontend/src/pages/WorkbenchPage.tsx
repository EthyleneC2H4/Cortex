import { useState } from "react";
import type { FormEvent } from "react";

import { ReportPanel } from "@/components/ReportPanel";
import { StageTimeline } from "@/components/StageTimeline";
import { ErrorBanner } from "@/components/ui";
import { useResearchRun } from "@/hooks/useResearchRun";
import type { ResearchMode } from "@/hooks/useResearchRun";
import { useTicker } from "@/hooks/useTicker";
import { useToast } from "@/context/ToastContext";

const MODE_OPTIONS: Array<{ value: ResearchMode; title: string; desc: string }> = [
  {
    value: "pipeline",
    title: "确定性流水线",
    desc: "检索 → 研究 → 写作 → 审校 → 修订，五阶段可评测可复现（推荐）",
  },
  {
    value: "explore",
    title: "自主探索",
    desc: "编排器自行规划路径，适合边界模糊的开放问题",
  },
];

/** 研究工作台：提交主题 → 五阶段实时时间线 → 带引用研报。 */
export function WorkbenchPage() {
  const { state, start, stop, dismissOrphan } = useResearchRun();
  const toast = useToast();
  const [topic, setTopic] = useState("");
  const [mode, setMode] = useState<ResearchMode>("pipeline");
  // 状态机不依赖 React；这里只在运行中/断连等待时驱动计时器跳动（服务端确实仍在执行）
  const nowMs = useTicker(state.phase === "running" || state.phase === "detached");

  const running = state.phase === "running";
  const detached = state.phase === "detached";
  const locked = running || detached;
  const canSubmit = topic.trim().length > 0 && !locked;

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (!canSubmit) return;
    void start(topic.trim(), mode);
  };

  const retry = () => {
    if (state.topic) void start(state.topic, (state.mode as ResearchMode | undefined) ?? "pipeline");
    else setTopic((t) => t); // 无历史主题时仅刷新界面
  };

  return (
    <section aria-labelledby="wb-title">
      <h1 id="wb-title" className="mb-1 font-display text-2xl tracking-tight text-ink">
        研究工作台
      </h1>
      <p className="mb-8 text-sm text-ink-2">每条引用都可核验的研究报告，五个阶段全程可见</p>

      {/* 提交表单 */}
      <form
        onSubmit={submit}
        aria-busy={locked}
        className="rounded-lg border border-line bg-surface p-5 shadow-card"
      >
        <label htmlFor="topic" className="mb-2 block text-sm font-medium text-ink">
          研究主题
        </label>
        <textarea
          id="topic"
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          rows={3}
          maxLength={2000}
          disabled={locked}
          placeholder="例如：对比检索增强生成中的混合检索策略"
          className="w-full resize-y rounded-md border border-line bg-page px-3.5 py-2.5 text-[15px] leading-6 text-ink placeholder:text-ink-2/60 focus:border-accent focus:outline-none disabled:opacity-60"
        />

        <fieldset className="mt-4">
          <legend className="sr-only">研究模式</legend>
          <div className="grid gap-3 sm:grid-cols-2">
            {MODE_OPTIONS.map((opt) => (
              <label
                key={opt.value}
                className={`cursor-pointer rounded-md border px-4 py-3 transition-colors ${
                  mode === opt.value
                    ? "border-accent bg-accent/8"
                    : "border-line hover:border-ink-2/40"
                } ${locked ? "opacity-60" : ""}`}
              >
                <input
                  type="radio"
                  name="mode"
                  value={opt.value}
                  checked={mode === opt.value}
                  onChange={() => setMode(opt.value)}
                  disabled={locked}
                  className="sr-only"
                />
                <span className="block text-sm font-medium text-ink">{opt.title}</span>
                <span className="mt-0.5 block text-xs leading-5 text-ink-2">{opt.desc}</span>
              </label>
            ))}
          </div>
        </fieldset>

        <div className="mt-5 flex items-center gap-3">
          {running ? (
            <button
              type="button"
              onClick={stop}
              className="rounded-md border border-danger/50 px-6 py-2.5 text-sm font-medium text-danger transition-colors hover:bg-danger/10"
            >
              停止
            </button>
          ) : detached ? (
            <button
              type="button"
              onClick={dismissOrphan}
              className="rounded-md border border-warning/50 px-6 py-2.5 text-sm font-medium text-warning transition-colors hover:bg-warning/10"
            >
              我知道了
            </button>
          ) : (
            <button
              type="submit"
              disabled={!canSubmit}
              className="rounded-md bg-accent px-6 py-2.5 text-sm font-medium text-white shadow-sm transition-colors hover:bg-accent-strong disabled:cursor-not-allowed disabled:opacity-50"
            >
              开始研究
            </button>
          )}
          <p className="text-xs leading-5 text-ink-2">
            免费档完整研究约需 28–50 分钟。
            {running && "停止只断开连接，服务端会跑完并把报告落盘。"}
          </p>
        </div>
      </form>

      {/* 断连等待：连接已断但服务端仍在执行——诚实呈现孤儿 run 并轮询恢复 */}
      {detached && (
        <div
          role="status"
          className="mt-5 rounded-lg border border-warning/40 bg-warning/8 p-4"
        >
          <p className="font-display text-sm font-medium text-warning">连接已中断，服务端仍在执行本轮研究</p>
          <p className="mt-1 text-xs leading-5 text-ink-2">
            完成后报告会自动落盘到服务器 data/reports/ 目录。此页面每 5 秒检测一次服务端状态，
            任务结束后自动恢复可提交；期间无法开始新研究（会返回 409）。
          </p>
        </div>
      )}

      {/* 错误横幅：409 等重试必然再失败的情形不提供重试按钮 */}
      {state.phase === "error" && state.error && (
        <div className="mt-5">
          <ErrorBanner
            message={state.error.message}
            detail={state.error.detail}
            onRetry={state.error.retryable === false ? undefined : retry}
          />
        </div>
      )}

      {/* 阶段时间线：一旦有阶段事件就持续显示（含断连等待与结束后回看） */}
      {(state.stages.length > 0 || locked) && (
        <div className="mt-5">
          <StageTimeline state={state} nowMs={nowMs} />
        </div>
      )}

      {/* 报告面板 */}
      {state.phase === "done" && state.report && (
        <div className="mt-5">
          <ReportPanel report={state.report} onNotice={toast} />
        </div>
      )}
    </section>
  );
}
