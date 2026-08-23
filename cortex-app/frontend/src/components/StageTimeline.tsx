import type { ResearchRunState, StageState } from "@/lib/researchState";
import { fmtClock } from "@/lib/format";

/**
 * 五阶段时间线：长任务（28–50 分钟）的核心可见性组件。
 * 每行 = 图标状态 + 名称 + 计时；运行中每 500ms 跳动。
 */

function StatusGlyph({ stage }: { stage: StageState }) {
  switch (stage.status) {
    case "running":
      return (
        <svg viewBox="0 0 24 24" fill="none" className="size-5 animate-spin text-accent" aria-hidden="true">
          <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="3" opacity="0.2" />
          <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
        </svg>
      );
    case "done":
      return (
        <span className="flex size-5 items-center justify-center rounded-full bg-success/15 text-xs font-bold text-success" aria-hidden="true">
          ✓
        </span>
      );
    case "timeout":
      return (
        <span className="flex size-5 items-center justify-center rounded-full bg-warning/15 text-xs font-bold text-warning" aria-hidden="true">
          !
        </span>
      );
    case "failed":
      return (
        <span className="flex size-5 items-center justify-center rounded-full bg-danger/15 text-xs font-bold text-danger" aria-hidden="true">
          ✕
        </span>
      );
    case "skipped":
      return (
        <span className="flex size-5 items-center justify-center rounded-full bg-muted text-xs text-ink-2" aria-hidden="true">
          –
        </span>
      );
    default:
      return <span className="block size-5 rounded-full border-2 border-line" aria-hidden="true" />;
  }
}

function StageRow({ stage, nowMs }: { stage: StageState; nowMs: number }) {
  const live =
    stage.status === "running" && stage.startedAtMs !== undefined
      ? (nowMs - stage.startedAtMs) / 1000
      : null;
  const timedOut = stage.status === "timeout";

  return (
    <li
      className={`flex items-center gap-3 rounded-md px-3 py-2.5 transition-colors ${
        stage.status === "running" ? "bg-muted/70" : ""
      }`}
      aria-current={stage.status === "running" ? "step" : undefined}
    >
      <StatusGlyph stage={stage} />
      <div className="flex min-w-0 flex-1 items-baseline gap-2">
        <span className={`text-sm ${stage.status === "pending" ? "text-ink-2/70" : "text-ink"}`}>
          {stage.label}
        </span>
        {timedOut && (
          <span className="rounded-xs bg-warning/15 px-1.5 py-0.5 text-[11px] font-medium text-warning">
            看门狗超时 · 已降级继续
          </span>
        )}
        {stage.status === "failed" && (
          <span className="rounded-xs bg-danger/15 px-1.5 py-0.5 text-[11px] font-medium text-danger">
            阶段失败 · 流水线以占位内容继续
          </span>
        )}
        {stage.status === "skipped" && (
          <span className="rounded-xs bg-muted px-1.5 py-0.5 text-[11px] text-ink-2">已跳过</span>
        )}
      </div>
      {/* 计时列：完成用服务端耗时，运行中用本地时钟 */}
      <span className={`font-mono text-xs tabular-nums ${live !== null ? "text-accent-strong dark:text-accent" : "text-ink-2"}`}>
        {stage.status === "running"
          ? live !== null && fmtClock(live)
          : stage.elapsed !== undefined && stage.elapsed >= 0 && `${fmtClock(stage.elapsed)}`}
      </span>
    </li>
  );
}

export function StageTimeline({
  state,
  nowMs,
}: {
  state: ResearchRunState;
  /** 外部计时器注入的当前时刻（hook 里 setInterval 驱动） */
  nowMs: number;
}) {
  if (state.stages.length === 0) return null;
  const totalLive =
    state.runStartedAtMs !== undefined ? Math.max(0, (nowMs - state.runStartedAtMs) / 1000) : null;

  return (
    <section
      aria-label="研究阶段进度"
      aria-busy={state.phase === "running" || state.phase === "detached"}
      className="rounded-lg border border-line bg-surface p-4 shadow-card"
    >
      <div className="mb-2 flex items-baseline justify-between">
        <h2 className="font-display text-base text-ink">阶段进度</h2>
        {/* 断连等待期计时继续走：服务端确实仍在执行 */}
        {(state.phase === "running" || state.phase === "detached") && totalLive !== null && (
          <span className="font-mono text-xs tabular-nums text-ink-2">
            总用时 {fmtClock(totalLive)}
          </span>
        )}
      </div>
      {/* 屏幕阅读器播报当前阶段（视觉上由行内样式呈现） */}
      <p aria-live="polite" className="sr-only">
        {state.stages.find((s) => s.status === "running")?.label ?? ""}
      </p>
      <ul className="-mx-3 divide-y divide-line/50">
        {state.stages.map((s) => (
          <StageRow key={s.key} stage={s} nowMs={nowMs} />
        ))}
      </ul>
    </section>
  );
}
