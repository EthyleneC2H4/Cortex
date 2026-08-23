/**
 * 研究运行状态机：SSE 事件 → 界面状态的纯函数归并。
 * 与 React 解耦（可 vitest 直测）；useResearchRun 负责副作用与计时。
 */

import type { ResearchDonePayload, StageProgress } from "@/api/types";

export type StageStatus = "pending" | "running" | "done" | "timeout" | "failed" | "skipped";

export interface StageState {
  key: string;
  label: string;
  status: StageStatus;
  /** 完成阶段：服务端报告的耗时（秒） */
  elapsed?: number;
  /** 运行中阶段：开始的 Date.now()，供计时器换算 */
  startedAtMs?: number;
}

export interface ResearchError {
  message: string;
  detail?: string;
  /** false 时不展示重试按钮（如 409：服务端已有任务在跑，重试必然再次失败） */
  retryable?: boolean;
}

export interface ResearchRunState {
  /** detached：SSE 连接已断但服务端仍在执行——轮询 /health 直到空闲（诚实呈现孤儿 run） */
  phase: "idle" | "running" | "done" | "error" | "detached";
  mode?: string;
  topic?: string;
  stages: StageState[];
  runStartedAtMs?: number;
  report?: ResearchDonePayload;
  error?: ResearchError;
}

export const STAGE_LABELS: Record<string, string> = {
  gather: "检索 Gather",
  research: "研究 Research",
  write: "写作 Write",
  critique: "审校 Critique",
  revise: "修订 Revise",
  explore: "自主探索 Explore",
};

export function stageLabel(key: string): string {
  return STAGE_LABELS[key] ?? key;
}

export function initialStages(keys: string[]): StageState[] {
  return keys.map((key) => ({ key, label: stageLabel(key), status: "pending" }));
}

/** 归并一枚进度事件；nowMs 由调用方传入（便于测试注入时钟）。 */
export function applyProgressEvent(
  state: ResearchRunState,
  ev: StageProgress,
  nowMs: number,
): ResearchRunState {
  switch (ev.type) {
    case "run_start":
      return {
        ...state,
        phase: "running",
        stages: initialStages(ev.stages),
        runStartedAtMs: nowMs,
        error: undefined,
        report: undefined,
      };

    case "stage_start":
      return {
        ...state,
        stages: state.stages.map((s) =>
          s.key === ev.stage ? { ...s, status: "running", startedAtMs: nowMs } : s,
        ),
      };

    case "stage_end":
      return {
        ...state,
        stages: state.stages.map((s) => {
          if (s.key !== ev.stage) return s;
          // ok===false：阶段彻底失败（非超时降级）——流水线会带占位继续，但 UI 必须如实示警
          const status: StageStatus = ev.ok === false ? "failed" : ev.timed_out ? "timeout" : "done";
          return {
            ...s,
            status,
            elapsed: ev.elapsed ?? s.elapsed,
            startedAtMs: undefined,
          };
        }),
      };

    case "note":
      // 目前唯一 note：revise_skipped —— 对应阶段标记为跳过
      if (ev.code === "revise_skipped") {
        return {
          ...state,
          stages: state.stages.map((s) =>
            s.key === "revise" ? { ...s, status: "skipped", startedAtMs: undefined } : s,
          ),
        };
      }
      return state;

    case "run_end":
      // 终态由 done/error SSE 事件驱动，这里只记录
      return state;

    default:
      return state;
  }
}
