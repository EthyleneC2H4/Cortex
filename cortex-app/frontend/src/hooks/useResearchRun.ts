import { useCallback, useEffect, useRef, useState } from "react";

import { apiGet } from "@/api/client";
import { ssePost, SseHttpError } from "@/api/sse";
import type { StageProgress } from "@/api/types";
import { useApiKey } from "@/context/ApiKeyContext";
import { humanizeNetworkError } from "@/lib/humanize";
import { applyProgressEvent } from "@/lib/researchState";
import type { ResearchRunState } from "@/lib/researchState";

export type ResearchMode = "pipeline" | "explore";

/** 断连后孤儿 run 检测的轮询间隔 */
const ORPHAN_POLL_MS = 5000;

/**
 * 研究工作台的运行 hook：SSE 消费 + 状态归并 + 中断控制。
 *
 * 连接断开（用户停止 / 网络中断）≠ 任务终止：服务端会继续执行并把报告落盘。
 * 此时进入 detached 态——保留已收到的阶段痕迹与计时、轮询 /health 的
 * research_running，直到服务端空闲再回到 idle；期间禁止提交（必然 409）。
 */
export function useResearchRun() {
  const [state, setState] = useState<ResearchRunState>({ phase: "idle", stages: [] });
  const abortRef = useRef<AbortController | null>(null);
  const { openModal } = useApiKey();

  // 卸载时中断流（服务端任务继续跑完并落盘，UI 如实提示）
  useEffect(() => () => abortRef.current?.abort(), []);

  // detached 态：轮询健康检查，服务端空闲后回到 idle（结果已在 data/reports/）
  useEffect(() => {
    if (state.phase !== "detached") return;
    let cancelled = false;
    const id = window.setInterval(() => {
      void (async () => {
        try {
          const h = await apiGet<{ research_running?: boolean }>("/health");
          if (!cancelled && h.research_running === false) {
            setState((s) => (s.phase === "detached" ? { ...s, phase: "idle" } : s));
          }
        } catch {
          /* 健康检查暂不可达：继续轮询，不惊扰用户 */
        }
      })();
    }, ORPHAN_POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [state.phase]);

  const start = useCallback(
    async (topic: string, mode: ResearchMode) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      setState({ phase: "running", mode, topic, stages: [] });

      try {
        await ssePost("/research", { topic, mode }, {
          signal: controller.signal,
          onEvent: (event, data) => {
            if (event === "progress") {
              try {
                const ev = JSON.parse(data) as StageProgress;
                setState((s) => applyProgressEvent(s, ev, Date.now()));
              } catch {
                /* 非法 JSON 容错：跳过该枚进度事件 */
              }
            } else if (event === "done") {
              try {
                const report = JSON.parse(data);
                setState((s) => ({ ...s, phase: "done", report }));
              } catch {
                setState((s) => ({
                  ...s,
                  phase: "error",
                  error: { message: "结果解析失败", detail: data.slice(0, 300) },
                }));
              }
            } else if (event === "error") {
              let message = "执行出错，请稍后重试";
              let detail: string | undefined;
              try {
                const p = JSON.parse(data);
                message = String(p.message ?? message);
                detail = p.detail ? String(p.detail) : undefined;
              } catch {
                detail = data.slice(0, 300);
              }
              setState((s) => ({ ...s, phase: "error", error: { message, detail } }));
            }
          },
        });
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") {
          // 用户主动停止（或卸载）：服务端仍会执行完毕 → 进入 detached 轮询
          setState((s) => (s.phase === "running" ? { ...s, phase: "detached" } : s));
        } else if (err instanceof SseHttpError && err.status === 401) {
          openModal();
          setState((s) => ({
            ...s,
            phase: "error",
            error: { message: "需要 API Key 或密钥无效", detail: err.detail },
          }));
        } else if (err instanceof SseHttpError && err.status === 409) {
          // 提交被拒：另一个任务正在服务端执行——重试只会再次 409，不提供重试按钮
          setState((s) => ({
            ...s,
            phase: "error",
            error: { message: err.detail || "已有研究任务在执行中，请等待其完成后再提交", retryable: false },
          }));
        } else {
          // 网络类失败：若已收到过进度事件说明流中途断开，服务端大概率仍在执行；
          // 一枚事件都没有则多半是连接根本没建立，如实报错并允许重试
          setState((s) =>
            s.stages.length > 0
              ? { ...s, phase: "detached" }
              : { ...s, phase: "error", error: { message: humanizeNetworkError(err) } },
          );
        }
      } finally {
        abortRef.current = null;
      }
    },
    [openModal],
  );

  /** 停止按钮：只断开连接；后端会跑完并把报告落盘（文案如实说明），随后转 detached 轮询。 */
  const stop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  /** 用户确认知晓「服务端仍在执行」后手动退出 detached 态。 */
  const dismissOrphan = useCallback(() => {
    setState((s) => (s.phase === "detached" ? { ...s, phase: "idle" } : s));
  }, []);

  return { state, start, stop, dismissOrphan };
}
