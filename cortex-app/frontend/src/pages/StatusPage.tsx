import { useCallback, useEffect, useState } from "react";

import { apiGet } from "@/api/client";
import type { HealthInfo } from "@/api/types";
import { KnowledgeStatsCards } from "@/components/KnowledgeStatsCards";
import { ErrorBanner, Spinner } from "@/components/ui";
import { fmtDate, fmtDuration } from "@/lib/format";
import { humanizeNetworkError } from "@/lib/humanize";

/** 服务名中文映射（未知 key 原样展示）。 */
const SERVICE_LABELS: Record<string, string> = {
  knowledge: "知识库（RAG）",
  memory: "记忆服务",
  context: "上下文引擎",
  mcp: "MCP 协议",
  a2a: "A2A 协议",
};

const POLL_MS = 10_000;

function StatusDot({ ok }: { ok: boolean }) {
  return (
    <span
      aria-hidden="true"
      className={`inline-block size-2.5 rounded-full ${ok ? "bg-success" : "bg-danger"}`}
    />
  );
}

/** 系统状态页：轮询 /health，展示服务健康、知识库统计与运行约束提示。 */
export function StatusPage() {
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [checkedAt, setCheckedAt] = useState<Date | null>(null);
  const [error, setError] = useState<{ message: string; detail?: string } | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const refresh = useCallback(async () => {
    setRefreshing(true);
    try {
      const h = await apiGet<HealthInfo>("/health");
      setHealth(h);
      setCheckedAt(new Date());
      setError(null);
    } catch (err) {
      setError({ message: humanizeNetworkError(err) });
    } finally {
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
    const id = window.setInterval(() => void refresh(), POLL_MS);
    return () => window.clearInterval(id);
  }, [refresh]);

  return (
    <section aria-labelledby="status-title" aria-busy={refreshing} className="flex flex-col gap-6">
      <header className="flex items-start justify-between gap-4">
        <div>
          <h1 id="status-title" className="font-display text-2xl tracking-tight text-ink">
            系统状态
          </h1>
          <p className="mt-1 text-sm text-ink-2">
            每 {POLL_MS / 1000} 秒自动刷新
            {checkedAt && <span className="text-ink-2/70"> · 上次检查 {fmtDate(checkedAt.toISOString())}</span>}
          </p>
        </div>
        <button
          type="button"
          onClick={() => void refresh()}
          disabled={refreshing}
          className="shrink-0 rounded-md border border-line px-3.5 py-1.5 text-sm text-ink-2 transition-colors hover:bg-muted hover:text-ink disabled:opacity-50"
        >
          刷新
        </button>
      </header>

      {error && (
        <ErrorBanner message={error.message} detail={error.detail} onRetry={() => void refresh()} />
      )}

      {!health && !error && (
        <div className="flex items-center gap-2 rounded-lg border border-line bg-surface px-4 py-10 text-sm text-ink-2 shadow-card" role="status">
          <Spinner className="size-4" /> 正在获取服务状态…
        </div>
      )}

      {health && (
        <>
          {/* 总览卡 */}
          <dl className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <div className="rounded-md border border-line bg-surface px-4 py-3 shadow-card">
              <dt className="text-xs text-ink-2">服务状态</dt>
              <dd className="mt-0.5 flex items-center gap-2 font-display text-lg text-ink">
                <StatusDot ok={health.status === "ok"} />
                {health.status === "ok" ? "正常" : health.status}
              </dd>
            </div>
            <div className="rounded-md border border-line bg-surface px-4 py-3 shadow-card">
              <dt className="text-xs text-ink-2">版本</dt>
              <dd className="mt-0.5 font-display text-lg tabular-nums text-ink">{health.version ?? "—"}</dd>
            </div>
            <div className="rounded-md border border-line bg-surface px-4 py-3 shadow-card">
              <dt className="text-xs text-ink-2">运行时长</dt>
              <dd className="mt-0.5 font-display text-lg text-ink">
                {health.uptime_seconds !== undefined ? fmtDuration(health.uptime_seconds) : "—"}
              </dd>
            </div>
            <div className="rounded-md border border-line bg-surface px-4 py-3 shadow-card">
              <dt className="text-xs text-ink-2">研究任务</dt>
              <dd className="mt-0.5 font-display text-lg text-ink">
                {health.research_running ? (
                  <span className="flex items-center gap-2 text-warning">
                    <Spinner className="size-3.5" /> 执行中
                  </span>
                ) : (
                  "空闲"
                )}
              </dd>
            </div>
          </dl>

          {/* 服务健康列表 */}
          <div className="rounded-lg border border-line bg-surface p-4 shadow-card">
            <h2 className="font-display text-base text-ink">子服务</h2>
            <ul className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2">
              {Object.entries(health.services ?? {}).map(([key, ok]) => (
                <li key={key} className="flex items-center gap-2.5 rounded-md bg-muted/50 px-3 py-2 text-sm">
                  <StatusDot ok={Boolean(ok)} />
                  <span className="text-ink">{SERVICE_LABELS[key] ?? key}</span>
                  <span className="ml-auto text-xs text-ink-2">{ok ? "可用" : "未启用"}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* 知识库统计 */}
          <div className="rounded-lg border border-line bg-surface p-4 shadow-card">
            <h2 className="font-display text-base text-ink">知识库统计</h2>
            <div className="mt-3">
              <KnowledgeStatsCards stats={health.knowledge} />
            </div>
          </div>

          {/* 运行约束（诚实边界） */}
          <div className="rounded-lg border border-line bg-muted/50 p-4 text-sm leading-6 text-ink-2">
            <h2 className="font-display text-base text-ink">运行约束</h2>
            <ul className="mt-2 list-disc space-y-1 pl-5">
              <li>API 鉴权{health.auth_required ? "已启用（请求需携带 X-API-Key）" : "未启用（本机使用）"}。</li>
              <li>同一时刻仅允许一个研究任务，重复提交返回 409。</li>
              <li>对话会话保存在服务内存中，后端重启后需开启新对话。</li>
            </ul>
          </div>
        </>
      )}
    </section>
  );
}
