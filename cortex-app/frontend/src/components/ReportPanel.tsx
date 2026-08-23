import type { ResearchDonePayload } from "@/api/types";
import { MarkdownView } from "@/components/MarkdownView";
import { copyText, downloadText } from "@/lib/download";
import { fmtDuration } from "@/lib/format";

/** 研报结果面板：元信息条 + Markdown 正文 + 复制/下载。 */
export function ReportPanel({
  report,
  onNotice,
}: {
  report: ResearchDonePayload;
  /** chip 复制成功等提示的出口（页面接 toast） */
  onNotice?: (msg: string) => void;
}) {
  const filenameBase = `cortex-report-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "")}`;

  return (
    <section aria-label="研究报告" className="rounded-lg border border-line bg-surface shadow-card">
      {/* 元信息条 */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-line px-5 py-3">
        <span className="rounded-full bg-success/15 px-2.5 py-0.5 text-xs font-medium text-success">
          已完成
        </span>
        <span className="text-sm text-ink-2">用时 {fmtDuration(report.duration_seconds)}</span>
        <span className="text-sm text-ink-2">
          {report.mode === "explore" ? "自主探索模式" : "流水线模式"}
        </span>
        {report.revised ? (
          <span className="rounded-full bg-accent/15 px-2.5 py-0.5 text-xs font-medium text-accent-strong dark:text-accent">
            已经审校修订
          </span>
        ) : (
          <span className="rounded-full bg-muted px-2.5 py-0.5 text-xs text-ink-2">未经修订</span>
        )}
        <div className="ml-auto flex gap-2">
          <button
            type="button"
            onClick={() =>
              void copyText(report.report_md).then((ok) =>
                onNotice?.(ok ? "已复制全文" : "复制失败"),
              )
            }
            className="rounded-md border border-line px-3 py-1 text-xs text-ink transition-colors hover:bg-muted"
          >
            复制全文
          </button>
          <button
            type="button"
            onClick={() => downloadText(`${filenameBase}.md`, report.report_md)}
            className="rounded-md border border-line px-3 py-1 text-xs text-ink transition-colors hover:bg-muted"
          >
            下载 .md
          </button>
        </div>
      </div>

      {/* 正文 */}
      <div className="px-6 py-5">
        <MarkdownView
          text={report.report_md}
          onCitationCopy={(id) => onNotice?.(`已复制引用 ${id}`)}
        />
      </div>

      {report.report_path && (
        <p className="border-t border-line px-5 py-2.5 text-xs text-ink-2">
          报告已落盘：<code className="font-mono">{report.report_path}</code>（断开连接也不会丢失）
        </p>
      )}
    </section>
  );
}
