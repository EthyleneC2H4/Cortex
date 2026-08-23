import { useCallback, useEffect, useRef, useState } from "react";
import type { DragEvent } from "react";

import { apiGet, apiSend, apiUpload, ApiError } from "@/api/client";
import type { DocumentItem, HealthInfo, IngestResult } from "@/api/types";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { KnowledgeStatsCards } from "@/components/KnowledgeStatsCards";
import { EmptyState, ErrorBanner, Spinner } from "@/components/ui";
import { useToast } from "@/context/ToastContext";
import { fmtDate } from "@/lib/format";
import { humanizeNetworkError } from "@/lib/humanize";

/** 拖拽上传区：dragover 高亮，点击打开文件选择器。 */
function UploadZone({
  busy,
  onFile,
}: {
  busy: boolean;
  onFile: (f: File) => void;
}) {
  const [over, setOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const accept = (e: DragEvent) => {
    e.preventDefault();
    setOver(e.type === "dragover");
  };
  const drop = (e: DragEvent) => {
    e.preventDefault();
    setOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f) onFile(f);
  };

  return (
    <div
      role="button"
      tabIndex={0}
      aria-label="上传文档到知识库"
      aria-disabled={busy}
      onClick={() => !busy && inputRef.current?.click()}
      onKeyDown={(e) => {
        if ((e.key === "Enter" || e.key === " ") && !busy) {
          e.preventDefault();
          inputRef.current?.click();
        }
      }}
      onDragOver={accept}
      onDragLeave={accept}
      onDrop={drop}
      className={`flex cursor-pointer flex-col items-center justify-center gap-1.5 rounded-lg border-2 border-dashed px-6 py-10 text-center transition-colors ${
        over ? "border-accent bg-accent/8" : "border-line bg-surface hover:bg-muted/60"
      } ${busy ? "cursor-progress opacity-70" : ""}`}
    >
      <input
        ref={inputRef}
        type="file"
        className="hidden"
        // MarkItDown 支持面：pdf/docx/pptx/xlsx/html/txt/md 等
        accept=".pdf,.docx,.pptx,.xlsx,.html,.htm,.txt,.md,.csv,.json"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
          e.target.value = ""; // 允许重复选同一文件
        }}
      />
      {busy ? (
        <>
          <Spinner className="size-6" />
          <p className="text-sm text-ink">解析并嵌入中，视文档大小可能需要数十秒…</p>
        </>
      ) : (
        <>
          <p className="text-sm text-ink">
            拖拽文件到此处，或<span className="ml-1 font-medium text-accent-strong underline underline-offset-2">点击选择</span>
          </p>
          <p className="text-xs text-ink-2">支持 PDF / Word / PPT / Excel / HTML / Markdown 等文本格式</p>
        </>
      )}
    </div>
  );
}

/** 折叠的纯文本摄取表单。 */
function TextIngestForm({
  busy,
  onSubmit,
}: {
  busy: boolean;
  onSubmit: (title: string, content: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const canSubmit = title.trim().length > 0 && content.trim().length > 0 && !busy;

  if (!open) {
    return (
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="self-start rounded-md border border-line px-4 py-1.5 text-sm text-ink-2 transition-colors hover:bg-muted hover:text-ink"
      >
        或粘贴纯文本摄取 →
      </button>
    );
  }
  return (
    <form
      className="flex flex-col gap-3 rounded-lg border border-line bg-surface p-4 shadow-card"
      onSubmit={(e) => {
        e.preventDefault();
        if (canSubmit) onSubmit(title.trim(), content.trim());
      }}
    >
      <div className="flex items-center justify-between">
        <h3 className="font-display text-base text-ink">文本摄取</h3>
        <button
          type="button"
          onClick={() => setOpen(false)}
          className="text-sm text-ink-2 hover:text-ink"
          aria-label="收起文本摄取表单"
        >
          收起 ✕
        </button>
      </div>
      <input
        value={title}
        onChange={(e) => setTitle(e.target.value)}
        maxLength={300}
        placeholder="标题（必填）"
        aria-label="文档标题"
        className="rounded-md border border-line bg-page px-3 py-2 text-sm text-ink placeholder:text-ink-2/60 focus:border-accent focus:outline-none"
      />
      <textarea
        value={content}
        onChange={(e) => setContent(e.target.value)}
        rows={6}
        maxLength={100_000}
        placeholder="粘贴正文内容…"
        aria-label="文档内容"
        className="resize-y rounded-md border border-line bg-page px-3 py-2 text-sm leading-6 text-ink placeholder:text-ink-2/60 focus:border-accent focus:outline-none"
      />
      <div className="flex justify-end">
        <button
          type="submit"
          disabled={!canSubmit}
          className="rounded-md bg-accent px-4 py-1.5 text-sm font-medium text-white transition-colors hover:bg-accent-strong disabled:cursor-not-allowed disabled:opacity-50"
        >
          {busy ? "摄取中…" : "摄取"}
        </button>
      </div>
    </form>
  );
}

/** 知识库管理页：上传 / 文本摄取 / 文档列表 / 删除确认。 */
export function KnowledgePage() {
  const toast = useToast();
  const [docs, setDocs] = useState<DocumentItem[]>([]);
  const [stats, setStats] = useState<HealthInfo["knowledge"]>(undefined);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false); // 文件上传占用
  const [textBusy, setTextBusy] = useState(false); // 文本摄取占用
  const [error, setError] = useState<{ message: string; detail?: string } | null>(null);
  const [pendingDelete, setPendingDelete] = useState<DocumentItem | null>(null);
  const [deleting, setDeleting] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const list = await apiGet<{ documents: DocumentItem[]; total: number }>("/documents");
      setDocs(list.documents);
      setError(null);
    } catch (err) {
      if (err instanceof ApiError) setError({ message: err.humanMessage, detail: err.detail });
      else setError({ message: humanizeNetworkError(err) });
    } finally {
      setLoading(false);
    }
    try {
      const h = await apiGet<HealthInfo>("/health");
      setStats(h.knowledge);
    } catch {
      /* 统计仅辅助展示 */
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const ingestFile = async (f: File) => {
    if (uploading) return;
    setUploading(true);
    setError(null);
    try {
      const r = await apiUpload<IngestResult>("/ingest", f);
      toast(`已摄取「${r.title || f.name}」，切分为 ${r.n_chunks} 块`, "success");
      await refresh();
    } catch (err) {
      if (err instanceof ApiError) setError({ message: err.humanMessage, detail: err.detail });
      else setError({ message: humanizeNetworkError(err) });
    } finally {
      setUploading(false);
    }
  };

  const ingestText = async (title: string, content: string) => {
    setTextBusy(true);
    setError(null);
    try {
      const r = await apiSend<IngestResult>("POST", "/ingest-text", { title, content });
      toast(`已摄取「${title}」，切分为 ${r.n_chunks} 块`, "success");
      await refresh();
    } catch (err) {
      if (err instanceof ApiError) setError({ message: err.humanMessage, detail: err.detail });
      else setError({ message: humanizeNetworkError(err) });
    } finally {
      setTextBusy(false);
    }
  };

  const doDelete = async () => {
    if (!pendingDelete) return;
    setDeleting(true);
    try {
      await apiSend("DELETE", `/documents/${pendingDelete.doc_id}`);
      toast(`已删除「${pendingDelete.title || pendingDelete.doc_id}」`, "success");
      setPendingDelete(null);
      await refresh();
    } catch (err) {
      const msg = err instanceof ApiError ? err.humanMessage : humanizeNetworkError(err);
      toast(`删除失败：${msg}`, "error");
    } finally {
      setDeleting(false);
    }
  };

  return (
    <section aria-labelledby="kb-title" className="flex flex-col gap-6">
      <header>
        <h1 id="kb-title" className="font-display text-2xl tracking-tight text-ink">
          知识库
        </h1>
        <p className="mt-1 text-sm text-ink-2">摄取文档、管理私有语料；对话与研究将自动检索这里的内容</p>
      </header>

      <KnowledgeStatsCards stats={stats} />

      {error && (
        <ErrorBanner message={error.message} detail={error.detail} onRetry={() => void refresh()} />
      )}

      <UploadZone busy={uploading} onFile={(f) => void ingestFile(f)} />
      <TextIngestForm busy={textBusy} onSubmit={(t, c) => void ingestText(t, c)} />

      {/* 文档列表 */}
      <div className="overflow-x-auto rounded-lg border border-line bg-surface shadow-card">
        {loading ? (
          <div className="flex items-center gap-2 px-4 py-10 text-sm text-ink-2" role="status">
            <Spinner className="size-4" /> 加载文档列表…
          </div>
        ) : docs.length === 0 ? (
          <EmptyState
            title="还没有文档"
            hint="上传第一个文件后，它会在这里出现，并可在研究与对话中被检索引用"
          />
        ) : (
          <table className="w-full min-w-[36rem] text-left text-sm">
            <caption className="sr-only">知识库文档列表</caption>
            <thead>
              <tr className="border-b border-line text-xs uppercase tracking-wide text-ink-2">
                <th scope="col" className="px-4 py-2.5 font-medium">标题</th>
                <th scope="col" className="px-4 py-2.5 font-medium">来源</th>
                <th scope="col" className="px-4 py-2.5 font-medium">分块</th>
                <th scope="col" className="px-4 py-2.5 font-medium">摄取时间</th>
                <th scope="col" className="px-4 py-2.5 font-medium text-right">操作</th>
              </tr>
            </thead>
            <tbody>
              {docs.map((d) => (
                <tr key={d.doc_id} className="border-b border-line/60 last:border-b-0 hover:bg-muted/50">
                  <td className="max-w-56 truncate px-4 py-3 text-ink" title={d.title}>
                    {d.title || d.doc_id}
                  </td>
                  <td className="max-w-40 truncate px-4 py-3 text-ink-2" title={d.source}>
                    {d.source || "—"}
                  </td>
                  <td className="px-4 py-3 tabular-nums text-ink-2">{d.n_chunks}</td>
                  <td className="whitespace-nowrap px-4 py-3 text-ink-2">
                    {d.created_at ? fmtDate(d.created_at) : "—"}
                  </td>
                  <td className="px-4 py-3 text-right">
                    <button
                      type="button"
                      onClick={() => setPendingDelete(d)}
                      className="rounded-sm px-2.5 py-1 text-danger transition-colors hover:bg-danger/10"
                    >
                      删除
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <ConfirmDialog
        open={pendingDelete !== null}
        title="删除这篇文档？"
        body={
          pendingDelete
            ? `「${pendingDelete.title || pendingDelete.doc_id}」的 ${pendingDelete.n_chunks} 个分块及其向量、倒排索引将被移除，操作不可撤销。注意：知识图谱中的历史关联暂不随删除清除。`
            : ""
        }
        confirmLabel={deleting ? "删除中…" : "确认删除"}
        danger
        onCancel={() => setPendingDelete(null)}
        onConfirm={() => void doDelete()}
      />
    </section>
  );
}
