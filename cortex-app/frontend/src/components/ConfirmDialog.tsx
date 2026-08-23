import { useEffect, useRef } from "react";

import { useDialogA11y } from "@/hooks/useDialogA11y";

/** 通用确认对话框（破坏性操作二次确认）：焦点圈闭 + Esc 关闭 + 关闭后焦点归还。 */
export function ConfirmDialog({
  open,
  title,
  body,
  confirmLabel = "确认",
  danger = false,
  onConfirm,
  onCancel,
}: {
  open: boolean;
  title: string;
  body?: string;
  confirmLabel?: string;
  /** 危险操作（删除等）用红色确认键 */
  danger?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const confirmRef = useRef<HTMLButtonElement>(null);
  useDialogA11y(open, containerRef, onCancel);

  useEffect(() => {
    if (open) requestAnimationFrame(() => confirmRef.current?.focus());
  }, [open]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-ink/30 p-4"
      onClick={onCancel}
      role="presentation"
    >
      <div
        ref={containerRef}
        role="alertdialog"
        aria-modal="true"
        aria-label={title}
        className="w-full max-w-sm rounded-lg border border-line bg-surface p-6 shadow-card"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 className="font-display text-lg text-ink">{title}</h2>
        {body && <p className="mt-2 text-sm leading-6 text-ink-2">{body}</p>}
        <div className="mt-5 flex justify-end gap-2">
          <button
            type="button"
            onClick={onCancel}
            className="rounded-md border border-line px-4 py-1.5 text-sm text-ink-2 transition-colors hover:bg-muted"
          >
            取消
          </button>
          <button
            ref={confirmRef}
            type="button"
            onClick={onConfirm}
            className={`rounded-md px-4 py-1.5 text-sm font-medium text-white transition-colors ${
              danger ? "bg-danger hover:bg-danger/85" : "bg-accent hover:bg-accent-strong"
            }`}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
