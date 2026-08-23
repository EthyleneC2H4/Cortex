import { useEffect, useRef, useState } from "react";

import { saveApiKey } from "@/api/key";
import { useDialogA11y } from "@/hooks/useDialogA11y";

/**
 * API Key 弹窗：后端启用 X-API-Key 鉴权时（401 触发或手动打开）录入密钥。
 * 密钥只存本机 localStorage，随请求头发送，不落任何其他位置。
 */
export function ApiKeyModal({
  open,
  onClose,
  onSave,
  currentKey,
}: {
  open: boolean;
  onClose: () => void;
  onSave: (key: string) => void;
  currentKey: string;
}) {
  const [value, setValue] = useState(currentKey);
  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  useDialogA11y(open, containerRef, onClose);

  useEffect(() => {
    if (open) {
      setValue(currentKey);
      // 聚焦圈闭的起点
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [open, currentKey]);

  if (!open) return null;

  const submit = () => {
    saveApiKey(value.trim());
    onSave(value.trim());
    onClose();
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-ink/30 p-4"
      onClick={onClose}
      role="presentation"
    >
      <div
        ref={containerRef}
        role="dialog"
        aria-modal="true"
        aria-label="设置 API Key"
        className="w-full max-w-md rounded-lg border border-line bg-surface p-6 shadow-card"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 className="font-display text-lg text-ink">API Key</h2>
        <p className="mt-2 text-sm text-ink-2">
          服务端启用了鉴权（.env 中配置了 API_KEY）。密钥仅保存在本机浏览器。
        </p>
        <input
          ref={inputRef}
          type="password"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") submit();
          }}
          placeholder="粘贴 API Key"
          aria-label="API Key 输入"
          className="mt-4 w-full rounded-sm border border-line bg-page px-3 py-2 font-mono text-sm text-ink placeholder:text-ink-2/60 focus:border-accent"
        />
        <div className="mt-5 flex justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            className="rounded-md border border-line px-4 py-1.5 text-sm text-ink-2 transition-colors hover:bg-muted"
          >
            取消
          </button>
          <button
            type="button"
            onClick={submit}
            className="rounded-md bg-accent px-4 py-1.5 text-sm font-medium text-white transition-colors hover:bg-accent-strong"
          >
            保存
          </button>
        </div>
      </div>
    </div>
  );
}
