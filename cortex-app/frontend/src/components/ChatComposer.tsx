import { useRef, useState } from "react";
import type { KeyboardEvent } from "react";

/** 对话输入区：Enter 发送 / Shift+Enter 换行；等待回复时显示停止。 */
export function ChatComposer({
  busy,
  onSend,
  onStop,
}: {
  busy: boolean;
  onSend: (text: string) => void;
  onStop?: () => void;
}) {
  const [value, setValue] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  const submit = () => {
    if (!value.trim() || busy) return;
    onSend(value);
    setValue("");
    requestAnimationFrame(() => ref.current?.focus());
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault(); // 中文输入法组词中的 Enter 不发送
      submit();
    }
  };

  return (
    <div className="flex items-end gap-2 rounded-lg border border-line bg-surface p-2 shadow-card focus-within:border-accent">
      <textarea
        ref={ref}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        rows={2}
        maxLength={4000}
        disabled={busy && !onStop ? true : false}
        placeholder={busy ? "编排器思考中…" : "输入消息，Enter 发送，Shift+Enter 换行"}
        aria-label="消息输入"
        className="max-h-40 min-h-11 flex-1 resize-none bg-transparent px-2.5 py-2 text-[15px] leading-6 text-ink placeholder:text-ink-2/60 focus:outline-none"
      />
      {busy ? (
        <button
          type="button"
          onClick={onStop}
          aria-label="停止生成"
          className="rounded-md border border-danger/50 px-4 py-2 text-sm font-medium text-danger transition-colors hover:bg-danger/10"
        >
          停止
        </button>
      ) : (
        <button
          type="button"
          onClick={submit}
          disabled={!value.trim()}
          aria-label="发送消息"
          className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-accent-strong disabled:cursor-not-allowed disabled:opacity-50"
        >
          发送
        </button>
      )}
    </div>
  );
}
