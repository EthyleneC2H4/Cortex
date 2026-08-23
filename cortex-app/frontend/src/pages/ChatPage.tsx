import { useCallback, useEffect, useRef, useState } from "react";

import { ChatComposer } from "@/components/ChatComposer";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { MarkdownView } from "@/components/MarkdownView";
import { Typewriter } from "@/components/Typewriter";
import { ErrorBanner, Spinner } from "@/components/ui";
import { useChatStream } from "@/hooks/useChatStream";
import type { ChatMessage } from "@/hooks/useChatStream";
import { useToast } from "@/context/ToastContext";

function UserBubble({ content }: { content: string }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[85%] rounded-lg rounded-br-sm bg-accent/12 px-4 py-2.5 text-[15px] leading-7 whitespace-pre-wrap text-ink">
        {content}
      </div>
    </div>
  );
}

function AssistantBubble({
  message,
  onRevealTick,
}: {
  message: ChatMessage;
  onRevealTick: () => void;
}) {
  // 打字机只服务「刚收到」的消息；显示完毕即切正式 Markdown 渲染（引用 chip、列表等）
  const [revealed, setRevealed] = useState(() => !message.fresh);
  return (
    <div className="flex flex-col items-start gap-1">
      <div className="max-w-full rounded-lg rounded-bl-sm border border-line bg-surface px-4 py-3 text-[15px] leading-7 shadow-card">
        {revealed ? (
          <MarkdownView text={message.content} />
        ) : (
          <Typewriter
            text={message.content}
            onTick={onRevealTick}
            onDone={() => setRevealed(true)}
          />
        )}
      </div>
      {message.duration_seconds !== undefined && (
        <span className="pl-1 text-[11px] text-ink-2/70">
          用时 {Math.round(message.duration_seconds)} 秒
        </span>
      )}
    </div>
  );
}

/** 智能对话页：气泡流 + SSE 整轮回复 + 新对话重置。 */
export function ChatPage() {
  const { messages, busy, error, send, stop, newChat } = useChatStream();
  const toast = useToast();
  const [confirmReset, setConfirmReset] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  const lastUserIndex = messages.map((m) => m.role).lastIndexOf("user");

  // 新消息与打字机逐字推进时贴底
  const scrollToBottom = useCallback(() => {
    bottomRef.current?.scrollIntoView({ block: "end" });
  }, []);

  useEffect(scrollToBottom, [messages, scrollToBottom]);

  return (
    <section aria-labelledby="chat-title" className="flex min-h-[calc(100dvh-9rem)] flex-col">
      <div className="mb-4 flex items-start justify-between gap-4">
        <div>
          <h1 id="chat-title" className="font-display text-2xl tracking-tight text-ink">
            智能对话
          </h1>
          <p className="mt-1 text-sm text-ink-2">
            与 Cortex 编排器多轮对话，对话会写入长期记忆。
            <span className="text-ink-2/70">会话保存在服务内存中，后端重启后需开启新对话。</span>
          </p>
        </div>
        <button
          type="button"
          onClick={() => setConfirmReset(true)}
          className="shrink-0 rounded-md border border-line px-3.5 py-1.5 text-sm text-ink-2 transition-colors hover:bg-muted hover:text-ink"
        >
          新对话
        </button>
      </div>

      {/* 消息区（忙碌状态由下方 role=status 指示条播报，容器不整段播报） */}
      <div
        className="flex-1 overflow-y-auto rounded-lg bg-muted/40 p-4"
        aria-busy={busy}
      >
        {messages.length === 0 && !busy && (
          <div className="flex h-full min-h-56 flex-col items-center justify-center gap-2 text-center">
            <img src="/favicon.svg" alt="" className="size-10 rounded-lg opacity-80" />
            <p className="font-display text-base text-ink">开始一段新对话</p>
            <p className="max-w-xs text-xs leading-5 text-ink-2">
              可以先在「知识库」摄取资料，对话中的问题将结合你的私有语料与长期记忆回答
            </p>
          </div>
        )}

        <div className="flex flex-col gap-4">
          {messages.map((m, i) =>
            m.role === "user" ? (
              <UserBubble key={i} content={m.content} />
            ) : (
              <AssistantBubble key={i} message={m} onRevealTick={scrollToBottom} />
            ),
          )}
          {busy && (
            <div className="flex items-center gap-2 pl-1 text-sm text-ink-2" role="status">
              <Spinner className="size-4" />
              编排器规划与执行中，免费档可能需要一两分钟…
            </div>
          )}
        </div>
        <div ref={bottomRef} />
      </div>

      {error && (
        <div className="mt-3">
          <ErrorBanner
            message={error.message}
            detail={error.detail}
            onRetry={messages.length > 0 && lastUserIndex >= 0 ? () => void send(messages[lastUserIndex].content) : undefined}
          />
        </div>
      )}

      {/* 输入区 */}
      <div className="mt-3">
        <ChatComposer
          busy={busy}
          onSend={(t) => void send(t)}
          onStop={() => {
            // 诚实边界：停止只断开本地流，服务端本轮仍会执行完毕并写入记忆库
            stop();
            toast("已停止等待回复；服务端本轮仍会执行完毕并写入记忆库");
          }}
        />
      </div>

      <ConfirmDialog
        open={confirmReset}
        title="开启新对话？"
        body="当前对话上下文将被清空（编排器不再记得本轮内容）。沉淀的长期记忆不受影响。"
        confirmLabel="开启新对话"
        danger
        onCancel={() => setConfirmReset(false)}
        onConfirm={() => {
          newChat();
          setConfirmReset(false);
          toast("已开启新对话");
        }}
      />
    </section>
  );
}
