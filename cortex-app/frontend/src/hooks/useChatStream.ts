import { useCallback, useEffect, useRef, useState } from "react";

import type { ChatMessagePayload, SessionInfoT } from "@/api/types";
import { apiGet } from "@/api/client";
import { ssePost, SseHttpError } from "@/api/sse";
import { useApiKey } from "@/context/ApiKeyContext";
import { humanizeNetworkError } from "@/lib/humanize";

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  /** assistant 消息是否正在打字机展示 */
  fresh?: boolean;
  duration_seconds?: number;
}

const SESSION_KEY = "cortex-chat-session";

/**
 * 对话流 hook：消息列表 + SSE 消费 + 会话持久化。
 *
 * v1 诚实边界（UI 同步提示）：后端编排器同步 run 无 token 流 → 整轮返回；
 * 会话保存在服务进程内存，后端重启即失。
 * 「停止」只断开本地流（AbortController）——服务端本轮仍会执行完毕并写记忆，
 * 文案如实说明。代次计数器保证被中止/重置的旧连接事件不再污染新状态。
 */
export function useChatStream() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<{ message: string; detail?: string } | null>(null);
  // 「新对话」后置位：下一次发送携带 reset=true 清空编排器内部历史
  const pendingReset = useRef(false);
  const abortRef = useRef<AbortController | null>(null);
  // 代次：每次发送递增；旧代（已停止/已被取代）的事件一律丢弃
  const genRef = useRef(0);
  const { openModal } = useApiKey();

  // 刷新后恢复会话历史（进程未重启时有效）
  useEffect(() => {
    const sid = localStorage.getItem(SESSION_KEY);
    if (!sid) return;
    void (async () => {
      try {
        // 走 apiGet 以携带 X-API-Key；服务未启动等场景静默，页面显示空态
        const info = await apiGet<SessionInfoT>(`/session/${sid}`);
        setMessages(
          info.history.map((m) => ({
            role: m.role === "user" ? ("user" as const) : ("assistant" as const),
            content: m.content,
            fresh: false,
          })),
        );
        setSessionId(sid);
      } catch {
        /* 会话不存在（服务重启）或网络不可达：保持空态 */
      }
    })();
  }, []);

  // 卸载时断开在途流，避免对已卸载组件 setState
  useEffect(
    () => () => {
      abortRef.current?.abort();
    },
    [],
  );

  const send = useCallback(
    async (text: string) => {
      const content = text.trim();
      if (!content || busy) return;
      const gen = ++genRef.current;
      const controller = new AbortController();
      abortRef.current = controller;
      setError(null);
      setBusy(true);
      setMessages((ms) => [...ms, { role: "user", content }]);
      const reset = pendingReset.current;
      pendingReset.current = false;

      try {
        await ssePost("/chat/stream", { message: content, session_id: sessionId, reset }, {
          signal: controller.signal,
          onEvent: (event, data) => {
            if (genRef.current !== gen) return; // 已被停止/新发送/新对话取代
            if (event === "session") {
              try {
                const sid = JSON.parse(data).session_id as string;
                setSessionId(sid);
                localStorage.setItem(SESSION_KEY, sid);
              } catch {
                /* 容错 */
              }
            } else if (event === "message") {
              try {
                const payload = JSON.parse(data) as ChatMessagePayload;
                setMessages((ms) => [
                  ...ms,
                  {
                    role: "assistant",
                    content: payload.content,
                    fresh: true,
                    duration_seconds: payload.duration_seconds,
                  },
                ]);
              } catch {
                setError((e) => e ?? { message: "回复解析失败" });
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
              setError({ message, detail });
            }
          },
        });
      } catch (err) {
        if (genRef.current !== gen) {
          /* 本轮已被停止或取代：不再改任何状态 */
        } else if (err instanceof DOMException && err.name === "AbortError") {
          /* 用户停止：保留已有内容 */
        } else if (err instanceof SseHttpError && err.status === 401) {
          openModal();
          setError({ message: "需要 API Key 或密钥无效", detail: err.detail });
        } else {
          setError({ message: humanizeNetworkError(err) });
        }
      } finally {
        if (abortRef.current === controller) abortRef.current = null;
        if (genRef.current === gen) setBusy(false);
      }
    },
    [busy, sessionId, openModal],
  );

  /** 停止等待：断开本地 SSE 流。服务端本轮仍会执行完毕并写入记忆库。 */
  const stop = useCallback(() => {
    genRef.current += 1; // 作废在途事件
    abortRef.current?.abort();
    abortRef.current = null;
    setBusy(false);
  }, []);

  /** 新对话：先断开在途流再清空视图；下次发送 reset=true 让编排器重置内部历史。 */
  const newChat = useCallback(() => {
    stop();
    pendingReset.current = true;
    setMessages([]);
    setSessionId(null);
    setError(null);
    localStorage.removeItem(SESSION_KEY);
  }, [stop]);

  return { messages, sessionId, busy, error, send, stop, newChat };
}
