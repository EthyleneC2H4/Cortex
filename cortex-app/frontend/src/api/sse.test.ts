import { describe, expect, it, vi } from "vitest";

import { consumeSse, extractEvents } from "./sse";
import { SseHttpError, ssePost } from "./sse";

describe("extractEvents（纯解析器）", () => {
  it("解析单个 LF 分隔事件", () => {
    const { events, rest } = extractEvents('event: done\ndata: {"a":1}\n\n');
    expect(events).toEqual([{ event: "done", data: '{"a":1}' }]);
    expect(rest).toBe("");
  });

  it("处理 CRLF 边界", () => {
    const { events } = extractEvents("event: progress\r\ndata: hello\r\n\r\n");
    expect(events).toEqual([{ event: "progress", data: "hello" }]);
  });

  it("忽略心跳注释块（无 data 的块不产生事件）", () => {
    const { events, rest } = extractEvents(": ping - keepalive\n\n");
    expect(events).toEqual([]);
    expect(rest).toBe("");
  });

  it("多行 data 以 \\n 连接；event 缺省为 message", () => {
    const { events } = extractEvents("data: line1\ndata: line2\n\n");
    expect(events).toEqual([{ event: "message", data: "line1\nline2" }]);
  });

  it("不完整的事件留在 buffer 中等待下个 chunk", () => {
    const first = extractEvents("event: progress\ndata: 半个");
    expect(first.events).toEqual([]);
    const second = extractEvents(`${first.rest} 事件\n\n`);
    expect(second.events).toEqual([{ event: "progress", data: "半个 事件" }]);
  });

  it("data 值冒号后多个空格只剥一个（SSE 规范行为）", () => {
    const { events } = extractEvents("data:  前导空格保留\n\n");
    expect(events).toEqual([{ event: "message", data: " 前导空格保留" }]);
  });
});

/** 构造一个按给定「原始字节块」序列流出的 SSE Response（可从多字节字符中间切断）。 */
function sseResponseBytes(chunks: Uint8Array[]): Response {
  let i = 0;
  const stream = new ReadableStream({
    pull(controller) {
      if (i < chunks.length) controller.enqueue(chunks[i++]);
      else controller.close();
    },
  });
  return new Response(stream, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

describe("consumeSse（流式消费）", () => {
  it("跨 chunk 截断的多字节中文正确拼接（TextDecoder stream 模式）", async () => {
    // 'data: {"content": "' 恰好 19 个 ASCII 字节；第 20 字节落在 你（3 字节）中间
    const full = new TextEncoder().encode('data: {"content": "你好世界"}\n\n');
    const cut = 20;
    const received: string[] = [];
    await consumeSse(sseResponseBytes([full.slice(0, cut), full.slice(cut)]), (m) =>
      received.push(m.data),
    );
    expect(JSON.parse(received[0])).toEqual({ content: "你好世界" });
  });

  it("EOF 无尾随空行时冲刷最后一个完整事件", async () => {
    const received: { event: string; data: string }[] = [];
    await consumeSse(sseResponseBytes([new TextEncoder().encode("event: done\ndata: tail")]), (m) =>
      received.push({ event: m.event, data: m.data }),
    );
    expect(received).toEqual([{ event: "done", data: "tail" }]);
  });

  it("心跳注释不触发回调，后续事件正常送达", async () => {
    const encoder = new TextEncoder();
    const received: string[] = [];
    await consumeSse(
      sseResponseBytes([
        encoder.encode(": ping - keepalive\n\n"),
        encoder.encode('event: message\ndata: {"ok":true}\n\n'),
      ]),
      (m) => received.push(m.data),
    );
    expect(received).toEqual(['{"ok":true}']);
  });
});

describe("ssePost", () => {
  it("非 2xx 抛 SseHttpError 并携带 status/detail", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response(JSON.stringify({ detail: "已有研究任务在执行中" }), {
          status: 409,
          headers: { "content-type": "application/json" },
        }),
      ),
    );
    await expect(ssePost("/research", {}, { onEvent: () => {} })).rejects.toMatchObject({
      status: 409,
      detail: "已有研究任务在执行中",
    });
    vi.unstubAllGlobals();
  });

  it("正常流逐事件回调", async () => {
    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode("event: session\ndata: {\"id\":\"s1\"}\n\n"));
        controller.enqueue(encoder.encode("event: done\ndata: {}\n\n"));
        controller.close();
      },
    });
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response(stream, { status: 200, headers: { "content-type": "text/event-stream" } }),
      ),
    );
    const seen: string[] = [];
    await ssePost("/chat/stream", { message: "hi" }, { onEvent: (e) => seen.push(e) });
    expect(seen).toEqual(["session", "done"]);
    vi.unstubAllGlobals();
  });
});

// SseHttpError 导出自检（保持导入被使用）
describe("SseHttpError", () => {
  it("记录 status 与 detail", () => {
    const err = new SseHttpError(500, "boom");
    expect(err.status).toBe(500);
    expect(err.detail).toBe("boom");
  });
});
