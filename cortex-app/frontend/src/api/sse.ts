/**
 * SSE 客户端：fetch POST + ReadableStream 手工解析。
 *
 * 为什么不用 EventSource：浏览器原生 EventSource 只能 GET，无法带 JSON 体；
 * 为什么不用 @microsoft/fetch-event-source：核心只是 ~60 行解析器，自写可控、
 * 可精确测试多字节中文跨 chunk 边界（TextDecoder stream 模式）。
 *
 * 不自动重连：研究任务断线重连会造成孤儿 run 且拿不到已完成的结果 ——
 * UI 明示「连接中断，结果已落盘」，根治方案（job store）属未来工作。
 */

export interface SseMessage {
  event: string;
  data: string;
}

const BOUNDARY = /\r?\n\r?\n/;

/** 解析一段 SSE 文本增量：返回其中的完整事件与未完结的剩余 buffer。纯函数。 */
export function extractEvents(buffer: string): { events: SseMessage[]; rest: string } {
  const events: SseMessage[] = [];
  let rest = buffer;
  let m: RegExpExecArray | null;
  while ((m = BOUNDARY.exec(rest)) !== null) {
    const block = rest.slice(0, m.index);
    rest = rest.slice(m.index + m[0].length);
    const ev = parseBlock(block);
    if (ev) events.push(ev);
  }
  return { events, rest };
}

/** 解析单个事件块；无 data 的块（纯注释/心跳）返回 null。 */
function parseBlock(block: string): SseMessage | null {
  let event = "message";
  const data: string[] = [];
  for (const line of block.split(/\r?\n/)) {
    if (!line || line.startsWith(":")) continue; // 注释/心跳（sse-starlette ping 即保活）
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1); // 规范：冒号后一个可选空格
    if (field === "event") event = value;
    else if (field === "data") data.push(value);
    // id / retry 忽略（不自动重连）
  }
  if (data.length === 0) return null;
  return { event, data: data.join("\n") };
}

/** 消费一个已建立的 SSE Response，逐事件回调。EOF 时冲刷最后一块。 */
export async function consumeSse(
  res: Response,
  onEvent: (msg: SseMessage) => void,
): Promise<void> {
  if (!res.body) throw new Error("响应无可读流");
  const reader = res.body.getReader();
  // stream:true 让跨 chunk 截断的多字节中文在解码器内部正确拼接
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const { events, rest } = extractEvents(buffer);
      buffer = rest;
      for (const e of events) onEvent(e);
    }
    buffer += decoder.decode(); // 冲刷解码器尾部
    const tail = extractEvents(`${buffer}\n\n`); // EOF 无尾随空行的最后一个事件
    for (const e of tail.events) onEvent(e);
  } finally {
    reader.releaseLock();
  }
}

export interface SsePostOptions {
  apiKey?: string;
  signal?: AbortSignal;
  /** 收到非法 JSON 载荷时也回调原始文本（容错不静默）。 */
  onEvent: (event: string, data: string) => void;
}

/** POST JSON 并消费 SSE 流；连接失败/非 2xx/非事件流抛错（调用方负责人性化展示）。 */
export async function ssePost(
  url: string,
  body: unknown,
  opts: SsePostOptions,
): Promise<void> {
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(opts.apiKey ? { "X-API-Key": opts.apiKey } : {}),
    },
    body: JSON.stringify(body),
    signal: opts.signal,
  });
  if (!res.ok) {
    let detail = "";
    try {
      detail = String((await res.json())?.detail ?? "");
    } catch {
      /* 非 JSON 错误体 */
    }
    throw new SseHttpError(res.status, detail);
  }
  const contentType = res.headers.get("content-type") ?? "";
  if (!contentType.includes("text/event-stream")) {
    throw new Error("服务响应不是事件流（请确认直连后端而非中间代理错误页）");
  }
  await consumeSse(res, (m) => opts.onEvent(m.event, m.data));
}

/** SSE 前置失败（非 2xx）；status 供上层映射中文文案。 */
export class SseHttpError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(`HTTP ${status}`);
    this.name = "SseHttpError";
    this.status = status;
    this.detail = detail;
  }
}
