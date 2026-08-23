/** 轻量 fetch 封装：自动附 X-API-Key、统一 ApiError（含中文 humanMessage）。 */

import { loadApiKey } from "./key";
import { humanizeStatus } from "@/lib/humanize";

export class ApiError extends Error {
  status: number;
  humanMessage: string;
  detail: string;

  constructor(status: number, humanMessage: string, detail = "") {
    super(humanMessage);
    this.name = "ApiError";
    this.status = status;
    this.humanMessage = humanMessage;
    this.detail = detail;
  }

  static async fromResponse(res: Response): Promise<ApiError> {
    let detail = "";
    try {
      const body = await res.json();
      detail = String(body?.detail ?? "");
    } catch {
      /* 非 JSON 响应体 */
    }
    // 后端 error 载荷已中文化的（409 busy 等）直接透传
    const message = res.status === 409 && detail ? detail : humanizeStatus(res.status, detail);
    return new ApiError(res.status, message, detail);
  }
}

function headers(extra?: Record<string, string>): Record<string, string> {
  const key = loadApiKey();
  return {
    "Content-Type": "application/json",
    ...(key ? { "X-API-Key": key } : {}),
    ...extra,
  };
}

/** GET + JSON 解析；非 2xx 抛 ApiError。 */
export async function apiGet<T>(path: string, signal?: AbortSignal): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, { headers: headers(), signal });
  } catch (err) {
    throw err instanceof DOMException && err.name === "AbortError" ? err : new ApiError(0, "网络错误");
  }
  if (!res.ok) throw await ApiError.fromResponse(res);
  return (await res.json()) as T;
}

/** POST/DELETE + JSON 体。 */
export async function apiSend<T>(
  method: "POST" | "DELETE",
  path: string,
  body?: unknown,
): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, {
      method,
      headers: headers(),
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, "网络错误");
  }
  if (!res.ok) throw await ApiError.fromResponse(res);
  return (await res.json()) as T;
}

/** multipart 文件上传：Content-Type（含 boundary）由浏览器生成，不可手工指定。 */
export async function apiUpload<T>(path: string, file: File): Promise<T> {
  const key = loadApiKey();
  const form = new FormData();
  form.append("file", file);
  let res: Response;
  try {
    res = await fetch(path, {
      method: "POST",
      headers: key ? { "X-API-Key": key } : {},
      body: form,
    });
  } catch {
    throw new ApiError(0, "网络错误");
  }
  if (!res.ok) throw await ApiError.fromResponse(res);
  return (await res.json()) as T;
}
