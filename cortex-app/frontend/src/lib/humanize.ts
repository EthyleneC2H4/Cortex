/** 错误人性化：把底层异常翻译成可操作的中文短句（与后端 _humanize_error 对齐）。 */

export function humanizeNetworkError(err: unknown): string {
  if (err instanceof DOMException && err.name === "AbortError") {
    return "已停止（注意：服务端任务会继续执行，结果将落盘保存）";
  }
  return "无法连接服务，请确认后端已启动（python -m cortex.api.server）";
}

export function humanizeStatus(status: number, detail: string): string {
  switch (status) {
    case 401:
      return "需要 API Key 或密钥无效";
    case 404:
      return "资源不存在";
    case 422:
      return detail || "请求参数不合法";
    case 429:
      return "模型配额受限，请稍后重试";
    default:
      if (status >= 500) return "服务内部错误，请稍后重试";
      return detail || `请求失败（${status}）`;
  }
}
