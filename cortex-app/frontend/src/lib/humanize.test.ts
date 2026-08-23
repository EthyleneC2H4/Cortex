import { describe, expect, it } from "vitest";

import { humanizeNetworkError, humanizeStatus } from "./humanize";

describe("humanizeStatus", () => {
  it("401 → 提示配置 API Key", () => {
    expect(humanizeStatus(401, "")).toContain("API Key");
  });
  it("404 → 资源不存在", () => {
    expect(humanizeStatus(404, "")).toBe("资源不存在");
  });
  it("429 → 配额受限", () => {
    expect(humanizeStatus(429, "")).toContain("配额");
  });
  it("5xx → 服务内部错误", () => {
    expect(humanizeStatus(502, "")).toContain("服务内部错误");
  });
  it("422 透传后端中文 detail", () => {
    expect(humanizeStatus(422, "摄取失败")).toBe("摄取失败");
  });
});

describe("humanizeNetworkError", () => {
  it("AbortError → 说明服务端任务继续执行", () => {
    const err = new DOMException("aborted", "AbortError");
    const msg = humanizeNetworkError(err);
    expect(msg).toContain("已停止");
    expect(msg).toContain("落盘");
  });
  it("普通网络异常 → 指引启动后端", () => {
    expect(humanizeNetworkError(new TypeError("Failed to fetch"))).toContain(
      "python -m cortex.api.server",
    );
  });
});
