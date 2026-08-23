import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import type { IncomingMessage } from "node:http";
import { fileURLToPath, URL } from "node:url";

// 文档级导航（F5 / 直开 / 分享的 /research、/chat 深链接，Accept 含 text/html）
// 交还 Vite SPA 中间件——否则会被代理到 FastAPI 拿到生产 dist 或 500。
// fetch/XHR/SSE 的 Accept 默认为 */*，照常走代理。必须返回字符串才生效。
const bypassHtml = (req: IncomingMessage) =>
  req.method === "GET" && String(req.headers.accept ?? "").includes("text/html")
    ? "/index.html"
    : undefined;

// 开发态：Vite 托管页面（:5173），API 路径代理到 FastAPI（:8000），零 CORS。
// 生产态：npm run build 后由 FastAPI 直接托管 dist/（见 cortex/api/server.py）。
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    proxy: {
      "^/(research|ingest|ingest-text|documents|chat|session|health)": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        bypass: bypassHtml,
      },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/setupTests.ts",
    css: false,
  },
});
