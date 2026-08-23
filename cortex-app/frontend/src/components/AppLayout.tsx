import { useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { useTheme } from "@/context/ThemeContext";

/** 线性描边图标（stroke=currentColor，随文字色）。 */
function Icon({ d, className = "" }: { d: string; className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={`size-5 shrink-0 ${className}`}
      aria-hidden="true"
    >
      <path d={d} />
    </svg>
  );
}

const ICONS = {
  research:
    "M9 3h6M10 3v5.5L4.8 18a2 2 0 0 0 1.8 3h10.8a2 2 0 0 0 1.8-3L14 8.5V3M7 15h10",
  chat: "M21 12a8 8 0 0 1-8 8H4l2.2-3.3A8 8 0 1 1 21 12z",
  knowledge: "M4 19.5A2.5 2.5 0 0 1 6.5 17H20M4 19.5A2.5 2.5 0 0 0 6.5 22H20V2H6.5A2.5 2.5 0 0 0 4 4.5v15zM8 7h8M8 11h5",
  status: "M3 12h4l3-8 4 16 3-8h4",
} as const;

const NAV_ITEMS = [
  { to: "/research", label: "研究工作台", icon: ICONS.research },
  { to: "/chat", label: "智能对话", icon: ICONS.chat },
  { to: "/knowledge", label: "知识库", icon: ICONS.knowledge },
  { to: "/status", label: "系统状态", icon: ICONS.status },
];

function SunIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" className="size-5" aria-hidden="true">
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2m0 16v2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4M2 12h2m16 0h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className="size-5" aria-hidden="true">
      <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
    </svg>
  );
}

export function AppLayout() {
  const { theme, toggle } = useTheme();
  const [drawerOpen, setDrawerOpen] = useState(false);

  const nav = (
    <nav aria-label="主导航" className="flex flex-col gap-1">
      {NAV_ITEMS.map((item) => (
        <NavLink
          key={item.to}
          to={item.to}
          onClick={() => setDrawerOpen(false)}
          className={({ isActive }) =>
            `flex items-center gap-3 rounded-md px-3 py-2.5 text-[15px] transition-colors ${
              isActive
                ? "bg-muted font-medium text-accent-strong dark:text-accent"
                : "text-ink-2 hover:bg-muted hover:text-ink"
            }`
          }
        >
          <Icon d={item.icon} />
          {item.label}
        </NavLink>
      ))}
    </nav>
  );

  const themeToggle = (
    <button
      type="button"
      onClick={toggle}
      aria-label={theme === "dark" ? "切换到浅色模式" : "切换到深色模式"}
      className="flex items-center gap-3 rounded-md px-3 py-2.5 text-[15px] text-ink-2 transition-colors hover:bg-muted hover:text-ink"
    >
      {theme === "dark" ? <SunIcon /> : <MoonIcon />}
      {theme === "dark" ? "浅色模式" : "深色模式"}
    </button>
  );

  return (
    <div className="min-h-dvh bg-page text-ink">
      {/* 跳转链接：键盘用户直达主内容 */}
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:rounded-md focus:bg-surface focus:px-4 focus:py-2 focus:text-sm focus:text-accent-strong"
      >
        跳到主内容
      </a>

      {/* 桌面侧栏 */}
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-64 flex-col border-r border-line bg-surface px-4 py-6 md:flex">
        <div className="mb-8 flex items-center gap-2.5 px-2">
          <img src="/favicon.svg" alt="" className="size-8 rounded-md" />
          <span className="font-display text-xl tracking-wide text-ink">Cortex</span>
        </div>
        {nav}
        <div className="mt-auto">{themeToggle}</div>
      </aside>

      {/* 移动端顶栏 */}
      <header className="sticky top-0 z-40 flex items-center gap-3 border-b border-line bg-surface/90 px-4 py-3 backdrop-blur md:hidden">
        <button
          type="button"
          aria-label={drawerOpen ? "关闭导航菜单" : "打开导航菜单"}
          aria-expanded={drawerOpen}
          onClick={() => setDrawerOpen((v) => !v)}
          className="rounded-md p-1.5 text-ink transition-colors hover:bg-muted"
        >
          <Icon d={drawerOpen ? "M6 6l12 12M18 6L6 18" : "M4 7h16M4 12h16M4 17h16"} />
        </button>
        <span className="font-display text-lg">Cortex</span>
      </header>

      {/* 移动端抽屉 */}
      {drawerOpen && (
        <>
          <div
            className="fixed inset-0 z-40 bg-ink/25 md:hidden"
            onClick={() => setDrawerOpen(false)}
            role="presentation"
          />
          <aside className="fixed inset-y-0 left-0 z-50 flex w-72 flex-col border-r border-line bg-surface px-4 py-6 md:hidden">
            <div className="mb-8 flex items-center gap-2.5 px-2">
              <img src="/favicon.svg" alt="" className="size-8 rounded-md" />
              <span className="font-display text-xl">Cortex</span>
            </div>
            {nav}
            <div className="mt-auto">{themeToggle}</div>
          </aside>
        </>
      )}

      <main id="main" className="px-4 py-6 md:ml-64 md:px-10 md:py-10">
        <div className="mx-auto max-w-4xl">
          <Outlet />
        </div>
      </main>
    </div>
  );
}

export { Icon };
