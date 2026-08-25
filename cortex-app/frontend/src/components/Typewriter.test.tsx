import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { Typewriter } from "./Typewriter";

/** jsdom 未实现 matchMedia，统一桩掉；reduced-motion 直接整段呈现。 */
function stubMatchMedia(matches: boolean) {
  vi.stubGlobal(
    "matchMedia",
    vi.fn().mockReturnValue({
      matches,
      addEventListener() {},
      removeEventListener() {},
    }),
  );
}

describe("Typewriter", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it("prefers-reduced-motion 时跳过动画直接显示全文，且挂载即回调 onDone 恰好一次", () => {
    stubMatchMedia(true);
    const onDone = vi.fn();
    render(<Typewriter text="完整呈现的一段话" onDone={onDone} />);
    expect(screen.getByText(/完整呈现的一段话/)).toBeTruthy();
    expect(onDone).toHaveBeenCalledTimes(1);
  });

  it("正常环境初始为空、点击后立即补全并回调 onDone 恰好一次", () => {
    stubMatchMedia(false);
    const onTick = vi.fn();
    const onDone = vi.fn();
    const { container } = render(<Typewriter text="慢慢打出来" onTick={onTick} onDone={onDone} />);
    // 动画尚未推进（fake 环境无真实 interval 推进前）点击应立即补全
    fireEvent.click(container.firstElementChild as Element);
    expect(screen.getByText(/慢慢打出来/)).toBeTruthy();
    // 补全时也回调一次 onTick
    expect(onTick).toHaveBeenCalled();
    expect(onDone).toHaveBeenCalledTimes(1);
  });

  it("自然播完回调 onDone 恰好一次", () => {
    stubMatchMedia(false);
    vi.useFakeTimers();
    const onDone = vi.fn();
    // step = max(2, ceil(10/400)) = 2 → 5 跳 × 16ms = 80ms
    render(<Typewriter text="一二三四五六七八九十" onDone={onDone} />);
    act(() => {
      vi.advanceTimersByTime(300);
    });
    expect(onDone).toHaveBeenCalledTimes(1);
  });

  it("动画中可键盘操作：Enter/Space 立即补全，且补全后摘除交互语义", () => {
    stubMatchMedia(false);
    const onDone = vi.fn();
    const { container } = render(<Typewriter text="键盘补全" onDone={onDone} />);
    const span = container.firstElementChild as HTMLElement;

    // 动画中：button 角色 + 可聚焦 + 有 aria 说明
    expect(span.getAttribute("role")).toBe("button");
    expect(span.getAttribute("tabindex")).toBe("0");
    expect(span.getAttribute("aria-label")).toContain("Enter");

    fireEvent.keyDown(span, { key: "Enter" });
    expect(screen.getByText(/键盘补全/)).toBeTruthy();
    expect(onDone).toHaveBeenCalledTimes(1);

    // 补全后：交互语义全部摘除，不再占据 tab 序
    expect(span.getAttribute("role")).toBeNull();
    expect(span.getAttribute("tabindex")).toBeNull();
    expect(span.getAttribute("aria-label")).toBeNull();
  });

  it("Space 同样触发补全且不滚动页面（preventDefault）", () => {
    stubMatchMedia(false);
    const { container } = render(<Typewriter text="空格补全" />);
    const span = container.firstElementChild as HTMLElement;
    // fireEvent 返回 false ⇔ preventDefault 被调用（页面不会滚动）
    const notPrevented = fireEvent.keyDown(span, { key: " ", cancelable: true });
    expect(notPrevented).toBe(false);
    expect(screen.getByText(/空格补全/)).toBeTruthy();
  });

  it("其他按键不触发补全", () => {
    stubMatchMedia(false);
    const onDone = vi.fn();
    const { container } = render(<Typewriter text="别的键无效" onDone={onDone} />);
    fireEvent.keyDown(container.firstElementChild as HTMLElement, { key: "Tab" });
    fireEvent.keyDown(container.firstElementChild as HTMLElement, { key: "a" });
    expect(onDone).not.toHaveBeenCalled();
  });
});
