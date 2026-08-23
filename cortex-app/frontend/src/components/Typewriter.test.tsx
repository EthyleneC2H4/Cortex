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
});
