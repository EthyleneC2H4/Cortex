import { useEffect, useRef, useState } from "react";

const prefersReducedMotion = () =>
  typeof window !== "undefined" &&
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/**
 * 打字机逐字显示：弥补后端整轮返回的感知延迟。
 * 尊重 prefers-reduced-motion —— 直接整段呈现；动画中点击则立即补全。
 * 全部显示完毕（含跳过/降级路径）后回调 onDone 恰好一次，供父组件切换为正式渲染。
 */
export function Typewriter({
  text,
  onTick,
  onDone,
}: {
  text: string;
  onTick?: () => void;
  onDone?: () => void;
}) {
  const [shown, setShown] = useState(() => (prefersReducedMotion() ? text : ""));
  const [done, setDone] = useState(() => prefersReducedMotion());
  // 回调走 ref：动画 effect 不因父组件重渲染（onTick 闭包变化）而重启
  const onTickRef = useRef(onTick);
  const onDoneRef = useRef(onDone);
  onTickRef.current = onTick;
  onDoneRef.current = onDone;
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    if (prefersReducedMotion()) {
      setShown(text);
      setDone(true);
      onDoneRef.current?.();
      return;
    }
    // 每跳显示多个字符，长文本不至于等太久
    const step = Math.max(2, Math.ceil(text.length / 400));
    let i = 0;
    setShown("");
    setDone(false);
    const id = window.setInterval(() => {
      i = Math.min(text.length, i + step);
      setShown(text.slice(0, i));
      onTickRef.current?.();
      if (i >= text.length) {
        // 本轮回调内即清除，后续 tick 不会再触发（onDone 恰好一次）
        window.clearInterval(id);
        setDone(true);
        onDoneRef.current?.();
      }
    }, 16);
    timerRef.current = id;
    return () => {
      window.clearInterval(id);
      timerRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text]);

  const finishNow = () => {
    if (!done) {
      if (timerRef.current !== null) window.clearInterval(timerRef.current);
      setShown(text);
      setDone(true);
      onTickRef.current?.();
      onDoneRef.current?.();
    }
  };

  // 键盘可达性：动画中是可操作控件（Enter/Space 立即补全），播完即摘除交互语义
  const interactive = !done;

  return (
    <span
      onClick={interactive ? finishNow : undefined}
      role={interactive ? "button" : undefined}
      tabIndex={interactive ? 0 : undefined}
      aria-label={interactive ? "回答生成中，按 Enter 立即显示全文" : undefined}
      onKeyDown={
        interactive
          ? (e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault(); // 防空格触发页面滚动
                finishNow();
              }
            }
          : undefined
      }
      className={interactive ? "cursor-pointer rounded-sm focus-visible:outline focus-visible:outline-accent" : ""}
    >
      {shown}
      {interactive && <span className="animate-pulse text-accent">▍</span>}
    </span>
  );
}
