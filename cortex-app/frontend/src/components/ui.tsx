import type { ReactNode } from "react";

/** 细小通用件：Spinner / EmptyState / ErrorBanner。 */

export function Spinner({ className = "" }: { className?: string }) {
  return (
    <svg
      className={`animate-spin text-accent ${className}`}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" opacity="0.2" />
      <path d="M22 12a10 10 0 0 0-10-10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}

export function EmptyState({
  icon,
  title,
  hint,
}: {
  icon?: ReactNode;
  title: string;
  hint?: string;
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-line bg-surface px-6 py-14 text-center">
      {icon && <div className="text-ink-2/60">{icon}</div>}
      <p className="font-display text-base text-ink">{title}</p>
      {hint && <p className="max-w-sm text-sm text-ink-2">{hint}</p>}
    </div>
  );
}

export function ErrorBanner({
  message,
  detail,
  onRetry,
}: {
  message: string;
  detail?: string;
  onRetry?: () => void;
}) {
  return (
    <div
      role="alert"
      className="rounded-md border border-danger/40 bg-danger/8 px-4 py-3 text-sm text-danger"
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <p>{message}</p>
          {detail && <p className="mt-1 text-xs break-all text-ink-2">{detail}</p>}
        </div>
        {onRetry && (
          <button
            type="button"
            onClick={onRetry}
            className="shrink-0 rounded-sm border border-danger/50 px-3 py-1 text-xs font-medium transition-colors hover:bg-danger/15"
          >
            重试
          </button>
        )}
      </div>
    </div>
  );
}
