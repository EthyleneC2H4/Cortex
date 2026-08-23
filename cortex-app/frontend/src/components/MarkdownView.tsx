import type { MouseEvent } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { copyText } from "@/lib/download";
import { rehypeCitationChips } from "@/lib/citation";

/**
 * 研报/对话共用的 Markdown 渲染。
 * 禁 raw HTML（XSS 面）；[doc_id/chunk_id] 引用标记经 rehype 插件
 * 变成 chip，点击复制（事件委托挂在容器上）。
 */
export function MarkdownView({
  text,
  onCitationCopy,
}: {
  text: string;
  onCitationCopy?: (id: string) => void;
}) {
  const handleClick = (e: MouseEvent<HTMLDivElement>) => {
    const el = (e.target as HTMLElement).closest("[data-citation]");
    if (!el) return;
    const id = el.getAttribute("data-citation");
    if (id) void copyText(id).then((ok) => ok && onCitationCopy?.(id));
  };

  return (
    <div className="text-[15px] leading-7 text-ink" onClick={handleClick}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeCitationChips]}
        components={{
          h1: (p) => (
            <h1 className="mt-7 mb-4 font-display text-2xl font-semibold tracking-tight text-ink first:mt-0" {...p} />
          ),
          h2: (p) => (
            <h2 className="mt-6 mb-3 font-display text-xl font-semibold tracking-tight text-ink" {...p} />
          ),
          h3: (p) => (
            <h3 className="mt-5 mb-2 font-display text-lg font-semibold text-ink" {...p} />
          ),
          p: (p) => <p className="my-3" {...p} />,
          ul: (p) => <ul className="my-3 list-disc space-y-1 pl-6" {...p} />,
          ol: (p) => <ol className="my-3 list-decimal space-y-1 pl-6" {...p} />,
          blockquote: (p) => (
            <blockquote className="my-4 border-l-3 border-accent bg-muted px-4 py-2 text-ink-2" {...p} />
          ),
          code: ({ className, children, ...rest }) => {
            const isBlock = /language-/.test(className ?? "");
            if (isBlock)
              return (
                <code className={`block overflow-x-auto rounded-md bg-muted p-4 font-mono text-[13px] leading-6 ${className ?? ""}`} {...rest}>
                  {children}
                </code>
              );
            return (
              <code className="rounded-xs bg-muted px-1.5 py-0.5 font-mono text-[13px] text-accent-strong dark:text-accent" {...rest}>
                {children}
              </code>
            );
          },
          pre: (p) => <pre className="my-4" {...p} />,
          table: (p) => (
            <div className="my-4 overflow-x-auto rounded-md border border-line">
              <table className="w-full border-collapse text-sm" {...p} />
            </div>
          ),
          th: (p) => <th className="border-b border-line bg-muted px-3 py-2 text-left font-medium" {...p} />,
          td: (p) => <td className="border-b border-line/60 px-3 py-2 align-top" {...p} />,
          a: (p) => <a className="text-accent-strong underline underline-offset-2 dark:text-accent" target="_blank" rel="noreferrer" {...p} />,
          hr: () => <hr className="my-6 border-line" />,
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}
