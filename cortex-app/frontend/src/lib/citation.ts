/**
 * 引用标记识别与 rehype 插件：把报告正文中的 [doc_id/chunk_id]
 * 渲染为可点击复制的 chip。
 *
 * 真实标记形态（writer 提示词约定 + 实测）：[fda79080dd090ee5/3]、
 * [doc_id/doc_id:3]；正则按「十六进制 doc 段 / 数字冒号段」收敛，
 * 避免 [TODO] / [1] 这类普通方括号被误伤。
 */

export const CITATION_RE = /\[([a-f0-9]{8,32})\/([a-zA-Z0-9_:-]{1,64})\]/g;

export type CitationPart =
  | { kind: "text"; value: string }
  | { kind: "cite"; id: string };

/** 把一段文本切分为 text/cite 片段序列。 */
export function splitCitations(text: string): CitationPart[] {
  const parts: CitationPart[] = [];
  let last = 0;
  CITATION_RE.lastIndex = 0;
  for (let m = CITATION_RE.exec(text); m !== null; m = CITATION_RE.exec(text)) {
    if (m.index > last) parts.push({ kind: "text", value: text.slice(last, m.index) });
    parts.push({ kind: "cite", id: m[0].slice(1, -1) });
    last = m.index + m[0].length;
  }
  if (last < text.length) parts.push({ kind: "text", value: text.slice(last) });
  return parts;
}

// ---- rehype 插件（hast 树变换） --------------------------------- //

interface HastNode {
  type: string;
  value?: string;
  tagName?: string;
  properties?: Record<string, unknown>;
  children?: HastNode[];
}

function citationChip(id: string): HastNode {
  return {
    type: "element",
    tagName: "button",
    properties: {
      type: "button",
      className: ["citation-chip"],
      "data-citation": id,
      title: `引用 ${id}（点击复制）`,
      ariaLabel: `复制引用标识 ${id}`,
    },
    children: [{ type: "text", value: id }],
  };
}

/** 就地遍历 hast，把含引用标记的 text 节点替换为 chip 序列。 */
export function rehypeCitationChips() {
  return (tree: HastNode) => walk(tree);

  function walk(node: HastNode): void {
    if (!node.children) return;
    const out: HastNode[] = [];
    let changed = false;
    for (const child of node.children) {
      if (child.type === "text" && child.value && child.value.includes("[")) {
        const parts = splitCitations(child.value);
        if (parts.some((p) => p.kind === "cite")) {
          changed = true;
          for (const part of parts) {
            out.push(part.kind === "cite" ? citationChip(part.id) : { type: "text", value: part.value });
          }
          continue;
        }
      }
      if (child.children) walk(child);
      out.push(child);
    }
    if (changed) node.children = out;
  }
}
