import { describe, expect, it } from "vitest";

import { rehypeCitationChips, splitCitations } from "./citation";

describe("splitCitations", () => {
  it("切出十六进制 doc 段引用（真实报告形态 [hex/int]）", () => {
    const parts = splitCitations("前文[fda79080dd090ee5/3]后文");
    expect(parts).toEqual([
      { kind: "text", value: "前文" },
      { kind: "cite", id: "fda79080dd090ee5/3" },
      { kind: "text", value: "后文" },
    ]);
  });

  it("支持 [doc/doc:idx] 完整形态与中文夹注", () => {
    const parts = splitCitations("结论见[ab12cd34/ab12cd34:2]；另据[ab12cd34/7]。");
    expect(parts.filter((p) => p.kind === "cite").map((p) => (p as { id: string }).id)).toEqual([
      "ab12cd34/ab12cd34:2",
      "ab12cd34/7",
    ]);
  });

  it("不误伤普通方括号（[TODO] [1] [note]）", () => {
    const text = "见 [TODO] 与 [1] 和 [note]，无引用。";
    expect(splitCitations(text)).toEqual([{ kind: "text", value: text }]);
  });

  it("无标记文本原样返回单片段", () => {
    expect(splitCitations("纯文本")).toEqual([{ kind: "text", value: "纯文本" }]);
  });
});

describe("rehypeCitationChips（hast 变换）", () => {
  const plugin = rehypeCitationChips();

  it("把 text 节点替换为 chip 元素序列", () => {
    const tree = {
      type: "root",
      children: [
        {
          type: "element",
          tagName: "p",
          children: [{ type: "text", value: "依据[ab12cd34/3]所述" }],
        },
      ],
    };
    plugin(tree as never);
    const p = tree.children[0] as { children: Array<Record<string, unknown>> };
    expect(p.children).toHaveLength(3);
    const chip = p.children[1] as { tagName: string; properties: Record<string, unknown> };
    expect(chip.tagName).toBe("button");
    expect(chip.properties["data-citation"]).toBe("ab12cd34/3");
    expect(chip.properties.className).toContain("citation-chip");
  });

  it("不含引用的树保持原样", () => {
    const tree = {
      type: "root",
      children: [{ type: "text", value: "[1] 普通文本" }],
    };
    plugin(tree as never);
    expect((tree.children[0] as { value: string }).value).toBe("[1] 普通文本");
  });
});
