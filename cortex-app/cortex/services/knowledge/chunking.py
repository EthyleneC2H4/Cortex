"""Markdown 感知切块：512/64 tokens，标题栈→heading_path，代码块/表格原子。

每块正文前缀 heading_path（提升向量与 BM25 双侧可检索性）。
token 计数用 tiktoken（cl100k_base），不可用时 len//3 兜底。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Optional

_SENT_SPLIT = re.compile(r"(?<=[。！？!?\.])\s*")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE = re.compile(r"^```")


@dataclass
class Chunk:
    idx: int
    text: str
    heading_path: str


def default_token_counter() -> Callable[[str], int]:
    """tiktoken cl100k_base；任何失败降级 len//3。"""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")

        return lambda s: len(enc.encode(s))
    except Exception:
        return lambda s: max(1, len(s) // 3)


def _split_sentences(text: str) -> List[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def chunk_markdown(
    md: str,
    size: int = 512,
    overlap: int = 64,
    counter: Optional[Callable[[str], int]] = None,
) -> List[Chunk]:
    """切块主入口。返回顺序 Chunk 列表（idx 从 0）。"""
    counter = counter or default_token_counter()
    chunks: List[Chunk] = []
    heading_stack: list[tuple[int, str]] = []
    buf: list[str] = []  # 当前段落聚合缓冲
    buf_heading = ""

    def flush():
        nonlocal buf, buf_heading
        if not buf:
            return
        text = "\n\n".join(buf).strip()
        buf = []
        if not text:
            return
        prefix = f"[{buf_heading}]\n" if buf_heading else ""
        pieces = _pack(text, size - (counter(prefix) if prefix else 0), overlap, counter)
        for piece in pieces:
            chunks.append(Chunk(idx=len(chunks), text=prefix + piece, heading_path=buf_heading))

    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            i += 1
            continue

        # 围栏代码块：整块原子（超长则按行硬切）
        if _FENCE.match(line):
            block = [line]
            i += 1
            while i < len(lines):
                block.append(lines[i])
                if _FENCE.match(lines[i]):
                    i += 1
                    break
                i += 1
            buf_heading = " / ".join(t for _, t in heading_stack) or ""
            text = "\n".join(block)
            if counter(text) <= size:
                buf.append(text)
            else:
                flush()
                for piece in _hard_wrap(text, size, counter):
                    chunks.append(
                        Chunk(idx=len(chunks), text=f"[{buf_heading}]\n{piece}" if buf_heading else piece,
                              heading_path=buf_heading)
                    )
            continue

        # 表格：连续 | 开头行为一个原子块
        if line.lstrip().startswith("|"):
            block = [line]
            i += 1
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append(lines[i])
                i += 1
            flush()
            buf_heading = " / ".join(t for _, t in heading_stack) or ""
            text = "\n".join(block)
            buf.append(text)
            flush()
            continue

        # 空行 → 段落边界
        if not line.strip():
            flush()
            i += 1
            continue

        cur_heading = " / ".join(t for _, t in heading_stack) or ""
        if cur_heading != buf_heading:
            flush()
            buf_heading = cur_heading
        buf.append(line.rstrip())
        i += 1
    flush()
    return chunks


# --------------------------------------------------------------------- #
def _pack(text: str, size: int, overlap: int, counter: Callable[[str], int]) -> List[str]:
    """句级贪心装箱 + 尾部重叠携带。"""
    sentences = _split_sentences(text)
    out: List[str] = []
    cur: List[str] = []
    cur_tok = 0
    for sent in sentences:
        n = counter(sent)
        if n > size:  # 单句超限 → 行硬切
            if cur:
                out.append(" ".join(cur))
                carry = _tail(cur, overlap, counter)
                cur, cur_tok = ([] if not carry else [carry]), counter(carry) if carry else 0
            for piece in _hard_wrap(sent, size, counter):
                out.append(piece)
            continue
        if cur_tok + n > size and cur:
            out.append(" ".join(cur))
            carry = _tail(cur, overlap, counter)
            cur = [carry] if carry else []
            cur_tok = counter(carry) if carry else 0
        cur.append(sent)
        cur_tok += n
    if cur:
        out.append(" ".join(cur))
    return out


def _tail(pieces: List[str], overlap: int, counter: Callable[[str], int]) -> str:
    """取末尾约 overlap token 的句子作为下一块开头。"""
    acc: List[str] = []
    total = 0
    for sent in reversed(pieces):
        n = counter(sent)
        if total + n > overlap and acc:
            break
        acc.insert(0, sent)
        total += n
    return " ".join(acc)


def _hard_wrap(text: str, size: int, counter: Callable[[str], int]) -> List[str]:
    """按字符行硬切超长原子块。"""
    if counter(text) <= size:
        return [text]
    lines, out, cur = text.splitlines(), [], []
    cur_tok = 0
    for ln in lines:
        n = max(1, counter(ln))
        if cur_tok + n > size and cur:
            out.append("\n".join(cur))
            cur, cur_tok = [], 0
        cur.append(ln)
        cur_tok += n
    if cur:
        out.append("\n".join(cur))
    return out
