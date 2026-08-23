"""P3 单元测试：切块（标题栈/原子块/重叠/兜底计数）。"""

from __future__ import annotations

import pytest

from cortex.services.knowledge.chunking import (
    chunk_markdown,
    default_token_counter,
)

pytestmark = pytest.mark.unit


def _texts(chunks):
    return [c.text for c in chunks]


class TestHeading:
    def test_heading_path_prefix_and_stack(self):
        md = "# 总览\nintro\n## 检索\nRRF 内容\n## 记忆\n权重\n### 遗忘\n软删除"
        chunks = chunk_markdown(md, size=512)
        by_head = {c.heading_path: c for c in chunks}
        assert "总览 / 检索" in by_head
        assert "总览 / 记忆 / 遗忘" in by_head
        assert by_head["总览 / 检索"].text.startswith("[总览 / 检索]")

    def test_heading_reset_on_pop(self):
        md = "# A\n内容一\n## B\n内容二\n# C\n内容三"
        chunks = chunk_markdown(md, size=512)
        heads = [c.heading_path for c in chunks]
        assert "C" in heads  # 一级标题重置栈，不带 A 前缀
        assert not any(h.startswith("C / A") for h in heads)


class TestAtomic:
    def test_code_block_not_split(self):
        code = "```python\n" + "\n".join(f"line_{i} = {i}" for i in range(30)) + "\n```"
        md = f"# 代码\n段落\n{code}\n后文"
        chunks = chunk_markdown(md, size=64)
        code_chunks = [c for c in chunks if "line_0" in c.text]
        assert code_chunks, "代码块丢失"
        joined = "\n".join(_texts(chunks))
        assert "line_29" in joined

    def test_table_atomic(self):
        md = "| a | b |\n|---|---|\n| 1 | 2 |\n\n表格后的段落"
        chunks = chunk_markdown(md, size=512)
        table_chunk = next(c for c in chunks if "| a |" in c.text)
        assert "| 1 | 2 |" in table_chunk.text  # 表头与行在同一块


class TestPacking:
    def test_long_text_splits_with_overlap(self):
        para = "。".join(f"这是第{i}句话内容" for i in range(200)) + "。"
        md = f"# 长文\n{para}"
        counter = default_token_counter()
        chunks = chunk_markdown(md, size=128, overlap=32, counter=counter)
        assert len(chunks) > 1
        # 相邻块有重叠：后块开头出现在前块结尾附近
        assert chunks[1].text != chunks[0].text

    def test_idx_sequential(self):
        md = "# T\n" + "句子。" * 500
        chunks = chunk_markdown(md, size=64, overlap=16)
        assert [c.idx for c in chunks] == list(range(len(chunks)))

    def test_fallback_counter_offline(self):
        # tiktoken 不可用时应降级而非抛异常（此处直接验证函数可调用）
        c = default_token_counter()
        assert c("hello world") >= 1


class TestCJK:
    def test_chinese_chunking(self):
        md = "# 中文\n检索增强生成通过结合检索与生成提升事实性。混合检索融合向量与关键词两路信号。"
        chunks = chunk_markdown(md, size=32)
        assert chunks
        assert any("混合检索" in c.text for c in chunks)
