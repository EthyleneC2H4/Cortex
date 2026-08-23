"""NoteTool — 个人笔记工具（P2，cortex.md §8.4 结构化笔记锚点）。

研究计划（RESEARCH_PLAN）/结论索引（FINDINGS）/未解问题（OPEN_QUESTIONS）等
以普通笔记形式持久化，作为长程任务的"外部记忆锚"。存储为纯文件：
每条笔记一个 Markdown 文件 + ``<file>.meta.json`` 版本元数据，支持乐观锁更新。

扩展点：继承框架 ``Tool``（expandable），注册后按方法名展开为
``note_create / note_read / note_update / note_search / note_list / note_delete``
六个子工具；所有失败返回结构化 ``ToolResponse.error``，绝不抛异常。
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

from hello_agents.tools import Tool, ToolResponse, tool_action
from hello_agents.tools.errors import ToolErrorCode

_META_SUFFIX = ".meta.json"


def _slugify(title: str) -> str:
    """标题 → 安全文件名主干（保留 CJK/字母数字）。"""
    title = unicodedata.normalize("NFKC", title).strip()
    slug = re.sub(r"[^\w一-鿿-]+", "-", title, flags=re.UNICODE)
    slug = re.sub(r"-{2,}", "-", slug).strip("-").lower()
    return (slug or "untitled")[:80]


class NoteTool(Tool):
    """结构化笔记的创建/读取/更新/检索；乐观锁防并发覆盖。"""

    def __init__(self, notes_dir: Path | str) -> None:
        self.notes_dir = Path(notes_dir)
        self.notes_dir.mkdir(parents=True, exist_ok=True)
        self._registry = None
        super().__init__(
            name="note",
            description="个人笔记管理：创建/读取/更新/检索研究计划与结论笔记",
            expandable=True,
        )

    def bind_registry(self, registry) -> None:
        """注入注册表后，read 会同步版本号到注册表读缓存（乐观锁辅助）。"""
        self._registry = registry

    # ------------------------------------------------------------------ #
    # 内部存储 helpers                                                     #
    # ------------------------------------------------------------------ #
    def _paths(self, title: str) -> tuple[Path, Path]:
        stem = _slugify(title)
        return self.notes_dir / f"{stem}.md", self.notes_dir / f"{stem}{_META_SUFFIX}"

    def _find_by_title(self, title: str) -> Optional[tuple[Path, Path, dict]]:
        """按精确标题查找（meta.title 比对）；找不到返回 None。"""
        direct_md, direct_meta = self._paths(title)
        for candidate_meta in sorted(self.notes_dir.glob(f"*{_META_SUFFIX}")):
            try:
                meta = json.loads(candidate_meta.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if meta.get("title") == title:
                md = candidate_meta.with_suffix("").with_suffix(".md")
                return md, candidate_meta, meta
        if direct_meta.exists():
            try:
                meta = json.loads(direct_meta.read_text(encoding="utf-8"))
                return direct_md, direct_meta, meta
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def _write_note(self, title: str, content: str, meta: dict, md_path: Path, meta_path: Path) -> None:
        md_path.write_text(content, encoding="utf-8")
        tmp = meta_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(meta_path)

    def _all_metas(self) -> List[dict]:
        out = []
        for meta_path in sorted(self.notes_dir.glob(f"*{_META_SUFFIX}")):
            try:
                out.append(json.loads(meta_path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                continue
        return out

    # ------------------------------------------------------------------ #
    # 展开动作（方法名决定子工具名：note_create 等）                        #
    # ------------------------------------------------------------------ #
    @tool_action(description="创建一条新笔记；标题重复时报 CONFLICT")
    def create(self, title: str, content: str) -> ToolResponse:
        """创建新笔记。

        Args:
            title: 笔记标题（唯一键）
            content: Markdown 正文
        """
        if not title.strip():
            return ToolResponse.error(ToolErrorCode.INVALID_PARAM, "标题不能为空")
        found = self._find_by_title(title)
        if found:
            return ToolResponse.error(ToolErrorCode.CONFLICT, f"笔记已存在: {title}")
        md_path, meta_path = self._paths(title)
        now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        meta = {"title": title.strip(), "version": 1, "created_at": now, "updated_at": now}
        self._write_note(title, content, meta, md_path, meta_path)
        return ToolResponse.success(
            text=f"已创建笔记「{title}」（v1）",
            data={"title": title, "version": 1},
        )

    @tool_action(description="读取笔记全文；data.version 为当前版本号（更新时需回传）")
    def read(self, title: str) -> ToolResponse:
        """读取笔记内容。

        Args:
            title: 笔记标题
        """
        found = self._find_by_title(title)
        if not found or not found[0].exists():
            return ToolResponse.error(ToolErrorCode.NOT_FOUND, f"笔记不存在: {title}")
        md_path, meta_path, meta = found
        content = md_path.read_text(encoding="utf-8")
        if self._registry is not None:
            try:
                self._registry.cache_read_metadata(str(md_path), {"version": meta.get("version")})
            except Exception:
                pass
        return ToolResponse.success(
            text=f"[{meta['title']} v{meta['version']}]\n{content}",
            data={"title": meta["title"], "version": meta["version"], "content": content},
        )

    @tool_action(description="按版本号乐观锁更新笔记；版本不匹配报 CONFLICT")
    def update(self, title: str, content: str, expected_version: int) -> ToolResponse:
        """乐观锁更新。

        Args:
            title: 笔记标题
            content: 新的 Markdown 正文
            expected_version: 调用方持有的版本号（read 时获得）
        """
        found = self._find_by_title(title)
        if not found or not found[0].exists():
            return ToolResponse.error(ToolErrorCode.NOT_FOUND, f"笔记不存在: {title}")
        md_path, meta_path, meta = found
        current = int(meta.get("version", 1))
        if int(expected_version) != current:
            return ToolResponse.error(
                ToolErrorCode.CONFLICT,
                f"版本冲突: 当前 v{current}，调用方持有 v{expected_version}",
            )
        meta["version"] = current + 1
        meta["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self._write_note(title, content, meta, md_path, meta_path)
        return ToolResponse.success(
            text=f"已更新笔记「{title}」至 v{meta['version']}",
            data={"title": title, "version": meta["version"]},
        )

    @tool_action(description="按关键词检索笔记标题与正文，返回匹配列表")
    def search(self, query: str, limit: int = 10) -> ToolResponse:
        """关键词检索笔记。

        Args:
            query: 关键词（大小写不敏感子串匹配）
            limit: 最多返回条数
        """
        q = query.lower().strip()
        if not q:
            return ToolResponse.error(ToolErrorCode.INVALID_PARAM, "查询词不能为空")
        hits = []
        for meta in self._all_metas():
            md_path, _ = self._paths(meta["title"])
            try:
                body = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
            except OSError:
                body = ""
            hay = f"{meta['title']}\n{body}".lower()
            if q in hay:
                pos = body.lower().find(q)
                snippet = body[max(0, pos - 20) : pos + 60].replace("\n", " ") if pos >= 0 else ""
                hits.append({"title": meta["title"], "version": meta.get("version", 1), "snippet": snippet})
            if len(hits) >= int(limit):
                break
        if not hits:
            return ToolResponse.success(text=f"没有匹配「{query}」的笔记", data={"hits": []})
        lines = [f"- {h['title']} (v{h['version']}): {h['snippet']}…" for h in hits]
        return ToolResponse.success(text="\n".join(lines), data={"hits": hits})

    @tool_action(description="列出全部笔记（标题+版本+更新时间）")
    def list(self) -> ToolResponse:
        """列出全部笔记。"""
        metas = self._all_metas()
        if not metas:
            return ToolResponse.success(text="（暂无笔记）", data={"notes": []})
        lines = [
            f"- {m['title']} (v{m.get('version', 1)}, 更新于 {m.get('updated_at', '?')})"
            for m in metas
        ]
        return ToolResponse.success(text="\n".join(lines), data={"notes": metas})

    @tool_action(description="删除笔记及其元数据")
    def delete(self, title: str) -> ToolResponse:
        """删除笔记。

        Args:
            title: 笔记标题
        """
        found = self._find_by_title(title)
        if not found:
            return ToolResponse.error(ToolErrorCode.NOT_FOUND, f"笔记不存在: {title}")
        md_path, meta_path, _ = found
        for p in (md_path, meta_path):
            if p.exists():
                p.unlink()
        return ToolResponse.success(text=f"已删除笔记「{title}」", data={"title": title})

    # Tool 抽象接口（expandable 注册路径不会用到，但保持契约完整）
    def get_parameters(self) -> List[Any]:  # pragma: no cover
        from hello_agents.tools import ToolParameter

        return [ToolParameter(name="action", type="string", description="未展开直调时的动作名")]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:  # pragma: no cover
        return ToolResponse.error(
            ToolErrorCode.INVALID_PARAM, "NoteTool 应通过展开后的 note_* 子工具调用"
        )
