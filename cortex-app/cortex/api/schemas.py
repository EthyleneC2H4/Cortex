"""API 请求/响应模型。"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=2000)
    mode: str = Field(default="pipeline", pattern="^(pipeline|explore)$")


class ChatRequest(BaseModel):
    """对话请求：session_id 为空 → 新建会话；reset=True → 清空编排器内部历史。"""

    message: str = Field(min_length=1, max_length=4000)
    session_id: Optional[str] = None
    reset: bool = False


class DocumentOut(BaseModel):
    doc_id: str
    title: str
    source: str = ""
    n_chunks: int
    created_at: str = ""


class DocumentList(BaseModel):
    documents: list[DocumentOut]
    total: int


class IngestTextRequest(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    content: str = Field(min_length=1)


class SessionCreateRequest(BaseModel):
    """创建会话（可选带初始历史）。"""

    history: list[dict] = Field(default_factory=list)


class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    turns: int
    history: list[dict]
