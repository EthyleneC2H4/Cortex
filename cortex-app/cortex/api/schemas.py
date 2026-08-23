"""API 请求/响应模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=2000)
    mode: str = Field(default="pipeline", pattern="^(pipeline|explore)$")


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
