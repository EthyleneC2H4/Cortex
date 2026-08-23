"""服务层 LLM 调用的限流/瞬断重试。

OpenRouter 免费档常见 429 与偶发 5xx/超时；本模块只包装**服务层**直接发起的
调用（MQE 改写、HyDE、记忆重要性判定、judge 等）。框架 agent 内部的工具循环
不做包裹（步数低、由上游重试兜底）。
"""

from __future__ import annotations

import random
import re
import time
from typing import Callable, TypeVar

T = TypeVar("T")

_RETRYABLE_MARKERS = (
    "rate limit",
    "429",
    "timeout",
    "timed out",
    "connection",
    "temporarily",
    "overloaded",
    "502",
    "503",
    "504",
    "reset by peer",
)


class RetryExhaustedError(RuntimeError):
    """重试次数用尽仍失败。"""


def is_retryable(exc: BaseException) -> bool:
    try:
        import httpx

        if isinstance(exc, httpx.TransportError):
            return True
    except ImportError:  # pragma: no cover
        pass
    text = str(exc).lower()
    return any(marker in text for marker in _RETRYABLE_MARKERS)


def _retry_after_seconds(exc: BaseException) -> float | None:
    m = re.search(r"retry[-_ ]?after[:\s]*(\d+(?:\.\d+)?)", str(exc), re.IGNORECASE)
    return float(m.group(1)) if m else None


def _with_retries(
    fn: Callable[[], T],
    *,
    retries: int,
    base_delay: float,
    max_delay: float,
    sleep: Callable[[float], None],
) -> T:
    last_exc: BaseException | None = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt >= retries or not is_retryable(exc):
                raise
            delay = _retry_after_seconds(exc)
            if delay is None:
                delay = min(max_delay, base_delay * (2**attempt))
            sleep(delay + random.uniform(0, 0.5))
    raise RetryExhaustedError(str(last_exc)) from last_exc  # pragma: no cover


def resilient_invoke(
    llm,
    messages: list[dict],
    *,
    retries: int = 4,
    base_delay: float = 1.5,
    max_delay: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
    **kwargs,
):
    """llm.invoke 的退避重试版。"""
    return _with_retries(
        lambda: llm.invoke(messages, **kwargs),
        retries=retries,
        base_delay=base_delay,
        max_delay=max_delay,
        sleep=sleep,
    )


def resilient_invoke_with_tools(
    llm,
    messages: list[dict],
    tools: list[dict],
    *,
    tool_choice: str | dict = "auto",
    retries: int = 4,
    base_delay: float = 1.5,
    max_delay: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
    **kwargs,
):
    """llm.invoke_with_tools 的退避重试版。"""
    return _with_retries(
        lambda: llm.invoke_with_tools(messages, tools, tool_choice=tool_choice, **kwargs),
        retries=retries,
        base_delay=base_delay,
        max_delay=max_delay,
        sleep=sleep,
    )
