"""P1 单元测试：服务层 LLM 重试策略。"""

from __future__ import annotations

import pytest

from cortex.llm.resilience import (
    is_retryable,
    resilient_invoke,
)

pytestmark = pytest.mark.unit


class Flaky:
    def __init__(self, failures: int, exc_factory=None):
        self.remaining = failures
        self.calls = 0
        self.exc_factory = exc_factory or (lambda: ConnectionError("connection reset by peer"))

    def invoke(self, messages, **kwargs):
        self.calls += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise self.exc_factory()
        return f"ok after {self.calls} calls"


def test_retries_then_succeeds():
    flaky = Flaky(failures=2)
    sleeps = []
    got = resilient_invoke(flaky, [], retries=4, sleep=sleeps.append, base_delay=1.5, max_delay=30)
    assert got == "ok after 3 calls"
    assert len(sleeps) == 2 and sleeps[0] < sleeps[1]  # 指数退避


def test_non_retryable_raises_immediately():
    flaky = Flaky(failures=5, exc_factory=lambda: ValueError("bad param"))
    sleeps = []
    with pytest.raises(ValueError):
        resilient_invoke(flaky, [], retries=4, sleep=sleeps.append)
    assert sleeps == [] and flaky.calls == 1


def test_retry_after_header_honored():
    flaky = Flaky(failures=1, exc_factory=lambda: RuntimeError("429 Too Many Requests (retry-after: 7)"))
    sleeps = []
    resilient_invoke(flaky, [], retries=3, sleep=sleeps.append)
    assert sleeps and sleeps[0] >= 7


def test_exhaustion_raises_last_error():
    flaky = Flaky(failures=10)
    with pytest.raises(ConnectionError):
        resilient_invoke(flaky, [], retries=2, sleep=lambda _: None)
    assert flaky.calls == 3  # 首次 + 2 次重试


def test_is_retryable_classification():
    assert is_retryable(RuntimeError("HTTP 429 rate limit"))
    assert is_retryable(TimeoutError("request timed out"))
    assert not is_retryable(ValueError("invalid schema"))
