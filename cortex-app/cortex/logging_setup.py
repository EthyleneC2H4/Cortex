"""日志与 stdout 静音工具。

框架（hello-agents）在 register_tool / HistoryManager.compress 等处直接 print，
服务层调用这些 API 时用 :func:`quiet` 包住，避免污染 REPL/API 输出。
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import sys


def setup_logging(level: str | None = None) -> None:
    level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )


@contextlib.contextmanager
def quiet():
    """静音 Python 层 stdout（框架 print）；返回捕获缓冲便于调试。"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf
