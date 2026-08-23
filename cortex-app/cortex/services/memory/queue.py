"""MemoryWriteQueue：单守护线程异步写队列（FIFO 有序、flush 可等待、close 降级直写）。

- put() 入队即返回；队列满且 drop_oldest=True 时丢最旧并计数 dropped；
- flush(timeout)：阻塞直到「已入队 == 已处理完」或超时，返回是否排空；
- close(timeout)：投递哨兵结束工作线程；关闭后 put() 降级为同步直调 sink；
- sink 抛异常不外泄：计数 errors 后继续消费。
"""

from __future__ import annotations

import queue as _queue
import threading
import time
from typing import Any, Callable

_SENTINEL = object()


class MemoryWriteQueue:
    """记忆写入的异步缓冲（async 模式由 MemoryService 按配置启停）。"""

    def __init__(
        self,
        sink: Callable[[Any], None],
        maxsize: int = 1000,
        drop_oldest: bool = True,
    ) -> None:
        self._sink = sink
        self._drop_oldest = bool(drop_oldest)
        self._q: "_queue.Queue" = _queue.Queue(maxsize=int(maxsize))
        self._lock = threading.Lock()
        self._enqueued = 0
        self._processed = 0
        self._dropped = 0
        self._errors = 0
        self._closed = False
        self._thread = threading.Thread(
            target=self._worker, name="memory-write-queue", daemon=True
        )
        self._thread.start()

    # ---------------------------------------------------------------- #
    def put(self, item: Any) -> None:
        """入队一条写入；已关闭则降级同步直调 sink（调用方无需特判）。"""
        with self._lock:
            if self._closed:
                self._enqueued += 1
                sync_fallback = True
            else:
                self._enqueued += 1
                sync_fallback = False
        if sync_fallback:
            self._safe_sink(item)
            with self._lock:
                self._processed += 1
            return

        while True:
            try:
                self._q.put_nowait(item)
                return
            except _queue.Full:
                if not self._drop_oldest:
                    self._q.put(item)  # 阻塞直至有空间
                    return
                try:  # 丢最旧：被丢条目视为已离队（计入 processed 防止 flush 卡死）
                    self._q.get_nowait()
                    with self._lock:
                        self._dropped += 1
                        self._processed += 1
                except _queue.Empty:
                    continue

    def flush(self, timeout: float = 10.0) -> bool:
        """等待全部已入队条目处理完；超时返回 False。"""
        deadline = time.monotonic() + float(timeout)
        while True:
            with self._lock:
                if self._processed >= self._enqueued and self._q.empty():
                    return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.005)

    def close(self, timeout: float = 10.0) -> bool:
        """停机：置 closed → 投哨兵 → join 线程。返回线程是否按时退出。"""
        with self._lock:
            already = self._closed
            self._closed = True
            if not already:
                try:
                    self._q.put_nowait(_SENTINEL)
                except _queue.Full:  # 队满时挤掉一个最旧条目腾位
                    try:
                        self._q.get_nowait()
                        self._dropped += 1
                    except _queue.Empty:
                        pass
                    try:
                        self._q.put_nowait(_SENTINEL)
                    except _queue.Full:  # pragma: no cover —— 理论不可达
                        pass
        self._thread.join(float(timeout))
        return not self._thread.is_alive()

    # ---------------------------------------------------------------- #
    @property
    def stats(self) -> dict:
        """计数器快照：enqueued / processed / dropped / errors。"""
        with self._lock:
            return {
                "enqueued": self._enqueued,
                "processed": self._processed,
                "dropped": self._dropped,
                "errors": self._errors,
            }

    # ---------------------------------------------------------------- #
    def _worker(self) -> None:
        while True:
            item = self._q.get()
            if item is _SENTINEL:
                self._q.task_done()
                return
            self._safe_sink(item)
            with self._lock:
                self._processed += 1
            self._q.task_done()

    def _safe_sink(self, item: Any) -> None:
        try:
            self._sink(item)
        except Exception:  # noqa: BLE001 —— 单条失败不拖垮队列
            with self._lock:
                self._errors += 1
