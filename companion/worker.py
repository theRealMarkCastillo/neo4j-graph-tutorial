"""Lesson 14 — run memory extraction in the background, so replies never wait for it.

Extraction is an extra LLM call plus graph writes: one to several seconds per message. The
reply doesn't need it, so the chat hands the job to a worker and answers straight away.

One worker THREAD per conversation, not a pool:
- jobs for the same user run in order. Job 2's prompt includes the memories job 1 wrote
  ("you already know Maya is the sister"), so they must not overlap.
- MemoryStore._write() also locks the user's node, so even writers in other processes (a
  web server with several workers, a nightly job) can't interleave. The queue is for order
  and latency, the lock is for correctness.

Results come back as log lines: `drain()` returns what's finished without blocking (the chat
prints them before the next prompt), `wait()` blocks until the queue is empty (call it before
the end-of-episode pass, which must see every memory).
"""
from __future__ import annotations

import queue
import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor


class Background:
    def __init__(self) -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="memory")
        self._pending: list[Future] = []
        self._done: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._lock = threading.Lock()

    def submit(self, job) -> Future:
        """Queue `job()` (returns a list of log lines). Returns at once."""
        def run():
            try:
                lines = job() or []
            except Exception as e:  # a background error must surface somewhere
                traceback.print_exc()
                lines = [f"memory update failed: {type(e).__name__}: {e}"]
            for line in lines:
                self._done.put(line)

        with self._lock:
            future = self._pool.submit(run)
            self._pending.append(future)
        return future

    def drain(self) -> list[str]:
        """Log lines from jobs that have finished so far. Never blocks."""
        lines = []
        while True:
            try:
                lines.append(self._done.get_nowait())
            except queue.Empty:
                return lines

    def wait(self, timeout: float | None = None) -> list[str]:
        """Block until every queued job has finished, then return their log lines."""
        with self._lock:
            pending, self._pending = self._pending, []
        for future in pending:
            future.result(timeout)
        return self.drain()

    @property
    def busy(self) -> bool:
        with self._lock:
            return any(not f.done() for f in self._pending)

    @property
    def queued(self) -> int:
        """Jobs submitted but not finished (the one running included)."""
        with self._lock:
            return sum(not f.done() for f in self._pending)

    def close(self) -> None:
        self._pool.shutdown(wait=True)
