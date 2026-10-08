"""Background job queue.

Every long-running operation in Fiducia goes through here: ortho generation,
mosaicking, tie point collection, stereo DEM extraction, LiDAR rasterising.
Nothing blocks the interface, everything reports progress, and anything can be
cancelled mid-flight.

This is the structural answer to "the software is slow and it crashes". It is
not that the arithmetic is faster -- it is that the arithmetic happens
somewhere the operator is not waiting on it, several jobs run at once, a job
that dies takes only itself down, and the queue survives to tell you what
happened.

Jobs are run on threads, not processes. The heavy numerical work inside them
(ortho tiling, mosaic blending) spawns its own process pools, and numpy and
GDAL release the GIL during the parts that matter, so a thread here is the
right level of granularity.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .storage import explain

__all__ = ["Job", "JobQueue", "JobCancelled"]


class JobCancelled(RuntimeError):
    pass


@dataclass
class Job:
    id: str
    kind: str
    label: str
    status: str = "queued"          # queued | running | done | failed | cancelled
    progress: float = 0.0
    message: str = ""
    result: Any = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        elapsed = None
        if self.started_at:
            elapsed = (self.finished_at or time.time()) - self.started_at
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "status": self.status,
            "progress": round(self.progress, 4),
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "traceback": self.traceback,
            "createdAt": self.created_at,
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "elapsedSeconds": elapsed,
            "meta": self.meta,
        }


class JobQueue:
    """A bounded pool of worker threads with progress and cancellation.

    ``max_concurrent`` limits how many jobs run at once. Keeping it above one
    matters in practice: an operator can kick off a mosaic and carry on
    collecting tie points for the next block while it runs.
    """

    def __init__(self, max_concurrent: int = 3, history: int = 200):
        self._jobs: dict[str, Job] = {}
        self._order: deque[str] = deque(maxlen=history)
        self._cancels: dict[str, threading.Event] = {}
        self._lock = threading.RLock()
        self._semaphore = threading.Semaphore(max_concurrent)
        self._listeners: list[Callable[[dict], None]] = []

    # -- subscription ----------------------------------------------------

    def subscribe(self, listener: Callable[[dict], None]) -> Callable[[], None]:
        with self._lock:
            self._listeners.append(listener)

        def unsubscribe() -> None:
            with self._lock:
                if listener in self._listeners:
                    self._listeners.remove(listener)

        return unsubscribe

    def _emit(self, job: Job) -> None:
        payload = {"type": "job", "job": job.to_dict()}
        for listener in list(self._listeners):
            try:
                listener(payload)
            except Exception:
                pass

    # -- submission ------------------------------------------------------

    def submit(
        self,
        kind: str,
        label: str,
        work: Callable[..., Any],
        meta: Optional[dict] = None,
    ) -> Job:
        """Queue a unit of work.

        ``work`` is called as ``work(progress, should_cancel)`` where
        ``progress(fraction, message)`` reports status and ``should_cancel()``
        returns True once cancellation has been requested. Work that ignores
        ``should_cancel`` simply runs to completion; work that honours it stops
        promptly.
        """
        job = Job(id=f"job_{uuid.uuid4().hex[:12]}", kind=kind, label=label, meta=meta or {})

        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._cancels[job.id] = threading.Event()

        self._emit(job)

        thread = threading.Thread(target=self._run, args=(job, work), daemon=True)
        thread.start()
        return job

    def _run(self, job: Job, work: Callable[..., Any]) -> None:
        cancel_event = self._cancels[job.id]

        # Wait for a slot, but honour a cancel issued while still queued.
        while not self._semaphore.acquire(timeout=0.2):
            if cancel_event.is_set():
                job.status = "cancelled"
                job.message = "Cancelled before starting"
                job.finished_at = time.time()
                self._emit(job)
                return

        try:
            if cancel_event.is_set():
                job.status = "cancelled"
                job.message = "Cancelled before starting"
                job.finished_at = time.time()
                self._emit(job)
                return

            job.status = "running"
            job.started_at = time.time()
            job.message = "Starting"
            self._emit(job)

            last_emit = 0.0

            def progress(fraction: float, message: str = "") -> None:
                nonlocal last_emit
                job.progress = max(0.0, min(1.0, float(fraction)))
                if message:
                    job.message = message
                now = time.time()
                # Throttle: a tiled job can call this thousands of times, and
                # flooding the websocket is its own kind of slow.
                if now - last_emit > 0.12 or job.progress >= 1.0:
                    last_emit = now
                    self._emit(job)

            def should_cancel() -> bool:
                return cancel_event.is_set()

            result = work(progress, should_cancel)

            if cancel_event.is_set():
                job.status = "cancelled"
                job.message = "Cancelled"
            else:
                job.status = "done"
                job.progress = 1.0
                job.result = result
                job.message = "Complete"

        except (JobCancelled, InterruptedError):
            job.status = "cancelled"
            job.message = "Cancelled"
        except Exception as exc:
            job.status = "failed"
            # An error written for the operator is shown as written; anything
            # else keeps its type, which is the first clue to what broke.
            if isinstance(exc, OSError) and not getattr(exc, "for_operator", False):
                exc = explain(exc, action="read or write")
            job.error = (str(exc) if getattr(exc, "for_operator", False)
                         else f"{type(exc).__name__}: {exc}")
            job.traceback = traceback.format_exc()
            job.message = "Failed"
        finally:
            job.finished_at = time.time()
            self._semaphore.release()
            self._emit(job)

    # -- control ---------------------------------------------------------

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            event = self._cancels.get(job_id)
            job = self._jobs.get(job_id)
        if event is None or job is None:
            return False
        if job.status in ("done", "failed", "cancelled"):
            return False
        event.set()
        job.message = "Cancelling"
        self._emit(job)
        return True

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 50, active_only: bool = False) -> list[dict]:
        with self._lock:
            jobs = [self._jobs[i] for i in reversed(self._order) if i in self._jobs]
        if active_only:
            jobs = [j for j in jobs if j.status in ("queued", "running")]
        return [j.to_dict() for j in jobs[:limit]]

    def dismiss(self, job_id: str) -> bool:
        """Forget one finished job. Running or queued jobs must be cancelled first."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status not in ("done", "failed", "cancelled"):
                return False
            self._jobs.pop(job_id, None)
            self._cancels.pop(job_id, None)
        return True

    def clear_finished(self) -> int:
        with self._lock:
            finished = [
                i for i, j in self._jobs.items()
                if j.status in ("done", "failed", "cancelled")
            ]
            for job_id in finished:
                self._jobs.pop(job_id, None)
                self._cancels.pop(job_id, None)
        return len(finished)
