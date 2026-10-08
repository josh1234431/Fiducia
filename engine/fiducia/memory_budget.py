"""How much of the machine a job may use.

Parallel work is only faster while it fits in memory. Past that point it is
catastrophically slower: Windows starts paging, then refuses allocations, and
the first thing to fall over is often not the program that caused it but the
desktop compositor, which takes the screen with it. A workstation also runs
other things -- ArcGIS, a browser, the operator's own spreadsheets -- so the
budget is what is free now, not what is installed.

Three rules follow, and every pool in Fiducia goes through them:

* size a pool by free memory as well as by cores;
* cap each worker's raster cache, because GDAL's default (5% of RAM, per
  process) multiplies by the worker count and alone can exhaust a machine;
* keep watching while the job runs, and back off before Windows has to.

"Free memory" here is available commit -- RAM plus page file not yet
promised to anyone -- because that, not free RAM, is what runs out when
allocations start failing.
"""

from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass

__all__ = [
    "available_bytes",
    "total_bytes",
    "plan_workers",
    "WorkerPlan",
    "WORKER_CACHE_MB",
    "ENGINE_CACHE_MB",
    "limit_raster_cache",
    "single_threaded_children",
]

GB = 1024 ** 3
MB = 1024 ** 2

# Raster block cache per worker. A worker renders one output tile at a time
# and reads only the source block under it, so a small cache loses almost
# nothing; left at GDAL's default it would hold most of a decoded photo.
WORKER_CACHE_MB = 96
ENGINE_CACHE_MB = 256

# What one render worker costs: the interpreter with numpy, scipy and GDAL
# loaded (about 80 MB with one maths thread), its capped raster cache, and the
# arrays for one 512 px tile. Measured on a real 13080 x 20010 JPEG frame by
# scratch/test_memory_budget.py, and rounded up so the estimate errs towards fewer
# workers; that test fails if a worker ever exceeds it.
PER_WORKER_BYTES = 200 * MB   # measured peak 72 MB

# Always leave this much for everything else on the machine.
RESERVE_BYTES = int(1.5 * GB)


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _windows_status() -> _MemoryStatus | None:
    if sys.platform != "win32":
        return None
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(_MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return status


def available_bytes() -> int:
    """Memory that can still be allocated right now, across the machine."""
    status = _windows_status()
    if status is not None:
        # Commit still available (RAM + page file), and never more than the
        # RAM actually free plus what the page file could still take.
        return int(status.ullAvailPageFile)
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return 4 * GB


def total_bytes() -> int:
    status = _windows_status()
    if status is not None:
        return int(status.ullTotalPhys)
    try:
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return 8 * GB


@dataclass
class WorkerPlan:
    workers: int
    reason: str
    available: int

    def describe(self) -> str:
        return f"{self.workers} worker{'s' if self.workers != 1 else ''} ({self.reason})"


def plan_workers(jobs: int, requested: int = 0, per_worker: int = PER_WORKER_BYTES,
                 available: int | None = None) -> WorkerPlan:
    """How many worker processes this machine can afford for ``jobs`` pieces.

    Half of the memory above the reserve goes to the pool; the rest stays for
    the engine, the interface and whatever else the operator is running.
    """
    cores = os.cpu_count() or 4
    free = available_bytes() if available is None else available
    affordable = max(0, (free - RESERVE_BYTES) // 2) // per_worker
    ceiling = requested or cores

    workers = max(1, min(ceiling, cores, jobs, int(affordable)))
    if workers == min(ceiling, cores, jobs):
        reason = "one per core" if workers == cores else "as requested" if requested else "enough for the job"
    else:
        reason = f"limited by free memory, {free / GB:.1f} GB available"
    return WorkerPlan(workers, reason, free)


BLAS_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def single_threaded_children() -> None:
    """Make processes started from here use one maths thread each.

    numpy's OpenBLAS sets aside a buffer for every core, in every process,
    when it loads: measured at 1.3 GB committed by an idle worker on a
    20-core machine, against 79 MB with one thread. Workers are parallel
    already -- one per process -- so the extra threads buy nothing and the
    buffers, multiplied by the pool, are enough to exhaust a machine on their
    own. Spawned processes inherit the environment, so setting it here
    reaches them before they import numpy.
    """
    for name in BLAS_VARIABLES:
        os.environ[name] = "1"


def limit_raster_cache(megabytes: int) -> None:
    """Cap GDAL's block cache for this process.

    Must run before the first raster is read. Set through the environment so
    it also reaches child processes started afterwards.
    """
    os.environ["GDAL_CACHEMAX"] = str(int(megabytes))
    try:
        from rasterio.env import set_gdal_config

        set_gdal_config("GDAL_CACHEMAX", int(megabytes))
    except Exception:  # noqa: BLE001
        pass
