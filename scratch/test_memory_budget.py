"""Validation for memory-aware orthorectification.

Built around a real failure: generating an orthophoto of a 13080 x 20010
JPEG-compressed frame started one worker per core (20), each with GDAL's
default raster cache of 5% of RAM, and exhausted the machine's memory -- the
desktop compositor and several Windows services crashed with it.

Checks that the pool is sized by free memory, that each worker's cache is
actually capped, that parallel and single-process output are identical, that
the job backs off to one process when memory runs short and stops cleanly when
it runs out, and that a stuck pool is abandoned rather than waited on forever.

Part 5 runs a bounded area of a real photo (FIDUCIA_REAL_PROJECT, a solved
.fidu project) while sampling the memory of every process involved.
"""

import os
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import rasterio

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "scratch"))

from fiducia import memory_budget, ortho          # noqa: E402
from fiducia.camera import CameraModel      # noqa: E402
from fiducia.collinearity import rotation_matrix  # noqa: E402
from fiducia.ortho import OrthoSpec, generate_ortho  # noqa: E402

PASS, FAIL = [], []
GB = 1024 ** 3


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def children_of(pid):
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"(Get-CimInstance Win32_Process -Filter 'ParentProcessId={pid}' | "
         "Where-Object Name -eq 'python.exe').ProcessId"],
        capture_output=True, text=True)
    return [int(x) for x in out.stdout.split() if x.strip().isdigit()]


def build_scene(work):
    """A synthetic frame and the spec to orthorectify it."""
    cols, rows, focal, pitch, gsd = 1600, 1200, 50.0, 0.01, 0.25
    height = focal / 1000 * gsd / (pitch / 1000)
    camera = CameraModel(kind="digital", name="Synthetic", focal_mm=focal, pixel_pitch_mm=pitch,
                         columns=cols, rows=rows)
    eo = np.array([40000.0, 6100000.0, height, 0.01, -0.008, 0.6])
    cc, rr = np.meshgrid(np.arange(cols), np.arange(rows))
    film = camera.pixel_to_film(np.column_stack([cc.ravel(), rr.ravel()]))
    direction = np.column_stack([film, np.full(len(film), -focal)]) @ rotation_matrix(*eo[3:])
    t = -eo[2] / direction[:, 2]
    east = eo[0] + t * direction[:, 0]
    north = eo[1] + t * direction[:, 1]
    band = (((np.floor(east / 7) + np.floor(north / 7)) % 2) * 120 + 60
            + 40 * np.sin(east / 3.1)).reshape(rows, cols).astype(np.uint8)
    path = work / "frame.tif"
    with rasterio.open(path, "w", driver="GTiff", width=cols, height=rows, count=3, dtype="uint8",
                       tiled=True, blockxsize=256, blockysize=256, compress="JPEG") as ds:
        for b in (1, 2, 3):
            ds.write(band, b)
    crs = "+proj=tmerc +lat_0=0 +lon_0=19 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"

    def spec(name, workers=0):
        return OrthoSpec(image_path=str(path), output_path=str(work / f"{name}.tif"),
                         camera=camera.to_dict(), fiducial_fit=None, exterior=eo.tolist(),
                         output_crs=crs, pixel_size_x=gsd, pixel_size_y=gsd,
                         background_elevation=0.0, tile_size=128, max_workers=workers)
    return spec


def read(path):
    with rasterio.open(path) as ds:
        return ds.read()


if __name__ == "__main__":
    work = Path(tempfile.mkdtemp(prefix="fiducia-memory_budget-"))

    # ================================================================
    print("\n=== 1. Sizing the pool ===")
    cores = os.cpu_count()
    for free_gb, expect in ((40, cores), (4.9, None), (1.2, 1)):
        plan = memory_budget.plan_workers(900, available=int(free_gb * GB))
        ok = (plan.workers == expect) if expect else 1 < plan.workers < cores
        check(f"{free_gb} GB free on {cores} cores", ok, plan.describe())
    worst = memory_budget.RESERVE_BYTES + max(0, memory_budget.plan_workers(900, available=int(4.9 * GB)).workers) \
        * memory_budget.PER_WORKER_BYTES * 2
    check("a planned pool never claims more than half the memory above the reserve",
          worst <= 4.9 * GB + memory_budget.PER_WORKER_BYTES, f"{worst / GB:.2f} GB")
    check("never zero workers, never more than there are jobs",
          memory_budget.plan_workers(3, available=100 * GB).workers == 3
          and memory_budget.plan_workers(900, available=0).workers == 1)

    # ================================================================
    print("\n=== 2. Each worker's raster cache is capped ===")
    import _pool_helpers
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=2, initializer=ortho._worker_init,
                             initargs=(memory_budget.WORKER_CACHE_MB,)) as pool:
        settings = list(pool.map(_pool_helpers.cache_setting, range(2)))
    check("workers run with the capped cache, not GDAL's 5% of RAM",
          all(str(s) == str(memory_budget.WORKER_CACHE_MB) for s in settings), f"GDAL_CACHEMAX = {settings}")

    # ================================================================
    print("\n=== 3. Parallel and single-process output are identical ===")
    spec = build_scene(work)
    serial = generate_ortho(spec("serial", workers=1))
    parallel = generate_ortho(spec("parallel", workers=3))
    a, b = read(serial.output_path), read(parallel.output_path)
    check("byte-identical orthophotos", a.shape == b.shape and np.array_equal(a, b),
          f"{a.shape}, {serial.tiles_written} tiles, {serial.valid_fraction:.0%} valid")
    check("no workers left behind", children_of(os.getpid()) == [], str(children_of(os.getpid())))

    # ================================================================
    print("\n=== 4. Running short of memory, and running out ===")
    real = memory_budget.available_bytes
    calls = {"n": 0}

    def draining():
        # Plenty at the start, then it falls away mid-job.
        calls["n"] += 1
        return 16 * GB if calls["n"] <= 12 else int(0.8 * GB)

    memory_budget.available_bytes = draining
    try:
        messages = []
        result = generate_ortho(spec("backoff", workers=4), progress=lambda f, m: messages.append(m))
    finally:
        memory_budget.available_bytes = real
    c = read(result.output_path)
    check("falls back to one process when memory runs short",
          any("Continuing in one process" in m for m in messages)
          and any("single process" in w for w in result.warnings),
          next((m for m in messages if "Continuing" in m), "no fallback message"))
    check("and the result is still identical", np.array_equal(a, c))
    check("no workers left behind after backing off", children_of(os.getpid()) == [])

    memory_budget.available_bytes = lambda: int(0.2 * GB)
    try:
        failed = None
        try:
            generate_ortho(spec("starved", workers=1))
        except ortho.LowMemory as exc:
            failed = exc
    finally:
        memory_budget.available_bytes = real
    check("with no memory left it stops with a plain message instead of crashing Windows",
          failed is not None and "keep your computer usable" in str(failed), str(failed)[:100])
    check("and leaves no half-written file", not (work / "starved.tif").exists())

    # ================================================================
    print("\n=== 5. A stuck pool is abandoned ===")
    original_init, original_stall = ortho._worker_init, ortho.STALL_SECONDS
    ortho._worker_init = _pool_helpers.hang_on_start
    ortho.STALL_SECONDS = 4
    started = time.time()
    try:
        messages = []
        stuck = generate_ortho(spec("stuck", workers=2), progress=lambda f, m: messages.append(m))
    finally:
        ortho._worker_init, ortho.STALL_SECONDS = original_init, original_stall
    elapsed = time.time() - started
    check("gives up on the pool and finishes in one process",
          any("no tile finished" in m for m in messages) and np.array_equal(a, read(stuck.output_path)),
          f"{elapsed:.1f} s")
    time.sleep(1)
    check("the stuck workers are terminated", children_of(os.getpid()) == [],
          str(children_of(os.getpid())))

    # ================================================================
    print("\n=== 6. The real photo, with memory measured ===")
    real = os.environ.get("FIDUCIA_REAL_PROJECT")
    if not real or not Path(real).exists():
        print("  (skipped: set FIDUCIA_REAL_PROJECT to a solved .fidu project)")
    else:
        import json
        state = json.load(open(Path(real) / "project.json", encoding="utf-8"))
        image = state["images"][0]
        eo = state["model"]["exterior"][image["id"]]
        x0, y0 = eo[0], eo[1]
        area = OrthoSpec(
            image_path=image["path"], output_path=str(work / "real_area.tif"),
            camera=state["camera"], fiducial_fit=None, exterior=eo, output_crs=state["projection"]["output"],
            pixel_size_x=0.15, pixel_size_y=0.15, dem_path=state["dem"]["referencePath"],
            bounds=(x0 - 300, y0 - 300, x0 + 300, y0 + 300))
        plan = memory_budget.plan_workers(64)
        peak = {"total": 0, "workers": 0, "count": 0}
        running = True

        def sample():
            me = os.getpid()
            while running:
                out = subprocess.run(
                    ["powershell", "-NoProfile", "-Command",
                     f"Get-CimInstance Win32_Process -Filter 'ParentProcessId={me}' | "
                     "Where-Object Name -eq 'python.exe' | ForEach-Object { $_.PrivatePageCount }"],
                    capture_output=True, text=True)
                sizes = [int(x) for x in out.stdout.split() if x.strip().isdigit()]
                if sizes:
                    peak["workers"] = max(peak["workers"], max(sizes))
                    peak["total"] = max(peak["total"], sum(sizes))
                    peak["count"] = max(peak["count"], len(sizes))
                time.sleep(0.5)

        thread = threading.Thread(target=sample, daemon=True)
        free_before = memory_budget.available_bytes()
        thread.start()
        started = time.time()
        result = generate_ortho(area)
        running = False
        thread.join()
        print(f"     {result.width} x {result.height} px, {result.tiles_written} tiles in "
              f"{time.time() - started:.1f} s with {plan.describe()}")
        print(f"     free memory before: {free_before / GB:.1f} GB; workers seen: {peak['count']}; "
              f"largest worker {peak['workers'] / 2**20:.0f} MB; all workers together "
              f"{peak['total'] / 2**20:.0f} MB")
        check("the planned pool fits well inside free memory",
              peak["total"] < 0.5 * max(0, free_before - memory_budget.RESERVE_BYTES) + 256 * 2**20,
              f"{peak['total'] / 2**20:.0f} MB used of {free_before / GB:.1f} GB free")
        check("the per-worker estimate is conservative",
              peak["workers"] <= memory_budget.PER_WORKER_BYTES,
              f"measured {peak['workers'] / 2**20:.0f} MB vs {memory_budget.PER_WORKER_BYTES / 2**20:.0f} MB assumed")
        check("the area is covered", result.valid_fraction > 0.95, f"{result.valid_fraction:.1%} valid")

    print("\n" + "=" * 64)
    print(f"  TOTAL {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("  FAILED: " + ", ".join(FAIL))
    print("=" * 64)
    sys.exit(1 if FAIL else 0)
