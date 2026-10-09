"""Memory stays bounded as point clouds grow.

Builds a synthetic forest, copies it side by side into clouds 4 and 16 times
the size, and runs every LiDAR tool on each in a fresh process, reading the
process's peak committed memory from Windows. A tool passes when its peak on
the largest cloud is no more than a third above its peak on the 4x cloud and
under 2 GB: memory follows the tile, not the cloud.
"""

import ctypes
import ctypes.wintypes as wt
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import laspy
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


# -- run as a probe: one tool on one cloud, report the peak ----------------------

if len(sys.argv) > 1 and sys.argv[1] == "--probe":
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

    kernel = ctypes.WinDLL("kernel32")
    kernel.GetCurrentProcess.restype = wt.HANDLE
    psapi = ctypes.WinDLL("psapi")
    psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(Counters), wt.DWORD]

    def committed():
        c = Counters()
        c.cb = ctypes.sizeof(c)
        psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.PeakPagefileUsage, c.PagefileUsage

    from fiducia import lidar, lidar_learn, lidar_tiles  # noqa: E402
    import scipy.spatial  # noqa: F401,E402
    import PIL.Image  # noqa: F401,E402

    tool, path, out, cache = sys.argv[2:6]
    lidar_tiles.set_cache_root(cache)
    base = committed()[1]
    started = time.time()
    if tool == "index":
        lidar.inspect_cloud(path)
    elif tool == "rasterise":
        lidar.rasterize(path, lidar.RasterizeOptions(output_path=out, classes=[2], cell_size=1.0))
    elif tool == "ground":
        lidar.classify_ground(path, lidar.GroundOptions(output_path=out))
    elif tool == "noise":
        lidar.filter_noise(path, lidar.NoiseOptions(output_path=out))
    elif tool == "height":
        lidar.height_above_ground(path, lidar.HeightOptions(output_path=out, method="pit_free"))
    elif tool == "view":
        lidar.overview(path)
        lidar.section(path, (500005, 7100120), (500235, 7100120))
    elif tool == "learn":
        labels = json.loads(Path(sys.argv[6]).read_text())
        lidar_learn.train_and_apply(path, labels, lidar_learn.LearnOptions(output_path=out, trees=20))
    print(json.dumps({"peak": committed()[0] - base, "seconds": time.time() - started}))
    sys.exit(0)


# -- the clouds ---------------------------------------------------------------------

work = Path(tempfile.mkdtemp(prefix="fiducia-lidar-memory-"))
print(f"\nWorking in {work}\n")
rng = np.random.default_rng(3)
WEST, SOUTH, SIZE = 500000.0, 7100000.0, 240.0


def block():
    """One 240 m block: rolling ground, trees, first and last returns, ground classified."""
    n = int(SIZE * SIZE * 8)
    x = rng.uniform(0, SIZE, n)
    y = rng.uniform(0, SIZE, n)
    ground = 1200 + 0.05 * x + 2 * np.sin(y / 30)
    trees = rng.uniform(5, SIZE - 5, (90, 2))
    heights = rng.uniform(8, 30, 90)
    top = np.zeros(n)
    for (tx, ty), h in zip(trees, heights):
        r = 0.22 * h + 1
        d = np.hypot(x - tx, y - ty)
        top = np.maximum(top, np.where(d < r, h - 0.35 * h * (d / r) ** 2, 0))
    in_tree = top > 0
    first_z = ground + top * np.where(rng.random(n) < 0.3, rng.uniform(0.65, 0.95, n), 1.0)
    xs = np.concatenate([x, x[in_tree]])
    ys = np.concatenate([y, y[in_tree]])
    zs = np.concatenate([np.where(in_tree, first_z, ground), ground[in_tree]])
    rn = np.concatenate([np.ones(n, int), np.full(in_tree.sum(), 2)])
    nr = np.concatenate([np.where(in_tree, 2, 1), np.full(in_tree.sum(), 2)])
    cls = np.concatenate([np.where(in_tree, 5, 2), np.full(in_tree.sum(), 2)]).astype(np.uint8)
    return xs, ys, zs, rn, nr, cls


base = block()
header = laspy.LasHeader(point_format=1, version="1.4")
header.offsets = [WEST, SOUTH, 1000.0]
header.scales = [0.001, 0.001, 0.001]
from pyproj import CRS  # noqa: E402
header.add_crs(CRS.from_epsg(32735))


def make(copies_per_side):
    path = work / f"forest_{copies_per_side ** 2}x.laz"
    with laspy.open(path, mode="w", header=header) as writer:
        for i in range(copies_per_side):
            for j in range(copies_per_side):
                xs, ys, zs, rn, nr, cls = base
                points = laspy.ScaleAwarePointRecord.zeros(xs.size, header=header)
                points.x, points.y, points.z = WEST + xs + i * SIZE, SOUTH + ys + j * SIZE, zs
                points.return_number, points.number_of_returns = rn, nr
                points.classification = cls
                writer.write_points(points)
    return path


clouds = {f"{k * k}x": make(k) for k in (1, 2, 4)}
for name, path in clouds.items():
    print(f"  {name}: {laspy.open(path).header.point_count:,} points")

# Labels: every point in one strip of the first block, by its class.
xs, ys, zs, rn, nr, cls = base
strip = np.nonzero(np.abs(ys - 120) <= 1.0)[0]
labels_path = work / "labels.json"
labels_path.write_text(json.dumps({str(int(i)): int(cls[i]) for i in strip}))


# -- the probes -------------------------------------------------------------------------

print("\n=== Peak memory by tool ===")
tools = ["index", "rasterise", "ground", "noise", "height", "view", "learn"]
peaks = {}
for name, path in clouds.items():
    cache = work / f"cache_{name}"
    for tool in tools:
        out = work / f"{name}_{tool}{'.tif' if tool in ('rasterise', 'height') else '.laz'}"
        command = [sys.executable, __file__, "--probe", tool, str(path), str(out), str(cache), str(labels_path)]
        done = subprocess.run(command, capture_output=True, text=True, cwd=str(ROOT))
        if done.returncode != 0:
            print(f"  {name} {tool}: failed\n{done.stderr[-800:]}")
            peaks[(name, tool)] = None
            continue
        result = json.loads(done.stdout.strip().splitlines()[-1])
        peaks[(name, tool)] = result["peak"]
        print(f"  {name:4s} {tool:10s} {result['peak'] / 1e6:8.0f} MB  {result['seconds']:6.1f} s")

print()
for tool in tools:
    small, large = peaks.get(("4x", tool)), peaks.get(("16x", tool))
    if small is None or large is None:
        check(f"{tool}: runs on every size", False)
        continue
    check(f"{tool}: memory does not grow with the cloud", large <= 1.33 * small + 50e6 and large < 2e9,
          f"{small / 1e6:.0f} MB at 4x, {large / 1e6:.0f} MB at 16x")

print("\n" + "=" * 64)
print(f"  {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("  FAILED: " + ", ".join(FAIL))
print("=" * 64)
sys.exit(1 if FAIL else 0)
