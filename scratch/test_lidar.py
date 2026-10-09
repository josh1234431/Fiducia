"""Validation for ground classification and height above ground.

A synthetic flight over rolling terrain with trees, two buildings and noise,
where the true ground and every tree's height are known. The pulses behave
like real ones under canopy: most first returns hit the top of a crown, but
a share slip deep into it (the cause of pits in a canopy height model), and
last returns land on the ground or on the understorey.
"""

import sys
import tempfile
from pathlib import Path

import laspy
import numpy as np
import rasterio
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))

from fiducia import lidar  # noqa: E402

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


work = Path(tempfile.mkdtemp(prefix="fiducia-lidar-"))
print(f"\nWorking in {work}\n")

CRS = "+proj=utm +zone=35 +south +datum=WGS84 +units=m +no_defs"
WEST, SOUTH, SIZE = 500000.0, 7100000.0, 240.0
rng = np.random.default_rng(7)


def terrain(x, y):
    u, v = x - WEST, y - SOUTH
    return 1200.0 + 0.08 * u + 3.0 * np.sin(u / 40.0) + 2.0 * np.cos(v / 30.0)


# Trees: positions kept apart, heights 8-32 m, paraboloid crowns.
trees = []
while len(trees) < 110:
    x, y = rng.uniform(WEST + 10, WEST + SIZE - 10), rng.uniform(SOUTH + 10, SOUTH + SIZE - 10)
    if WEST + 150 < x < WEST + 215 and SOUTH + 20 < y < SOUTH + 90:
        continue  # the buildings' block
    if all(np.hypot(x - tx, y - ty) > 6 for tx, ty, _, _ in trees):
        h = rng.uniform(8, 32)
        trees.append((x, y, h, 0.22 * h + 1.0))
trees = np.array(trees)
buildings = [(WEST + 160, SOUTH + 30, 20, 30, 9.0), (WEST + 190, SOUTH + 40, 18, 25, 12.0)]


def canopy(x, y):
    """Height of the top of the canopy or roof above the ground, and the tree index."""
    top = np.zeros_like(x)
    which = np.full(x.shape, -1)
    for i, (tx, ty, h, r) in enumerate(trees):
        d = np.hypot(x - tx, y - ty)
        surface = np.where(d < r, h - 0.35 * h * (d / r) ** 2, 0.0)
        higher = surface > top
        top = np.where(higher, surface, top)
        which = np.where(higher, i, which)
    roof = np.zeros_like(x)
    for bx, by, w, d, h in buildings:
        roof = np.where((x >= bx) & (x < bx + w) & (y >= by) & (y < by + d), h, roof)
    return top, which, roof


# Pulses, 8 per square metre.
n = int(SIZE * SIZE * 8)
px = rng.uniform(WEST, WEST + SIZE, n)
py = rng.uniform(SOUTH, SOUTH + SIZE, n)
ground = terrain(px, py)
top, which, roof = canopy(px, py)

xs, ys, zs, rn, nr, truth = [], [], [], [], [], []   # truth: 0 ground, 1 object, 2 low noise, 3 high noise


def add(sel, z, r, total, kind):
    xs.append(px[sel]); ys.append(py[sel]); zs.append(z)
    rn.append(np.full(sel.sum(), r)); nr.append(total); truth.append(np.full(sel.sum(), kind))


open_ground = (top == 0) & (roof == 0)
add(open_ground, ground[open_ground] + rng.normal(0, 0.03, open_ground.sum()), 1, np.ones(open_ground.sum()), 0)
on_roof = roof > 0
add(on_roof, ground[on_roof] + roof[on_roof] + rng.normal(0, 0.03, on_roof.sum()), 1, np.ones(on_roof.sum()), 1)

in_tree = (top > 0) & (roof == 0)
k = in_tree.sum()
deep = rng.random(k) < 0.35
depth = np.where(deep, rng.uniform(0.3, 0.8, k), rng.uniform(0, 0.15 / np.maximum(top[in_tree], 1), k))
first_z = ground[in_tree] + top[in_tree] * (1 - depth)
reaches_ground = rng.random(k) < 0.55
last_z = np.where(reaches_ground, ground[in_tree] + rng.normal(0, 0.03, k),
                  ground[in_tree] + rng.uniform(0.6, 3.0, k))
total = np.full(k, 2)
add(in_tree, first_z, 1, total, 1)
xs.append(px[in_tree]); ys.append(py[in_tree]); zs.append(last_z)
rn.append(np.full(k, 2)); nr.append(total); truth.append(np.where(reaches_ground, 0, 1))

# Noise: low multipath points and high birds.
low_i = rng.choice(n, 300, replace=False)
xs.append(px[low_i]); ys.append(py[low_i]); zs.append(ground[low_i] - rng.uniform(5, 20, 300))
rn.append(np.ones(300)); nr.append(np.ones(300)); truth.append(np.full(300, 2))
high_i = rng.choice(n, 60, replace=False)
xs.append(px[high_i]); ys.append(py[high_i]); zs.append(ground[high_i] + rng.uniform(80, 150, 60))
rn.append(np.ones(60)); nr.append(np.ones(60)); truth.append(np.full(60, 3))

X, Y, Z = np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)
RN, NR, TRUTH = np.concatenate(rn).astype(int), np.concatenate(nr).astype(int), np.concatenate(truth)
true_height = Z - terrain(X, Y)

header = laspy.LasHeader(point_format=1, version="1.4")
header.offsets = [WEST, SOUTH, 1000.0]
header.scales = [0.001, 0.001, 0.001]
from pyproj import CRS as ProjCRS  # noqa: E402
header.add_crs(ProjCRS.from_proj4(CRS))
las = laspy.LasData(header)
las.x, las.y, las.z = X, Y, Z
las.return_number, las.number_of_returns = RN, NR
las.classification = np.zeros(X.size, dtype=np.uint8)
source = work / "forest.laz"
las.write(source)
print(f"  {X.size:,} points, {len(trees)} trees, 2 buildings\n")


# ---------------------------------------------------------------------------
print("=== 1. Ground classification ===")
classified = work / "forest_ground.laz"
result = lidar.classify_ground(str(source), lidar.GroundOptions(output_path=str(classified)))
labels = np.asarray(laspy.read(classified).classification)

found = labels[TRUTH == 0] == 2
check("true ground found", found.mean() >= 0.95, f"{found.mean():.1%} of ground points labelled ground")
tall = (TRUTH == 1) & (true_height > 1.0)
wrong = labels[tall] == 2
check("objects not called ground", wrong.mean() <= 0.01, f"{wrong.mean():.2%} of points >1 m up labelled ground")
roof_pts = (TRUTH == 1) & (labels >= 0) & np.isin(np.arange(X.size), np.nonzero(TRUTH == 1)[0])
on_roofs = np.zeros(X.size, dtype=bool)
for bx, by, w, d, _ in buildings:
    on_roofs |= (X >= bx) & (X < bx + w) & (Y >= by) & (Y < by + d) & (TRUTH == 1)
check("roofs not called ground", (labels[on_roofs] == 2).mean() <= 0.01,
      f"{(labels[on_roofs] == 2).mean():.2%} of roof points")
low_found = (labels[TRUTH == 2] == 7).mean()
check("low noise labelled class 7", low_found >= 0.9, f"{low_found:.1%} of low points")
check("source file left untouched", (np.asarray(laspy.read(source).classification) == 0).all())
check("result counts add up", result["groundPoints"] == int((labels == 2).sum())
      and result["lowNoisePoints"] == int((labels == 7).sum()))

dtm_path = work / "dtm.tif"
lidar.rasterize(str(classified), lidar.RasterizeOptions(
    output_path=str(dtm_path), cell_size=1.0, classes=[2], returns="all", cell_assignment="idw"))
with rasterio.open(dtm_path) as src:
    dtm = src.read(1)
    t = src.transform
rows, cols = np.mgrid[0:dtm.shape[0], 0:dtm.shape[1]]
cx, cy = t * (cols + 0.5, rows + 0.5)
inner = (cx > WEST + 5) & (cx < WEST + SIZE - 5) & (cy > SOUTH + 5) & (cy < SOUTH + SIZE - 5) & (dtm != -9999)
error = dtm[inner] - terrain(cx[inner], cy[inner])
rmse = float(np.sqrt(np.mean(error ** 2)))
check("terrain model from the found ground", rmse <= 0.15, f"RMSE {rmse:.3f} m against the true terrain")


# ---------------------------------------------------------------------------
print("\n=== 2. Height above ground ===")


def chm(name, **kw):
    path = work / f"{name}.tif"
    info = lidar.height_above_ground(str(classified), lidar.HeightOptions(output_path=str(path), **kw))
    with rasterio.open(path) as src:
        grid = src.read(1).astype(float)
        grid[grid == src.nodata] = np.nan
        return grid, src.transform, info


highest, t, info_h = chm("chm_highest", cell_size=0.5)
pitfree, _, info_p = chm("chm_pitfree", cell_size=0.5, method="pit_free")
rows, cols = np.mgrid[0:highest.shape[0], 0:highest.shape[1]]
cx, cy = t * (cols + 0.5, rows + 0.5)
true_top, true_tree, true_roof = canopy(cx, cy)

tops = []
for i, (tx, ty, h, r) in enumerate(trees):
    near = np.hypot(cx - tx, cy - ty) < 0.5 * r
    if near.any():
        tops.append((np.nanmax(highest[near]) - h, np.nanmax(pitfree[near]) - h))
tops = np.array(tops)
check("tree heights, highest point", np.median(np.abs(tops[:, 0])) <= 0.3,
      f"median error {np.median(np.abs(tops[:, 0])):.2f} m over {len(tops)} trees")
check("tree heights, pit-free", np.median(np.abs(tops[:, 1])) <= 0.3,
      f"median error {np.median(np.abs(tops[:, 1])):.2f} m")

# Pits: cells in the inner half of a crown reading more than 2 m low. The
# outer half is left out, where the crown falls away nearly 3 m per metre and
# any half-metre cell reads low whatever the method.
distance = np.full(cx.shape, np.inf)
radius = np.ones(cx.shape)
for i, (tx, ty, h, r) in enumerate(trees):
    mine = true_tree == i
    distance[mine] = np.hypot(cx[mine] - tx, cy[mine] - ty)
    radius[mine] = r
core = (true_tree >= 0) & (true_roof == 0) & (distance < 0.5 * radius)
pits_h = np.nanmean(highest[core] < true_top[core] - 2.0)
pits_p = np.nanmean(pitfree[core] < true_top[core] - 2.0)
check("pit-free removes pits", pits_p <= 0.02 and pits_p < pits_h / 3,
      f"crown cells >2 m low: highest {pits_h:.1%}, pit-free {pits_p:.1%} "
      f"(edge limit {info_p['pitFreeMaxEdge']:.2f} m)")

for bx, by, w, d, h in buildings:
    inside = (cx > bx + 1) & (cx < bx + w - 1) & (cy > by + 1) & (cy < by + d - 1)
    got = float(np.nanmedian(highest[inside]))
    check(f"building {h:g} m tall", abs(got - h) <= 0.2, f"median {got:.2f} m")

clear = (true_top == 0) & (true_roof == 0)
from scipy.ndimage import binary_erosion  # noqa: E402
clear = binary_erosion(clear, iterations=4)
check("open ground reads zero", np.nanmedian(highest[clear]) <= 0.1 and np.isfinite(highest[clear]).mean() > 0.95,
      f"median {np.nanmedian(highest[clear]):.3f} m, {np.isfinite(highest[clear]).mean():.1%} filled")

check("high noise shows without a cap", info_h["maxHeight"] > 60, f"max {info_h['maxHeight']:.1f} m")
capped, _, info_c = chm("chm_capped", cell_size=0.5, max_height=50)
check("height cap drops high noise", info_c["maxHeight"] <= 40 and info_c["pointsAboveMaxHeight"] >= 50,
      f"max {info_c['maxHeight']:.1f} m, {info_c['pointsAboveMaxHeight']} points dropped")

# Ground from a terrain model instead of class 2 gives the same heights.
truth_dtm = work / "true_dtm.tif"
gx = np.arange(WEST - 5, WEST + SIZE + 5, 1.0) + 0.5
gy = np.arange(SOUTH + SIZE + 5, SOUTH - 5, -1.0) - 0.5
GX, GY = np.meshgrid(gx, gy)
with rasterio.open(truth_dtm, "w", driver="GTiff", width=GX.shape[1], height=GX.shape[0], count=1,
                   dtype="float32", crs=CRS, transform=Affine(1.0, 0, WEST - 5, 0, -1.0, SOUTH + SIZE + 5)) as dst:
    dst.write(terrain(GX, GY).astype("float32"), 1)
from_dtm, _, info_d = chm("chm_from_dtm", cell_size=0.5, dtm_path=str(truth_dtm))
both = np.isfinite(from_dtm) & np.isfinite(highest)
difference = np.nanmedian(np.abs(from_dtm[both] - highest[both]))
check("ground from a terrain model", difference <= 0.1 and info_d["groundSource"] == str(truth_dtm),
      f"median difference {difference:.3f} m from the class 2 result")

try:
    lidar.height_above_ground(str(source), lidar.HeightOptions(output_path=str(work / "x.tif")))
    check("no ground gives a clear error", False, "no error raised")
except ValueError as exc:
    check("no ground gives a clear error", "Classify the ground" in str(exc), str(exc)[:60])

vegetation_only, _, _ = chm("chm_classes", cell_size=0.5, classes=[0], returns="all")
check("class and return filters apply", np.isfinite(vegetation_only).mean() < np.isfinite(highest).mean())


# ---------------------------------------------------------------------------
print("\n=== 3. Through the engine, as the interface does it ===")
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from urllib import request  # noqa: E402

PORT = 8793


def call(method, route, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = request.Request(f"http://127.0.0.1:{PORT}{route}", data=data, method=method,
                          headers={"Content-Type": "application/json"} if data else {})
    with request.urlopen(req, timeout=120) as response:
        return json.loads(response.read().decode() or "null")


def wait_for_job(job_id, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = next((j for j in call("GET", "/jobs?limit=60")["jobs"] if j["id"] == job_id), None)
        if job and job["status"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.4)
    raise TimeoutError(job_id)


# FIDUCIA_ENGINE_EXE runs this part against a packaged engine.
engine_exe = os.environ.get("FIDUCIA_ENGINE_EXE")
command = [engine_exe] if engine_exe else [sys.executable, str(ROOT / "engine" / "server.py")]
log = open(work / "engine.log", "w")
server = subprocess.Popen(command + ["--port", str(PORT)], cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
try:
    for _ in range(120):
        try:
            if call("GET", "/health")["ok"]:
                break
        except Exception:
            time.sleep(0.5)
    call("POST", "/project/new", {"directory": str(work / "Lidar.fidu"), "name": "LiDAR"})
    job = wait_for_job(call("POST", "/lidar/classify-ground", {"path": str(source)})["job"]["id"])
    clouds = call("GET", "/project")["state"].get("lidarClouds", [])
    check("ground job runs and is recorded", job["status"] == "done" and len(clouds) == 1
          and Path(clouds[0]["outputPath"]).exists(), job.get("error") or clouds[0]["outputPath"])
    job = wait_for_job(call("POST", "/lidar/height", {
        "path": clouds[0]["outputPath"], "method": "pit_free", "cellSize": 0.5, "maxHeight": 50,
    })["job"]["id"])
    outputs = call("GET", "/project")["state"].get("lidar", [])
    check("height job runs and is recorded", job["status"] == "done" and bool(outputs)
          and outputs[-1]["kind"] == "height" and Path(outputs[-1]["outputPath"]).exists(),
          job.get("error") or f"max {outputs[-1]['maxHeight']:.1f} m")
finally:
    server.terminate()
    server.wait(timeout=20)
    log.close()

print("\n" + "=" * 64)
print(f"  {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("  FAILED: " + ", ".join(FAIL))
print("=" * 64)
sys.exit(1 if FAIL else 0)
