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
# A deep return still lands in the crown, on foliage or a branch below its surface.
depth = np.where(deep, rng.uniform(0.1, 0.35, k), rng.uniform(0, 0.15 / np.maximum(top[in_tree], 1), k))
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
print("\n=== 3. Noise ===")
for method in ("statistical", "isolated"):
    path = work / f"forest_{method}.laz"
    info = lidar.filter_noise(str(classified), lidar.NoiseOptions(output_path=str(path), method=method))
    labels_n = np.asarray(laspy.read(path).classification)
    birds = (labels_n[TRUTH == 3] == 18).mean()
    real = np.isin(TRUTH, (0, 1))
    false_alarm = np.isin(labels_n[real], (7, 18)).mean()
    check(f"{method}: birds labelled high noise", birds >= 0.9, f"{birds:.0%} of 60")
    check(f"{method}: real points left alone", false_alarm <= 0.005, f"{false_alarm:.2%} labelled noise")
check("noise filter keeps earlier labels", (labels_n[labels == 2] == 2).mean() > 0.99
      and (labels_n[TRUTH == 2] == 7).mean() > 0.9)


# ---------------------------------------------------------------------------
print("\n=== 4. Looking at the cloud ===")
view = lidar.overview(str(classified), size=256, colour="class")
check("overview image", view["image"].startswith("data:image/png;base64,") and view["width"] <= 256)
cut = lidar.section(str(classified), (WEST + 10, SOUTH + 120), (WEST + 230, SOUTH + 120), width=2.0)
along_ok = np.all(np.diff(cut["along"]) >= 0) and 0 <= min(cut["along"]) and max(cut["along"]) <= cut["length"]
idx = np.array(cut["index"])
check("section holds the corridor's points", cut["total"] > 1000 and along_ok
      and np.all(np.abs(Y[idx] - (SOUTH + 120)) <= 1.001), f"{cut['total']:,} points along {cut['length']:.0f} m")
check("section points keep their file numbers", np.allclose(np.array(cut["z"]), Z[idx], atol=0.002))


# ---------------------------------------------------------------------------
print("\n=== 5. Classes learned from labels ===")
from fiducia import lidar_learn  # noqa: E402

# What each point really is: ground, understorey (last returns off the
# ground), canopy (first returns, deep or not), buildings.
first = RN == 1
truth_class = np.full(X.size, 1)
truth_class[TRUTH == 0] = 2
truth_class[(TRUTH == 1) & ~first] = 3
truth_class[(TRUTH == 1) & first] = 5
truth_class[on_roofs] = 6
real = np.isin(TRUTH, (0, 1))

# The operator brushes everything in two sections, one through the trees and
# one through the buildings: a few minutes' labelling.
strips = (np.abs(Y - (SOUTH + 120)) <= 1.0) | (np.abs(X - (WEST + 175)) <= 1.0)
labelled = np.nonzero(strips & real)[0]
user_labels = {int(i): int(truth_class[i]) for i in labelled}
print(f"  {len(user_labels):,} points labelled in two sections ({len(user_labels) / X.size:.1%} of the cloud)")
learned = work / "forest_learned.laz"
report = lidar_learn.train_and_apply(str(classified), user_labels,
                                     lidar_learn.LearnOptions(output_path=str(learned)))
result_classes = np.asarray(laspy.read(learned).classification)
unseen = real & ~strips
accuracy = (result_classes[unseen] == truth_class[unseen]).mean()
check("learned classes on unseen points", accuracy >= 0.9, f"{accuracy:.1%} correct on {unseen.sum():,} points")
for code, name in ((2, "ground"), (3, "understorey"), (5, "canopy"), (6, "buildings")):
    mine = unseen & (truth_class == code)
    recall = (result_classes[mine] == code).mean()
    check(f"  {name} recovered", recall >= 0.85, f"{recall:.1%}")
check("out-of-bag estimate is honest", abs(report["accuracy"] - accuracy) <= 0.06,
      f"estimated {report['accuracy']:.1%}, measured {accuracy:.1%}")
check("labelled points keep their labels",
      all(result_classes[i] == c for i, c in list(user_labels.items())[:2000]))
check("noise is not relabelled", (result_classes[TRUTH == 2] == 7).mean() > 0.9)
check("uncertain spots offered", 1 <= len(report["uncertainSpots"]) <= 8,
      f"{len(report['uncertainSpots'])} spots, lowest confidence {report['uncertainSpots'][0]['confidence']:.2f}")
print("  most telling measurements: " + ", ".join(i["feature"] for i in report["importance"][:4]))

# The harder case: stray returns in the canopy column, with the same return
# pattern as real canopy hits. Returns do not give them away; where they sit
# and the shape of their neighbourhood can. Those that land inside a crown
# are indistinguishable from a real deep hit by any method, so a share of
# them is expected to be missed.
print("  -- stray returns inside dense crowns --")
n_stray = 4000
pick = rng.choice(np.nonzero(in_tree)[0], n_stray)
stray_x = px[pick] + rng.normal(0, 0.3, n_stray)
stray_y = py[pick] + rng.normal(0, 0.3, n_stray)
stray_z = terrain(stray_x, stray_y) + rng.uniform(1.0, top[pick])  # anywhere in the canopy column
noisy = laspy.read(classified)
base = len(noisy.x)
extra = laspy.ScaleAwarePointRecord.zeros(n_stray, header=noisy.header)
extra.x, extra.y, extra.z = stray_x, stray_y, stray_z
extra.return_number, extra.number_of_returns = np.ones(n_stray, int), np.full(n_stray, 2)
noisy_path = work / "forest_stray.laz"
with laspy.open(noisy_path, mode="w", header=noisy.header) as writer:
    writer.write_points(noisy.points)
    writer.write_points(extra)
truth_noisy = np.concatenate([truth_class, np.full(n_stray, 18)])
truth_noisy[np.nonzero(TRUTH == 2)[0]] = 7
truth_noisy[np.nonzero(TRUTH == 3)[0]] = 18
SX, SY = np.concatenate([X, stray_x]), np.concatenate([Y, stray_y])
strips_n = (np.abs(SY - (SOUTH + 120)) <= 1.0) | (np.abs(SX - (WEST + 175)) <= 1.0)
stray_labels = {int(i): int(truth_noisy[i]) for i in np.nonzero(strips_n & (truth_noisy != 7))[0]}
print(f"  {sum(v == 18 for v in stray_labels.values())} stray returns among "
      f"{len(stray_labels):,} labelled points")
stray_out = work / "forest_stray_learned.laz"
stray_report = lidar_learn.train_and_apply(str(noisy_path), stray_labels,
                                           lidar_learn.LearnOptions(output_path=str(stray_out)))
got = np.asarray(laspy.read(stray_out).classification)
unseen_n = ~strips_n
stray_unseen = unseen_n & (np.arange(got.size) >= base)
canopy_unseen = unseen_n & (truth_noisy == 5)
found_stray = (got[stray_unseen] == 18).mean()
lost_canopy = (got[canopy_unseen] == 18).mean()
# Strays in the open column under the crowns can be told apart; those inside
# a crown look exactly like a real deep hit, so they are only reported.
stray_rel = np.concatenate([np.zeros(base), stray_z - terrain(stray_x, stray_y)])
crown_floor = np.concatenate([np.zeros(base), 0.6 * top[pick]])
in_air = stray_unseen & (stray_rel < crown_floor)
in_crown = stray_unseen & (stray_rel >= crown_floor)
found_air = (got[in_air] == 18).mean()
print(f"  first round: {found_air:.1%} of {in_air.sum()} unseen strays below the crowns found; "
      f"inside crowns {(got[in_crown] == 18).mean():.1%} (they look like real hits)")

# Second round, as the operator would: a section through each spot the
# forest was least sure of, labelled, and train again.
more = np.zeros(SX.size, dtype=bool)
for spot in stray_report["uncertainSpots"]:
    more |= (np.abs(SY - spot["y"]) <= 1.0) & (np.abs(SX - spot["x"]) <= 15.0)
round_two = dict(stray_labels)
round_two.update({int(i): int(truth_noisy[i]) for i in np.nonzero(more & (truth_noisy != 7))[0]})
seen = strips_n | more
stray_out2 = work / "forest_stray_learned2.laz"
lidar_learn.train_and_apply(str(noisy_path), round_two, lidar_learn.LearnOptions(output_path=str(stray_out2)))
got2 = np.asarray(laspy.read(stray_out2).classification)
in_air2 = in_air & ~seen
found_air2 = (got2[in_air2] == 18).mean()
lost_canopy2 = (got2[canopy_unseen & ~seen] == 18).mean()
open_air = in_air2 & (stray_rel > 3.5)
found_open = (got2[open_air] == 18).mean()
check("second round finds strays in the open air under the crowns", found_open >= 0.85,
      f"{found_open:.1%} of {open_air.sum()} unseen, after {len(round_two) - len(stray_labels):,} more labels "
      f"in {len(stray_report['uncertainSpots'])} sections (first round {found_air:.1%} of all below the crowns)")
check("and still keeps the real canopy", lost_canopy2 <= 0.03, f"{lost_canopy2:.2%} called noise")
print(f"  among the understorey (1-3.5 m), where strays mix with real returns: "
      f"{(got2[in_air2 & (stray_rel <= 3.5)] == 18).mean():.1%} found")
noise_row = next(c for c in stray_report["classes"] if c["class"] == 18)
check("report gives the noise class its own recall", 0 <= noise_row["recall"] <= 1,
      f"out-of-bag recall {noise_row['recall']:.1%}, precision {noise_row['precision']:.1%}")
check("real canopy kept", lost_canopy <= 0.03, f"{lost_canopy:.2%} of unseen canopy called noise")
print(f"  out-of-bag estimate {stray_report['accuracy']:.1%}; most telling: "
      + ", ".join(i["feature"] for i in stray_report["importance"][:4]))

only = work / "forest_only.laz"
lidar_learn.train_and_apply(str(classified), user_labels,
                            lidar_learn.LearnOptions(output_path=str(only), change_classes=[2]))
only_classes = np.asarray(laspy.read(only).classification)
untouched = ~np.isin(labels, [2]) & ~np.isin(np.arange(X.size), labelled)
check("only the chosen classes change", (only_classes[untouched] == labels[untouched]).all())
try:
    lidar_learn.train_and_apply(str(classified), {1: 2, 2: 2}, lidar_learn.LearnOptions(output_path=str(work / "y.laz")))
    check("one class gives a clear error", False)
except ValueError as exc:
    check("one class gives a clear error", "two classes" in str(exc))


# ---------------------------------------------------------------------------
print("\n=== 6. Through the engine, as the interface does it ===")
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

    ground_cloud = clouds[0]["outputPath"]
    view = call("POST", "/lidar/overview", {"path": ground_cloud, "colour": "height"})
    line = {"path": ground_cloud, "start": [WEST + 10, SOUTH + 120], "end": [WEST + 230, SOUTH + 120], "width": 2}
    cut = call("POST", "/lidar/section", line)
    check("overview and section served", view["image"].startswith("data:image/png") and cut["total"] > 1000)

    # Label through the API as the side view does, then train.
    picked = np.array(cut["index"])
    call("POST", "/lidar/labels", {"path": ground_cloud,
                                   "set": {str(i): int(c) for i, c in zip(picked, truth_class[picked])}})
    counts = call("GET", f"/lidar/labels?path={request.quote(ground_cloud)}")
    returned = call("POST", "/lidar/section", line)["labels"]
    check("labels saved and returned with the section", counts["count"] == picked.size
          and len(returned) == picked.size, f"{counts['count']:,} labels, {counts['byClass']}")

    job = wait_for_job(call("POST", "/lidar/noise", {"path": ground_cloud})["job"]["id"])
    clouds = call("GET", "/project")["state"]["lidarClouds"]
    denoised = clouds[-1]["outputPath"]
    carried = call("GET", f"/lidar/labels?path={request.quote(denoised)}")["count"]
    check("noise job runs and labels follow the new cloud", job["status"] == "done"
          and clouds[-1]["kind"] == "noise" and carried == picked.size,
          job.get("error") or f"{clouds[-1]['noisePoints']} noise points, {carried:,} labels carried")

    job = wait_for_job(call("POST", "/lidar/learn", {"path": denoised})["job"]["id"], timeout=600)
    state = call("GET", "/project")["state"]
    check("training job runs and reports per class", job["status"] == "done"
          and state["lidarClouds"][-1]["kind"] == "learned" and len(state["lidarLearning"]["classes"]) >= 3,
          job.get("error") or f"{state['lidarClouds'][-1]['pointsChanged']:,} points changed, "
          f"out-of-bag {state['lidarLearning']['accuracy']:.1%}")
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
