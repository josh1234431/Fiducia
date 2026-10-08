"""Orthorectification across coordinate systems.

The orientation is solved in the control's coordinate system; the output may
be requested in another, and the DEM may be stored in a third. Before this was
handled, the output grid was laid out in control-frame numbers but stamped
with the output projection, and the DEM was read at control-frame numbers
whatever its own projection -- so a mismatch misplaced the orthophoto and read
heights from the wrong part of the DEM.

A synthetic photo is made over hilly ground near Cape Town, with its
orientation in north-oriented Lo19. Orthophotos are produced into three
output systems using DEMs stored in three systems, and every output pixel is
checked against the known ground pattern at the place it claims to show.
"""

import itertools
import math
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))

from fiducia.camera import CameraModel               # noqa: E402
from fiducia.collinearity import rotation_matrix     # noqa: E402
from fiducia.geodesy import resolve_crs              # noqa: E402
from fiducia.ortho import OrthoSpec, generate_ortho  # noqa: E402

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


CONTROL = "ZALO19_EN"
E0, N0 = -42000.0, -3742000.0


def terrain(e, n):
    """Hills: about 60 m of relief across the photo."""
    return 150 + 30 * np.sin((e - E0) / 60.0) * np.cos((n - N0) / 80.0) + 0.05 * (e - E0)


def _lattice(i, j):
    """Deterministic pseudo-random value in [0, 1) for each integer cell."""
    v = np.sin(i * 12.9898 + j * 78.233) * 43758.5453
    return v - np.floor(v)


def _value_noise(e, n, cell):
    u, v = e / cell, n / cell
    i, j = np.floor(u), np.floor(v)
    fu, fv = u - i, v - j
    fu, fv = fu * fu * (3 - 2 * fu), fv * fv * (3 - 2 * fv)
    a, b = _lattice(i, j), _lattice(i + 1, j)
    c, d = _lattice(i, j + 1), _lattice(i + 1, j + 1)
    return (a * (1 - fu) + b * fu) * (1 - fv) + (c * (1 - fu) + d * fu) * fv


def texture(e, n):
    """Ground clutter that never repeats: an image matcher cannot be fooled by
    periodicity, which a checkerboard would do (it matches one period off)."""
    return np.clip(40 + 150 * _value_noise(e, n, 4.0) + 70 * _value_noise(e, n, 1.3), 0, 255)


work = Path(tempfile.mkdtemp(prefix="fiducia-crs-"))
FOCAL, PITCH, COLS, ROWS = 50.0, 0.01, 900, 700
camera = CameraModel(kind="digital", name="Synthetic", focal_mm=FOCAL, pixel_pitch_mm=PITCH,
                     columns=COLS, rows=ROWS)
EO = np.array([E0, N0, 1650.0, math.radians(1.5), math.radians(-1.0), math.radians(25.0)])

# The photo: each pixel's ray, intersected with the hills by iteration.
cc, rr = np.meshgrid(np.arange(COLS), np.arange(ROWS))
film = camera.pixel_to_film(np.column_stack([cc.ravel(), rr.ravel()]))
direction = np.column_stack([film, np.full(len(film), -FOCAL)]) @ rotation_matrix(*EO[3:])
z = np.full(len(film), 150.0)
for _ in range(30):
    t = (z - EO[2]) / direction[:, 2]
    e, n = EO[0] + t * direction[:, 0], EO[1] + t * direction[:, 1]
    z = terrain(e, n)
photo = texture(e, n).reshape(ROWS, COLS).astype(np.uint8)
image_path = work / "photo.tif"
with rasterio.open(image_path, "w", driver="GTiff", width=COLS, height=ROWS, count=1, dtype="uint8",
                   tiled=True, blockxsize=256, blockysize=256) as ds:
    ds.write(photo, 1)


def write_dem(key):
    """The same hills, stored on a grid in another system."""
    crs = resolve_crs(key)
    to = Transformer.from_crs(resolve_crs(CONTROL), crs, always_xy=True)
    back = Transformer.from_crs(crs, resolve_crs(CONTROL), always_xy=True)
    cx, cy = to.transform([E0 - 400, E0 + 400, E0 - 400, E0 + 400], [N0 - 400, N0 - 400, N0 + 400, N0 + 400])
    step = 2.0
    x0, y1 = min(cx), max(cy)
    width = int((max(cx) - x0) / step) + 1
    height = int((y1 - min(cy)) / step) + 1
    gx, gy = np.meshgrid(x0 + (np.arange(width) + 0.5) * step, y1 - (np.arange(height) + 0.5) * step)
    ce, cn = back.transform(gx.ravel(), gy.ravel())
    heights = terrain(np.asarray(ce), np.asarray(cn)).reshape(height, width).astype("float32")
    path = work / f"dem_{key}.tif"
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=1, dtype="float32",
                       crs=crs, transform=from_origin(x0, y1, step, step)) as ds:
        ds.write(heights, 1)
    return str(path)


def score(result_path, output_key):
    """Correlation of the ortho with the true ground pattern where it claims."""
    with rasterio.open(result_path) as ds:
        band = ds.read(1)
        rows, cols = np.nonzero(band > 0)
        pick = np.random.default_rng(3).choice(len(rows), size=min(6000, len(rows)), replace=False)
        xs, ys = ds.transform * (cols[pick] + 0.5, rows[pick] + 0.5)
        to_control = Transformer.from_crs(ds.crs, resolve_crs(CONTROL), always_xy=True)
    ce, cn = to_control.transform(np.asarray(xs), np.asarray(ys))
    expected = texture(np.asarray(ce), np.asarray(cn))
    actual = band[rows[pick], cols[pick]].astype(float)
    return float(np.corrcoef(expected, actual)[0, 1]), len(rows)


print("\n=== Output system x DEM system (orientation in north-oriented Lo19) ===")
dems = {key: write_dem(key) for key in (CONTROL, "ZALO19", "UTM34S")}
worst = 1.0
for output_key, dem_key in itertools.product((CONTROL, "ZALO19", "UTM34S"), dems):
    spec = OrthoSpec(image_path=str(image_path), output_path=str(work / f"o_{output_key}_{dem_key}.tif"),
                     camera=camera.to_dict(), fiducial_fit=None, exterior=EO.tolist(),
                     output_crs=output_key, control_crs=CONTROL, pixel_size_x=0.3, pixel_size_y=0.3,
                     dem_path=dems[dem_key], max_workers=1, tile_size=256)
    result = generate_ortho(spec)
    correlation, count = score(result.output_path, output_key)
    worst = min(worst, correlation)
    print(f"  output {output_key:<10} DEM {dem_key:<10} correlation {correlation:.4f}  "
          f"({count:,} px, {result.valid_fraction:.0%} valid)")
check("every combination matches the true ground", worst > 0.95, f"worst correlation {worst:.4f}")

print("\n=== The old behaviour, for comparison ===")
# Orientation numbers read as if they were in the output system: what
# happened whenever the control and output projections differed.
spec = OrthoSpec(image_path=str(image_path), output_path=str(work / "old_way.tif"),
                 camera=camera.to_dict(), fiducial_fit=None, exterior=EO.tolist(),
                 output_crs="ZALO19", control_crs=None, pixel_size_x=0.3, pixel_size_y=0.3,
                 dem_path=dems[CONTROL], max_workers=1, tile_size=256)
try:
    result = generate_ortho(spec)
    correlation, _ = score(result.output_path, "ZALO19")
    print(f"  orientation in E/N, output stamped south-oriented, no conversion: correlation {correlation:.3f}")
    check("without conversion the output is wrong (the bug was real)", correlation < 0.5)
except Exception as exc:  # noqa: BLE001
    print(f"  failed outright: {exc}")
    check("without conversion the output is wrong (the bug was real)", True)

# A DEM read at the wrong place: heights from control-frame numbers on a UTM
# DEM miss it entirely and fall back to a flat plane, which on 60 m of relief
# misplaces the edges of the photo by metres.
spec = OrthoSpec(image_path=str(image_path), output_path=str(work / "flat.tif"),
                 camera=camera.to_dict(), fiducial_fit=None, exterior=EO.tolist(),
                 output_crs=CONTROL, control_crs=CONTROL, pixel_size_x=0.3, pixel_size_y=0.3,
                 dem_path=None, background_elevation=150.0, max_workers=1, tile_size=256)
flat, _ = score(generate_ortho(spec).output_path, CONTROL)
print(f"  heights ignored (flat plane): correlation {flat:.3f}")
check("the DEM matters here, so the test can tell", flat < worst - 0.1,
      f"{flat:.3f} flat vs {worst:.3f} with the DEM in any system")

print("\n=== Automatic control across systems ===")
# Orientation in north-oriented Lo19 and deliberately wrong by about 12 m; the
# reference orthophoto in UTM 34S; the DEM in south-oriented Lo19. Automatic control
# must still report where each matched feature truly is, in the control frame.
from fiducia import auto_control  # noqa: E402

reference = str(work / f"o_UTM34S_UTM34S.tif")
BIAS = np.array([9.13, -7.07, 0.0])
record = {"id": "img", "name": "photo", "path": str(image_path), "online": True,
          "width": COLS, "height": ROWS, "bandCount": 1,
          "exterior": (EO + np.r_[BIAS, 0, 0, 0]).tolist(), "fiducialFit": None, "clipRegion": None}
options = auto_control.AutoControlOptions(target_count=12, search_m=30.0, min_score=0.5,
                                     min_separation_m=25.0, patch_px=64, edge_margin_px=60)


def true_ground(col, row):
    f = camera.pixel_to_film([[col, row]])[0]
    d = rotation_matrix(*EO[3:]).T @ np.array([f[0], f[1], -FOCAL])
    height = 150.0
    for _ in range(30):
        t = (height - EO[2]) / d[2]
        e, n = EO[0] + t * d[0], EO[1] + t * d[1]
        height = terrain(e, n)
    return e, n, height


def errors(result):
    out = []
    for p in result.proposals:
        e, n, h = true_ground(p.col, p.row)
        out.append((math.hypot(p.x - e, p.y - n), abs(p.z - h)))
    return np.array(out) if out else np.zeros((0, 2))


same = auto_control.match_ground_control(record, camera.to_dict(), str(work / f"o_{CONTROL}_{CONTROL}.tif"),
                                      dems[CONTROL], None, options, control_crs=CONTROL)
same_err = errors(same)
print(f"  everything in one system: {len(same.proposals)} proposals; horizontal error median "
      f"{np.median(same_err[:, 0]) if len(same_err) else float('nan'):.3f} m")
check("in one system, proposals land where the features truly are",
      len(same_err) >= 5 and np.median(same_err[:, 0]) < 0.3)

found = auto_control.match_ground_control(record, camera.to_dict(), reference, dems["ZALO19"],
                                       None, options, control_crs=CONTROL)
err = errors(found)
print(f"  {len(found.proposals)} proposals; horizontal error median "
      f"{np.median(err[:, 0]) if len(err) else float('nan'):.3f} m, worst "
      f"{err[:, 0].max() if len(err) else float('nan'):.3f} m; height error median "
      f"{np.median(err[:, 1]) if len(err) else float('nan'):.3f} m  (bias was {np.hypot(*BIAS[:2]):.1f} m)")
check("proposals found across three systems", len(found.proposals) >= 5, f"{len(found.proposals)}")
check("they are where the features truly are", len(err) and np.median(err[:, 0]) < 0.3,
      "median within a pixel (0.3 m)")
check("with heights read from the DEM in its own system", len(err) and np.median(err[:, 1]) < 0.5)

old = auto_control.match_ground_control(record, camera.to_dict(), reference, dems["ZALO19"], None, options)
old_err = errors(old)
print(f"  without the control system (the old behaviour): {len(old.proposals)} proposals"
      + (f", median error {np.median(old_err[:, 0]):.1f} m" if len(old_err) else ""))
check("without it, matching fails or lands elsewhere (the bug was real)",
      len(old_err) == 0 or np.median(old_err[:, 0]) > 5)

print("\n" + "=" * 64)
print(f"  TOTAL {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("  FAILED: " + ", ".join(FAIL))
print("=" * 64)
sys.exit(1 if FAIL else 0)
