"""Validation for terrain operations and data exchange.

Every terrain test builds a surface whose answer is known analytically -- a
cone of known volume, a plane of known slope, a pyramid whose contours are
computable by hand -- and checks the routine recovers it.
"""

import csv
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))

from fiducia import exchange, terrain  # noqa: E402

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


work = Path(tempfile.mkdtemp(prefix="fiducia-terrain-"))
print(f"\nWorking in {work}\n")

CRS = "+proj=tmerc +lat_0=0 +lon_0=19 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
CELL = 1.0
WEST, NORTH = 50000.0, 6200000.0


def write_dem(path, array, cell=CELL, west=WEST, north=NORTH, nodata=-9999.0):
    data = np.where(np.isfinite(array), array, nodata).astype(np.float32)
    with rasterio.open(
        path, "w", driver="GTiff", width=data.shape[1], height=data.shape[0],
        count=1, dtype="float32", crs=CRS, nodata=nodata,
        transform=Affine(cell, 0, west, 0, -cell, north),
        compress="DEFLATE",
    ) as sink:
        sink.write(data, 1)
    return str(path)


# =====================================================================
print("=== 1. Hillshade ===")
# A plane tilted due east: aspect is constant, so shading must be uniform,
# and a light from the east must brighten it relative to one from the west.

size = 120
cols = np.arange(size)[None, :].repeat(size, 0).astype(float)
east_slope = cols * 0.20            # 20% grade rising to the east
plane = write_dem(work / "plane.tif", east_slope)

terrain.hillshade(plane, str(work / "plane_e.tif"), azimuth=90.0, altitude=45.0)
terrain.hillshade(plane, str(work / "plane_w.tif"), azimuth=270.0, altitude=45.0)

with rasterio.open(work / "plane_e.tif") as ds:
    lit_east = ds.read(1)[10:-10, 10:-10].astype(float)
with rasterio.open(work / "plane_w.tif") as ds:
    lit_west = ds.read(1)[10:-10, 10:-10].astype(float)

check("shading a uniform slope is uniform", float(lit_east.std()) < 1.0,
      f"std {lit_east.std():.3f} DN")
check("an east-facing slope is darker lit from the east",
      lit_east.mean() < lit_west.mean(),
      f"east {lit_east.mean():.0f} DN, west {lit_west.mean():.0f} DN")

expected = math.cos(math.radians(45)) * math.cos(math.atan(0.20)) \
    + math.sin(math.radians(45)) * math.sin(math.atan(0.20)) * math.cos(math.pi)
check("shading matches the analytic value",
      abs(lit_east.mean() / 255.0 - max(expected, 0)) < 0.02,
      f"got {lit_east.mean() / 255:.4f}, expected {max(expected, 0):.4f}")


# =====================================================================
print("\n=== 2. Volume ===")
# A right circular cone of radius R and height H has volume pi*R^2*H/3.

R, H = 40.0, 30.0
yy, xx = np.mgrid[0:size, 0:size].astype(float)
radius = np.hypot(xx - size / 2, yy - size / 2)
cone = np.clip(H * (1 - radius / R), 0, None)

surface = write_dem(work / "cone.tif", cone)
flat = write_dem(work / "flat.tif", np.zeros((size, size)))

volume = terrain.volume_between(surface, flat)
analytic = math.pi * R * R * H / 3.0

check("cut volume matches the analytic cone",
      abs(volume["cutVolume"] - analytic) / analytic < 0.02,
      f"{volume['cutVolume']:,.0f} m3 vs {analytic:,.0f} m3 "
      f"({abs(volume['cutVolume'] - analytic) / analytic:.2%})")
check("no fill against a flat base", volume["fillVolume"] < analytic * 0.001,
      f"{volume['fillVolume']:.1f} m3")
check("peak height recovered", abs(volume["maxAbove"] - H) < 0.6,
      f"{volume['maxAbove']:.2f} m vs {H} m")

inverted = terrain.volume_between(flat, surface)
check("reversing the surfaces swaps cut and fill",
      abs(inverted["fillVolume"] - volume["cutVolume"]) / analytic < 0.02,
      f"fill {inverted['fillVolume']:,.0f} m3")


# =====================================================================
print("\n=== 3. Contours ===")

result = terrain.contours(surface, str(work / "cone.geojson"), interval=5.0, base=0.0)
geo = json.loads((work / "cone.geojson").read_text(encoding="utf-8"))
levels = sorted({f["properties"]["elevation"] for f in geo["features"]})

check("contours written as GeoJSON", geo["type"] == "FeatureCollection",
      f"{len(geo['features'])} lines at {len(levels)} levels")
check("levels fall on the interval",
      all(abs(v % 5.0) < 1e-6 for v in levels), f"{levels[:6]}")
check("levels stay inside the elevation range",
      levels and 0 < min(levels) and max(levels) < H, f"{min(levels)} to {max(levels)}")

# A cone's contour at height h is a circle of radius R(1 - h/H).
ring = [f for f in geo["features"] if abs(f["properties"]["elevation"] - 15.0) < 1e-6]
if ring:
    coords = np.array(max(ring, key=lambda f: len(f["geometry"]["coordinates"]))
                      ["geometry"]["coordinates"])
    centre = coords.mean(axis=0)
    measured = float(np.hypot(coords[:, 0] - centre[0], coords[:, 1] - centre[1]).mean())
    expected_r = R * (1 - 15.0 / H)
    check("the 15 m contour is a circle of the right radius",
          abs(measured - expected_r) / expected_r < 0.08,
          f"{measured:.1f} m vs {expected_r:.1f} m")

check("index contours are flagged",
      any(f["properties"]["index"] for f in geo["features"])
      and not all(f["properties"]["index"] for f in geo["features"]),
      f"{sum(1 for f in geo['features'] if f['properties']['index'])} of "
      f"{len(geo['features'])} are index contours")

try:
    terrain.contours(surface, str(work / "x.geojson"), interval=0.001)
    check("an absurd interval is refused", False, "it was accepted")
except terrain.TerrainError as exc:
    check("an absurd interval is refused", True, str(exc)[:60])


# =====================================================================
print("\n=== 4. Void filling ===")

holed = cone.copy()
holed[50:60, 50:60] = np.nan          # small hole, inside the surface
holed[0:18, 0:18] = np.nan            # large hole in a corner
holed_path = write_dem(work / "holed.tif", holed)

filled = terrain.fill_voids(holed_path, str(work / "filled.tif"), max_radius_cells=6)
with rasterio.open(work / "filled.tif") as ds:
    after = ds.read(1).astype(float)
    after[after <= -9998] = np.nan

small_recovered = np.isfinite(after[52:58, 52:58]).all()
large_left = not np.isfinite(after[2:10, 2:10]).any()

check("a small void is interpolated", small_recovered,
      f"{filled['filledCells']:,} cells filled")
check("a large void is left alone", large_left,
      f"{filled['remainingVoids']:,} cells left as void")
error = np.abs(after[52:58, 52:58] - cone[52:58, 52:58])
check("the interpolated values are close to truth", float(np.nanmax(error)) < 1.5,
      f"worst {np.nanmax(error):.3f} m")


# =====================================================================
print("\n=== 5. Smoothing ===")

rng = np.random.default_rng(3)
noisy = cone + rng.normal(0, 1.2, cone.shape)
spikes = noisy.copy()
spikes[30, 30] = 250.0                 # a matching blunder
noisy_path = write_dem(work / "noisy.tif", spikes)

smoothed = terrain.smooth(noisy_path, str(work / "smoothed.tif"), "median", 5)
with rasterio.open(work / "smoothed.tif") as ds:
    clean_surface = ds.read(1).astype(float)

before_rms = float(np.sqrt(np.mean((spikes - cone) ** 2)))
after_rms = float(np.sqrt(np.mean((clean_surface - cone) ** 2)))
check("median smoothing reduces noise", after_rms < before_rms * 0.7,
      f"RMS {before_rms:.2f} -> {after_rms:.2f} m")
check("the spike is removed", abs(clean_surface[30, 30] - cone[30, 30]) < 3.0,
      f"spike now {clean_surface[30, 30]:.1f} m, truth {cone[30, 30]:.1f} m")


# =====================================================================
print("\n=== 6. Merge ===")

left = cone.copy()
left[:, 70:] = np.nan
right = cone.copy()
right[:, :50] = np.nan
merged = terrain.merge(
    [write_dem(work / "left.tif", left), write_dem(work / "right.tif", right)],
    str(work / "merged.tif"), method="feather",
)

with rasterio.open(work / "merged.tif") as ds:
    joined = ds.read(1).astype(float)
    joined[joined <= -9998] = np.nan

check("the merged surface covers both halves", merged["coverage"] > 0.98,
      f"{merged['coverage']:.1%} coverage")
difference = np.abs(joined - cone)
check("the merge is faithful to the truth", float(np.nanmax(difference)) < 0.6,
      f"worst {np.nanmax(difference):.3f} m, mean {np.nanmean(difference):.4f} m")

seam = joined[:, 55:65]
gradient = np.abs(np.diff(seam, axis=1))
check("no step at the seam", float(np.nanmax(gradient)) < 1.5,
      f"largest adjacent jump {np.nanmax(gradient):.3f} m")


# =====================================================================
print("\n=== 7. Surface to bare earth ===")

ground = 40.0 + xx * 0.05 + yy * 0.03
building = ground.copy()
building[40:60, 40:60] += 12.0
trees = building.copy()
trees[80:90, 20:30] += 8.0
dsm = write_dem(work / "dsm.tif", trees)

dtm = terrain.surface_to_terrain(dsm, str(work / "dtm.tif"),
                                 max_window_cells=33, initial_tolerance=0.4)
with rasterio.open(work / "dtm.tif") as ds:
    bare = ds.read(1).astype(float)

check("the building is removed",
      float(np.abs(bare[45:55, 45:55] - ground[45:55, 45:55]).max()) < 2.0,
      f"worst residual over the building {np.abs(bare[45:55, 45:55] - ground[45:55, 45:55]).max():.2f} m")
check("the trees are removed",
      float(np.abs(bare[82:88, 22:28] - ground[82:88, 22:28]).max()) < 2.0,
      f"worst residual over the trees {np.abs(bare[82:88, 22:28] - ground[82:88, 22:28]).max():.2f} m")
check("open ground is left alone",
      float(np.abs(bare[5:30, 70:95] - ground[5:30, 70:95]).max()) < 0.6,
      f"worst change on bare ground {np.abs(bare[5:30, 70:95] - ground[5:30, 70:95]).max():.3f} m")
check("most of the scene classifies as ground", dtm["groundFraction"] > 0.7,
      f"{dtm['groundFraction']:.0%} ground")


# =====================================================================
print("\n=== 8. Profile ===")

line = [[WEST + 10, NORTH - 60], [WEST + 110, NORTH - 60]]
section = terrain.profile(surface, line, samples=200)

check("profile length is correct", abs(section["length"] - 100.0) < 0.01,
      f"{section['length']:.3f} m")
check("profile finds the summit",
      abs(section["maxElevation"] - cone[60, 60]) < 1.0,
      f"peak {section['maxElevation']:.2f} m")
check("profile is fully sampled", section["coverage"] > 0.99,
      f"{section['coverage']:.1%}")


# =====================================================================
print("\n=== 9. Data exchange ===")

# A survey file in the shape one actually arrives in: semicolons, a BOM,
# a header in surveyor vocabulary, and the columns in an unhelpful order.
survey = work / "control.csv"
survey.write_bytes(
    "\ufeffPoint No;Elevation;Easting;Northing;Type\r\n"
    "TRIG01;123,456;50010,5;6199940,25;control\r\n"
    "TRIG02;98,1;50080,0;6199900,0;control\r\n"
    "CHK01;110,0;50040,0;6199870,0;check\r\n"
    "BAD01;;notanumber;6199800,0;control\r\n".encode("utf-8")
)

sniff = exchange.sniff_table(str(survey))
check("delimiter and header detected", sniff.delimiter == ";" and sniff.has_header,
      f"delimiter {sniff.delimiter!r}, header {sniff.has_header}")
check("surveyor column names are understood",
      {"id", "x", "y", "z", "type"} <= set(sniff.columns),
      f"mapped {sorted(sniff.columns)}")

points = exchange.read_control_points(str(survey))
check("good rows read, bad row reported",
      len(points["points"]) == 3 and len(points["problems"]) == 1,
      f"{len(points['points'])} points, {len(points['problems'])} problem(s)")

first = points["points"][0]
check("european decimal commas handled",
      abs(first["x"] - 50010.5) < 1e-6 and abs(first["z"] - 123.456) < 1e-6,
      f"x={first['x']}, z={first['z']}")
check("check points are recognised",
      sum(1 for p in points["points"] if p["isCheckPoint"]) == 1,
      f"{sum(1 for p in points['points'] if p['isCheckPoint'])} check point")

# Exterior orientation, in degrees, with a header nobody standardised.
log = work / "flight.txt"
log.write_text(
    "photo\tX\tY\tZ\troll\tpitch\tyaw\n"
    "frame_01\t50000.0\t6200000.0\t1200.5\t0.85\t-1.20\t92.4\n"
    "frame_02\t50180.0\t6200005.0\t1201.0\t-0.44\t0.91\t91.8\n",
    encoding="utf-8",
)
eo = exchange.read_exterior_orientation(str(log))
check("flight log read, degrees detected",
      len(eo["records"]) == 2 and eo["angleUnits"] == "degrees",
      f"{len(eo['records'])} frames, {eo['angleUnits']}")
check("angles converted to radians",
      abs(eo["records"][0]["exterior"][5] - math.radians(92.4)) < 1e-9,
      f"kappa {eo['records'][0]['exterior'][5]:.6f} rad")

# Round trip through the exporters.
out_csv = work / "control_out.csv"
exchange.write_control_points(str(out_csv), [
    {"id": "G0001", "x": 1.5, "y": 2.5, "z": 3.5, "isCheckPoint": False},
    {"id": "G0002", "x": 4.5, "y": 5.5, "z": 6.5, "isCheckPoint": True},
], residuals={"G0001": (0.1, -0.2, 0.05)})
rows = list(csv.DictReader(out_csv.open(encoding="utf-8")))
check("control export round-trips",
      len(rows) == 2 and rows[0]["id"] == "G0001" and rows[1]["type"] == "check",
      f"{len(rows)} rows, residual {rows[0]['residual']}")

library = work / "cameras.json"
exchange.save_camera_library(str(library), {"kind": "film", "focalMm": 153.69}, "RC30")
exchange.save_camera_library(str(library), {"kind": "digital", "focalMm": 100.0}, "UAV")
loaded = exchange.load_camera_library(str(library))
check("camera library saves and reloads",
      len(loaded["cameras"]) == 2
      and any(c["label"] == "RC30" for c in loaded["cameras"]),
      f"{[c['label'] for c in loaded['cameras']]}")

print("\n=== Surfaces in different coordinate systems ===")
from pyproj import Transformer  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

UTM = "EPSG:32734"
to_utm = Transformer.from_crs(CRS, UTM, always_xy=True)
from_utm = Transformer.from_crs(UTM, CRS, always_xy=True)


def plane(e, n):
    return 100.0 + 0.02 * (e - WEST) - 0.01 * (n - NORTH)


def write_in(path, crs, fn, west, north, cell, size):
    """A surface defined in the TM frame, stored on a grid in ``crs``."""
    xs, ys = np.meshgrid(west + (np.arange(size) + 0.5) * cell, north - (np.arange(size) + 0.5) * cell)
    if crs == UTM:
        e, n = from_utm.transform(xs.ravel(), ys.ravel())
        e, n = np.asarray(e).reshape(xs.shape), np.asarray(n).reshape(xs.shape)
    else:
        e, n = xs, ys
    with rasterio.open(path, "w", driver="GTiff", width=size, height=size, count=1, dtype="float32",
                       crs=crs, transform=from_origin(west, north, cell, cell), nodata=-9999.0) as ds:
        ds.write(fn(e, n).astype("float32"), 1)


# Two halves of one plane: the western in TM, the eastern stored in UTM.
write_in(work / "west_tm.tif", CRS, plane, WEST, NORTH, 5.0, 120)
ux, uy = to_utm.transform(WEST + 500, NORTH)
write_in(work / "east_utm.tif", UTM, plane, ux, uy, 5.0, 120)
merged = terrain.merge([str(work / "west_tm.tif"), str(work / "east_utm.tif")], str(work / "merged_mixed.tif"))
with rasterio.open(work / "merged_mixed.tif") as ds:
    band = ds.read(1).astype(float)
    band[band == ds.nodata] = np.nan
    rows, cols = np.nonzero(np.isfinite(band))
    xs, ys = ds.transform * (cols + 0.5, rows + 0.5)
    width_m = ds.bounds.right - ds.bounds.left
error = np.nanmax(np.abs(band[rows, cols] - plane(np.asarray(xs), np.asarray(ys))))
check("merging DEMs in different systems builds one correct grid",
      width_m > 1000 and error < 0.05,
      f"merged grid {width_m:.0f} m wide, worst height error {error:.3f} m")

# A cone in TM over a flat base stored in UTM: the volume must be the cone's.
radius, peak = 150.0, 30.0
cx, cy = WEST + 300, NORTH - 300


def cone(e, n):
    return np.maximum(0.0, peak * (1 - np.hypot(e - cx, n - cy) / radius))


write_in(work / "cone_tm.tif", CRS, cone, WEST, NORTH, 1.0, 600)
bx, by = to_utm.transform(WEST - 50, NORTH + 50)
write_in(work / "flat_utm.tif", UTM, lambda e, n: np.zeros_like(e), bx, by, 2.0, 360)
vol = terrain.volume_between(str(work / "cone_tm.tif"), str(work / "flat_utm.tif"))
analytic = math.pi * radius ** 2 * peak / 3
check("volume against a base in another system is still the cone's",
      abs(vol["cutVolume"] - analytic) / analytic < 0.01,
      f"{vol['cutVolume']:.0f} m3 vs analytic {analytic:.0f} m3")

# A DEM in degrees is refused with a reason, not given a meaningless answer.
with rasterio.open(work / "degrees.tif", "w", driver="GTiff", width=50, height=50, count=1, dtype="float32",
                   crs="EPSG:4326", transform=from_origin(18.5, -33.9, 0.0001, 0.0001)) as ds:
    ds.write(np.full((1, 50, 50), 100, dtype="float32"))
refused = []
for label, call in (("hillshade", lambda: terrain.hillshade(str(work / "degrees.tif"), str(work / "hs.tif"))),
                    ("volume", lambda: terrain.volume_between(str(work / "degrees.tif"), str(work / "degrees.tif"))),
                    ("bare earth", lambda: terrain.surface_to_terrain(str(work / "degrees.tif"), str(work / "be.tif"))),
                    ("profile", lambda: terrain.profile(str(work / "degrees.tif"), [(18.501, -33.901), (18.503, -33.903)]))):
    try:
        call()
        refused.append((label, False))
    except terrain.TerrainError as exc:
        refused.append((label, "degrees" in str(exc)))
check("tools that need metres refuse a DEM in degrees, and say why",
      all(ok for _, ok in refused), ", ".join(f"{l}: {'refused' if ok else 'ACCEPTED'}" for l, ok in refused))

print("\n" + "=" * 66)
print(f"  {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("  FAILED: " + ", ".join(FAIL))
print(f"  artefacts in {work}")
print("=" * 66)
sys.exit(1 if FAIL else 0)
