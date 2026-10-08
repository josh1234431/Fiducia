"""End-to-end test of the raster pipeline: ortho, mosaic, project, reports.

Builds a synthetic aerial survey — a known ground pattern, photographed from
three known camera stations — then runs the real pipeline over it and checks
the orthos land where the geometry says they should.
"""

import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))

from fiducia import raster
from fiducia.camera import CameraModel
from fiducia.collinearity import project_points
from fiducia.mosaic import MosaicSpec, generate_mosaic, mosaic_preview
from fiducia.ortho import OrthoSpec, compute_footprint, generate_ortho
from fiducia.project import Project
from fiducia.reports import project_report, residual_report

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


FOCAL = 100.0
PIXEL_PITCH = 0.006          # 6 um, a digital frame sensor
COLS, ROWS = 1200, 900
GSD = 0.30
FLYING_HEIGHT = FOCAL / 1000.0 / PIXEL_PITCH * 1000.0 * GSD / 1000.0 * 1000.0
FLYING_HEIGHT = (FOCAL / 1000.0) * (GSD / (PIXEL_PITCH / 1000.0))

work = Path(tempfile.mkdtemp(prefix="fiducia-test-"))
print(f"\nWorking in {work}")
print(f"Flying height {FLYING_HEIGHT:.1f} m, GSD {GSD} m\n")

ORIGIN_E, ORIGIN_N = 40000.0, 6100000.0


def ground_pattern(east, north):
    """A deterministic, high-contrast ground texture with known structure."""
    u = (east - ORIGIN_E) / 30.0
    v = (north - ORIGIN_N) / 30.0
    checker = ((np.floor(u) + np.floor(v)) % 2) * 90 + 40
    stripes = 55 * np.sin(u * 1.6) * np.cos(v * 1.1)
    grain = 22 * np.sin(u * 9.3 + v * 7.1)
    return np.clip(checker + stripes + grain, 0, 255)


print("=== 1. Synthesising three aerial frames ===")

camera = CameraModel(
    kind="digital", name="Synthetic frame", focal_mm=FOCAL,
    pixel_pitch_mm=PIXEL_PITCH, columns=COLS, rows=ROWS, image_scale=0.0,
)

base = 180.0
stations = {
    "A": np.array([ORIGIN_E + 0.0,        ORIGIN_N, FLYING_HEIGHT, 0.004, -0.003, 0.02]),
    "B": np.array([ORIGIN_E + base,       ORIGIN_N + 6, FLYING_HEIGHT + 5, -0.002, 0.005, -0.01]),
    "C": np.array([ORIGIN_E + 2 * base,   ORIGIN_N - 4, FLYING_HEIGHT - 3, 0.003, 0.001, 0.015]),
}

image_paths = {}
for name, eo in stations.items():
    cols, rows_idx = np.meshgrid(np.arange(COLS), np.arange(ROWS))
    pixels = np.column_stack([cols.ravel(), rows_idx.ravel()])

    # Backward map: for each image pixel, where on the flat ground does it look?
    film = camera.pixel_to_film(pixels)
    from fiducia.collinearity import rotation_matrix

    rot = rotation_matrix(eo[3], eo[4], eo[5])
    directions = np.column_stack(
        [film[:, 0], film[:, 1], np.full(film.shape[0], -FOCAL)]
    ) @ rot
    t = (0.0 - eo[2]) / directions[:, 2]
    east = eo[0] + t * directions[:, 0]
    north = eo[1] + t * directions[:, 1]

    band = ground_pattern(east, north).reshape(ROWS, COLS).astype(np.uint8)
    path = work / f"frame_{name}.tif"
    with rasterio.open(
        path, "w", driver="GTiff", width=COLS, height=ROWS, count=1,
        dtype="uint8", compress="DEFLATE", tiled=True, blockxsize=256, blockysize=256,
    ) as sink:
        sink.write(band, 1)
    image_paths[name] = str(path)
    print(f"  frame_{name}.tif  {COLS}x{ROWS}")

check("three frames written", len(image_paths) == 3)


print("\n=== 2. Raster access ===")
info = raster.raster_info(image_paths["A"])
check("reads dimensions", info.width == COLS and info.height == ROWS,
      f"{info.width}x{info.height}, {info.driver}")

built = raster.build_overviews(image_paths["A"])
check("builds overview pyramid", len(built) > 0, f"levels {built}")

tile = raster.render_tile(image_paths["A"], 0, 0, 0)
check("renders a PNG tile", tile[:8] == b"\x89PNG\r\n\x1a\n" and len(tile) > 500,
      f"{len(tile)} bytes")


print("\n=== 3. Orthorectification ===")
crs = "+proj=tmerc +lat_0=0 +lon_0=19 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"

ortho_paths = {}
for name, eo in stations.items():
    spec = OrthoSpec(
        image_path=image_paths[name],
        output_path=str(work / f"o_{name}.tif"),
        camera=camera.to_dict(),
        fiducial_fit=None,
        exterior=eo.tolist(),
        output_crs=crs,
        pixel_size_x=GSD,
        pixel_size_y=GSD,
        dem_path=None,
        background_elevation=0.0,
        resampling="bilinear",
        max_workers=1,
    )

    bounds = compute_footprint(spec)
    started = time.time()
    result = generate_ortho(spec)
    ortho_paths[name] = result.output_path
    print(f"  o_{name}.tif  {result.width}x{result.height}  "
          f"{result.valid_fraction:.0%} valid  {time.time() - started:.1f}s")

check("all three orthos generated", len(ortho_paths) == 3)

with rasterio.open(ortho_paths["A"]) as ds:
    ortho_a = ds.read(1)
    transform_a = ds.transform
    check("ortho carries the output CRS", ds.crs is not None, str(ds.crs)[:40])
    check("ortho pixel size matches request",
          abs(ds.res[0] - GSD) < 1e-9 and abs(ds.res[1] - GSD) < 1e-9,
          f"{ds.res}")

# The decisive test: does the orthorectified imagery actually agree with the
# ground pattern it was synthesised from, at the right map coordinates?
valid = ortho_a > 0
rows_v, cols_v = np.nonzero(valid)
sample = np.random.default_rng(5).choice(len(rows_v), size=min(4000, len(rows_v)), replace=False)
rs, cs = rows_v[sample], cols_v[sample]

east_s, north_s = transform_a * (cs + 0.5, rs + 0.5)
expected = ground_pattern(np.asarray(east_s), np.asarray(north_s))
actual = ortho_a[rs, cs].astype(float)

difference = np.abs(actual - expected)
agree = float(np.mean(difference < 28))
correlation = float(np.corrcoef(actual, expected)[0, 1])

check("ortho matches the true ground pattern", correlation > 0.95,
      f"correlation {correlation:.4f}, {agree:.1%} of pixels within tolerance")
check("median radiometric error is small", float(np.median(difference)) < 12,
      f"median |diff| {np.median(difference):.1f} DN")


print("\n=== 4. Geometric accuracy ===")
# Project a set of known ground points into the photo, then check the ortho
# places them at the correct map coordinate.
test_ground = np.array([
    [ORIGIN_E + 40, ORIGIN_N + 20, 0.0],
    [ORIGIN_E + 90, ORIGIN_N - 35, 0.0],
    [ORIGIN_E + 15, ORIGIN_N + 55, 0.0],
])
projected = project_points(test_ground, stations["A"], FOCAL)
back_pixels = camera.film_to_pixel(projected)

inside = (
    (back_pixels[:, 0] >= 0) & (back_pixels[:, 0] < COLS)
    & (back_pixels[:, 1] >= 0) & (back_pixels[:, 1] < ROWS)
)
check("test points fall inside the frame", inside.all(), f"{inside.sum()}/3")

with rasterio.open(ortho_paths["A"]) as ds:
    inv = ~ds.transform
    for i, (east, north, _) in enumerate(test_ground):
        col, row = inv * (east, north)
        in_ortho = 0 <= col < ds.width and 0 <= row < ds.height
        if not in_ortho:
            continue
        window_value = ds.read(
            1, window=rasterio.windows.Window(int(col) - 1, int(row) - 1, 3, 3)
        )
        expected_dn = ground_pattern(np.array([east]), np.array([north]))[0]
        got = float(np.median(window_value[window_value > 0])) if (window_value > 0).any() else np.nan
        ok = np.isfinite(got) and abs(got - expected_dn) < 40
        check(f"ground point {i + 1} lands correctly", ok,
              f"expected {expected_dn:.0f} DN, read {got:.0f} DN")


print("\n=== 5. Mosaicking ===")
spec = MosaicSpec(
    inputs=list(ortho_paths.values()),
    output_path=str(work / "mosaic.tif"),
    color_balance="linear",
    cutline_method="distance",
    blend_width_px=30.0,
    resampling="bilinear",
)

preview = mosaic_preview(spec, max_px=400)
check("preview renders", len(preview["previewPng"]) > 1000 and len(preview["seamPng"]) > 500,
      f"{preview['width']}x{preview['height']} preview, order {preview['order']}")

started = time.time()
mosaic_result = generate_mosaic(spec)
print(f"  mosaic.tif  {mosaic_result.width}x{mosaic_result.height}  "
      f"{time.time() - started:.1f}s")

with rasterio.open(mosaic_result.output_path) as ds:
    mosaic_data = ds.read(1)
    mosaic_transform = ds.transform

covered = mosaic_data > 0
check("mosaic covers more ground than one ortho",
      covered.sum() > valid.sum() * 1.5,
      f"{covered.sum():,} vs {valid.sum():,} pixels")

rows_m, cols_m = np.nonzero(covered)
sample_m = np.random.default_rng(9).choice(len(rows_m), size=min(4000, len(rows_m)), replace=False)
rm, cm = rows_m[sample_m], cols_m[sample_m]
em, nm = mosaic_transform * (cm + 0.5, rm + 0.5)
expected_m = ground_pattern(np.asarray(em), np.asarray(nm))
actual_m = mosaic_data[rm, cm].astype(float)
corr_m = float(np.corrcoef(actual_m, expected_m)[0, 1])

check("mosaic matches the true ground pattern", corr_m > 0.90,
      f"correlation {corr_m:.4f}")

# Seams should not be visible as a sharp discontinuity.
gradient = np.abs(np.diff(mosaic_data.astype(float), axis=1))
interior = gradient[:, 10:-10]
big_jumps = float(np.mean(interior > 110))
check("no hard seam artefacts", big_jumps < 0.02,
      f"{big_jumps:.2%} of adjacent pixels jump more than 110 DN")


print("\n=== 6. Project store: autosave, journal, recovery ===")
bundle_dir = work / "Demo.fidu"
project = Project.create(bundle_dir, "Pipeline demo")
project.mutate("camera", lambda s: s.update({"camera": camera.to_dict()}))
project.mutate("proj", lambda s: s["projection"].update(
    {"output": crs, "gcpSource": crs, "pixelSpacingX": GSD, "pixelSpacingY": GSD}))

for name, path in image_paths.items():
    def add(state, path=path, name=name):
        from fiducia.project import _blank_image
        entry = _blank_image(path)
        entry["name"] = f"frame_{name}"
        entry["exterior"] = stations[name].tolist()
        entry["storedPath"] = project.store_path(path)
        state["images"].append(entry)
    project.mutate("add", add)

project.flush()
check("project.json written", (bundle_dir / "project.json").exists())
check("journal written", any((bundle_dir / "journal").glob("*.jsonl")))
check("snapshot written", any((bundle_dir / "snapshots").glob("*.json")))

# Simulate a crash: mutate without flushing, then reopen.
project.mutate("late", lambda s: s.update({"description": "written just before the crash"}))
project._timer.cancel()          # kill the pending autosave, as a crash would
project._journal_handle.flush()

reopened = Project.open(bundle_dir)
check("recovers the unsaved edit from the journal",
      reopened.state.get("description") == "written just before the crash",
      f"recovered {reopened.state.get('_recoveredOperations', 0)} operation(s)")

check("stores relative paths for portability",
      all(not Path(i["storedPath"]).is_absolute() for i in reopened.state["images"]),
      str([i["storedPath"] for i in reopened.state["images"]][:1]))

# Simulate the drive-letter change: move the images, confirm relink finds them.
moved = work / "relocated"
moved.mkdir(exist_ok=True)
# Release cached GDAL handles first, the way closing a project does — otherwise
# Fiducia itself would be holding a lock on the operator's own files.
raster.close_all()
for name, path in image_paths.items():
    shutil.copy(path, moved / Path(path).name)
    Path(path).unlink()

reopened.state["searchRoots"] = [str(moved)]
statuses = reopened.relink_all()
check("relinks images that moved on disk",
      all(s.online for s in statuses),
      f"{sum(1 for s in statuses if s.online)}/{len(statuses)} back online")


print("\n=== 7. Reports ===")
reopened.state["model"] = {
    "method": "bundle", "converged": True, "iterations": 12, "sigma0": 0.0081,
    "rmsImageMm": 0.0079, "rmsControlM": 0.42, "rmsCheckM": 0.61,
    "degreesOfFreedom": 96, "message": "Converged",
    "solvedAt": "2026-09-12 14:00:00", "residuals": {},
}
text = project_report(reopened.state)
check("project report includes the camera", "CAMERA CALIBRATION" in text and "100.0000" in text,
      f"{len(text)} chars")
check("project report includes exterior orientation", "EXTERIOR ORIENTATION" in text)
check("project report names the app", "FIDUCIA" in text)

residual_text = residual_report(reopened.state, units="ground")
check("residual report renders", "RESIDUAL REPORT" in residual_text)

reopened.close()

print("\n" + "=" * 64)
print(f"  {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("  FAILED: " + ", ".join(FAIL))
print(f"  artefacts in {work}")
print("=" * 64)
sys.exit(1 if FAIL else 0)
