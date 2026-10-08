"""Validation for the refraction and earth-curvature corrections.

Neither correction is checked against its own formula -- that would only prove
the code matches itself. Each is checked against an independent simulation of
the physics it models:

* refraction: rays traced through the International Standard Atmosphere with
  Snell's law for a layered medium, and the resulting apparent image
  positions resected with and without the correction;
* curvature: ground points placed on a true sphere and photographed with
  exact 3-D geometry, then resected in flat map coordinates with and without
  the correction.
"""

import math
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))

from fiducia import corrections                 # noqa: E402
from fiducia.resection import resect            # noqa: E402

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


FOCAL = 153.0


# -- an independent ray tracer ------------------------------------------------
def n_of(z):
    T = 288.15 - 0.0065 * z
    P = 1013.25 * (T / 288.15) ** 5.25588
    return 1 + 77.6 * P / T * 1e-6


def horizontal_run(theta, h, H, steps=3000):
    invariant = n_of(H) * math.sin(theta)
    zs = np.linspace(h, H, steps + 1)
    mids = 0.5 * (zs[:-1] + zs[1:])
    s = invariant / n_of(mids)
    return float(np.sum(np.diff(zs) * s / np.sqrt(1 - s * s)))


def apparent_angle(d, h, H):
    """Off-nadir angle at which a camera at H sees a point at distance d, height h."""
    beta = math.atan2(d, H - h)
    if d < 1e-9:
        return 0.0
    return brentq(lambda t: horizontal_run(t, h, H) - d, beta * 0.95, beta * 1.05 + 1e-9)


print("\n=== 1. The refraction constant against the ray tracer ===")
worst = 0.0
for H, h in ((1500, 0), (3000, 0), (3000, 500), (6000, 0), (9000, 1500)):
    beta = math.radians(25)
    traced = (apparent_angle((H - h) * math.tan(beta), h, H) - beta) / math.tan(beta)
    model = corrections.refraction_constant(H, h)
    worst = max(worst, abs(model / traced - 1))
    print(f"  H {H:>5} m, ground {h:>4} m:  traced {traced * 1e6:6.2f} urad, Fiducia {model * 1e6:6.2f} urad")
check("K matches ray tracing through the standard atmosphere", worst < 0.005, f"worst {worst:.2%}")

# -- a vertical photo over a grid of ground points ----------------------------
rng = np.random.default_rng(11)
H, GROUND = 4000.0, 150.0
spread = (H - GROUND) * 0.65           # a full 23 cm frame: corners at r ~ 140 mm
grid = [(x, y) for x in np.linspace(-spread, spread, 5) for y in np.linspace(-spread, spread, 5)]
heights = np.array([GROUND + rng.uniform(-20, 20) for _ in grid])
CAMERA = np.array([10000.0, 20000.0, H, 0.0, 0.0, 0.0])


def film_straight(points):
    rel = points - CAMERA[:3]
    return np.column_stack([-FOCAL * rel[:, 0] / rel[:, 2], -FOCAL * rel[:, 1] / rel[:, 2]])


def resect_error(film, ground, refraction, curvature):
    fixed = corrections.to_flat(film, FOCAL, H, ground[:, 2], refraction, curvature)
    solved = resect(fixed, ground, FOCAL, initial=CAMERA + np.r_[3, -3, 5, 0.001, -0.001, 0.002])
    return float(np.linalg.norm(solved.eo[:3] - CAMERA[:3])), float(solved.rms_mm * 1000)


print("\n=== 2. Refraction, end to end ===")
ground = np.array([[CAMERA[0] + x, CAMERA[1] + y, z] for (x, y), z in zip(grid, heights)])
straight = film_straight(ground)
refracted = []
for (x, y), z, f in zip(grid, heights, straight):
    d = math.hypot(x, y)
    if d < 1e-9:
        refracted.append(f)
        continue
    r_true = FOCAL * math.tan(apparent_angle(d, z, H))
    refracted.append(f / np.hypot(*f) * r_true)
refracted = np.array(refracted)
shift_um = (np.hypot(*refracted.T) - np.hypot(*straight.T)).max() * 1000
print(f"  refraction moves the outermost point outward by {shift_um:.2f} um")
without = resect_error(refracted, ground, False, False)
with_ = resect_error(refracted, ground, True, False)
print(f"  resected without correction: camera off by {without[0]:.3f} m, residual {without[1]:.3f} um")
print(f"  resected with correction:    camera off by {with_[0]:.3f} m, residual {with_[1]:.3f} um")
check("refraction is displaced outward, as the correction assumes", shift_um > 0)
check("with the correction the camera is recovered", with_[0] < without[0] / 10 and with_[0] < 0.05,
      f"{with_[0]:.4f} m vs {without[0]:.3f} m")

print("\n=== 3. Earth curvature, on a true sphere ===")
R = corrections.EARTH_RADIUS_M


def on_sphere(e, n, h):
    """Map coordinates (distance along the sphere, height above it) -> exact
    local Cartesian, origin on the sphere under the camera."""
    s = math.hypot(e, n)
    if s < 1e-12:
        return np.array([0.0, 0.0, h])
    rho = s / R
    horizontal = (R + h) * math.sin(rho)
    vertical = (R + h) * math.cos(rho) - R
    return np.array([horizontal * e / s, horizontal * n / s, vertical])


map_points = np.array([[x, y, z] for (x, y), z in zip(grid, heights)])
true3d = np.array([on_sphere(*p) for p in map_points])
rel = true3d - np.array([0.0, 0.0, H])
curved_film = np.column_stack([-FOCAL * rel[:, 0] / rel[:, 2], -FOCAL * rel[:, 1] / rel[:, 2]])
flat_film = np.column_stack([-FOCAL * map_points[:, 0] / (map_points[:, 2] - H),
                             -FOCAL * map_points[:, 1] / (map_points[:, 2] - H)])
inward = (np.hypot(*flat_film.T) - np.hypot(*curved_film.T)).max() * 1000
print(f"  curvature moves the outermost point inward by {inward:.2f} um")
# In map coordinates the camera is at the origin.
ground_map = map_points + np.array([CAMERA[0], CAMERA[1], 0.0])
without = resect_error(curved_film, ground_map, False, False)
with_ = resect_error(curved_film, ground_map, False, True)
print(f"  resected without correction: camera off by {without[0]:.3f} m, residual {without[1]:.3f} um")
print(f"  resected with correction:    camera off by {with_[0]:.3f} m, residual {with_[1]:.3f} um")
check("curvature is displaced inward, as the correction assumes", inward > 0)
check("with the correction the residual falls to near zero", with_[1] < 0.05 * without[1],
      f"{with_[1]:.3f} um vs {without[1]:.3f} um")
# Ground at 150 m leaves a small position offset that is not curvature: map
# coordinates reduce horizontal distances to the datum, so points h above it
# are h/R (24 ppm here) short of their true separation -- the height scale
# factor, a separate map-projection effect. With the ground at sea level it
# vanishes and the curvature correction must recover the camera exactly.
sea = np.array([[x, y, 0.0] for (x, y) in grid])
sea_film = np.array([on_sphere(*p) for p in sea]) - np.array([0.0, 0.0, H])
sea_film = np.column_stack([-FOCAL * sea_film[:, 0] / sea_film[:, 2], -FOCAL * sea_film[:, 1] / sea_film[:, 2]])
sea_ground = sea + np.array([CAMERA[0], CAMERA[1], 0.0])
sea_without = resect_error(sea_film, sea_ground, False, False)
sea_with = resect_error(sea_film, sea_ground, False, True)
print(f"  ground at sea level: camera off by {sea_without[0]:.3f} m uncorrected, {sea_with[0]:.4f} m corrected")
check("at sea level the corrected camera is exact", sea_with[0] < 0.005 and sea_without[0] > 0.3,
      f"{sea_with[0]:.4f} m vs {sea_without[0]:.3f} m")
print(f"  at {GROUND:.0f} m the remaining {with_[0]:.3f} m is the height scale factor "
      f"({GROUND / corrections.EARTH_RADIUS_M * 1e6:.0f} ppm), not curvature")

print("\n=== 4. The inverse, for the orthorectifier ===")
film = rng.uniform(-110, 110, (500, 2))
back = corrections.from_flat(corrections.to_flat(film, FOCAL, H, GROUND, True, True), FOCAL, H, GROUND, True, True)
check("from_flat undoes to_flat", np.max(np.abs(back - film)) < 1e-7, f"{np.max(np.abs(back - film)):.1e} mm")
check("both off changes nothing",
      np.array_equal(corrections.to_flat(film, FOCAL, H, GROUND, False, False), film))

print("\n=== 5. Through the engine: adjustment, report and orthophoto ===")
import tempfile  # noqa: E402
import time      # noqa: E402

import rasterio  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402
from fiducia.camera import CameraModel  # noqa: E402
from fiducia.collinearity import project_points  # noqa: E402

work = Path(tempfile.mkdtemp(prefix="fiducia-corrections-"))
COLS, ROWS, PITCH, F2, H2 = 4000, 3000, 0.03, 60.0, 4200.0
cam = CameraModel(kind="digital", name="Wide", focal_mm=F2, pixel_pitch_mm=PITCH, columns=COLS, rows=ROWS,
                  apply_atmospheric=True, apply_earth_curvature=True)
truth = np.array([30000.0, -3740000.0, H2, 0.004, -0.003, 0.3])
pixels = [(c, r) for c in (150, 1000, 2000, 3000, 3850) for r in (150, 1500, 2850)]
film_flat = []
ground_pts = []
for c, r in pixels:
    f = cam.pixel_to_film([[c, r]])[0]
    d = np.array([f[0], f[1], -F2]) @ np.eye(3)
    from fiducia.collinearity import rotation_matrix as _rm  # noqa: E402
    direction = _rm(*truth[3:]).T @ np.array([f[0], f[1], -F2])
    z = 180.0 + 15 * math.sin(c / 700.0)
    t = (z - truth[2]) / direction[2]
    ground_pts.append(truth[:3] + t * direction)
ground_pts = np.array(ground_pts)
flat = project_points(ground_pts, truth, F2)
recorded = corrections.from_flat(flat, F2, H2, ground_pts[:, 2], True, True)
recorded_px = cam.film_to_pixel(recorded)

frame = work / "wide.tif"
with rasterio.open(frame, "w", driver="GTiff", width=COLS, height=ROWS, count=1, dtype="uint8",
                   tiled=True, blockxsize=256, blockysize=256) as ds:
    ds.write(np.full((1, ROWS, COLS), 120, dtype="uint8"))

client = TestClient(server.app)
client.post("/project/new", json={"directory": str(work / "Corrections"), "name": "Corrections"})
client.patch("/project", json={"projection": {"output": "ZALO19_EN", "gcpSource": "ZALO19_EN"}})
client.post("/camera", json=cam.to_dict())
image_id = client.post("/images/add", json={"paths": [str(frame)]}).json()["added"][0]["id"]
for i, (p, px) in enumerate(zip(ground_pts, recorded_px)):
    client.post("/points/gcp", json={"id": f"G{i}", "x": p[0], "y": p[1], "z": p[2],
                                     "measurements": [{"imageId": image_id, "col": px[0], "row": px[1]}]})


def solve_model():
    job = client.post("/model/compute", json={}).json()["job"]
    for _ in range(200):
        state = next(j for j in client.get("/jobs").json()["jobs"] if j["id"] == job["id"])
        if state["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    return client.get("/project").json()["state"]["model"]


on = solve_model()
error_on = float(np.linalg.norm(np.array(on["exterior"][image_id][:3]) - truth[:3]))
check("with corrections on, the engine recovers the camera", error_on < 0.02 and on["corrections"]["curvature"],
      f"{error_on:.4f} m, corrections {on['corrections']}")
report = client.post("/reports/project", json={}).text
check("the report says they were applied", "Earth curvature          : applied" in report
      and "Atmospheric refraction   : applied" in report)

client.post("/camera", json={**cam.to_dict(), "applyAtmospheric": False, "applyEarthCurvature": False})
off = solve_model()
error_off = float(np.linalg.norm(np.array(off["exterior"][image_id][:3]) - truth[:3]))
check("with them off, the same data solves measurably worse", error_off > 5 * error_on,
      f"{error_off:.3f} m vs {error_on:.4f} m")
report = client.post("/reports/project", json={}).text
check("and the report says not applied", "Earth curvature          : not applied" in report)

client.post("/camera", json=cam.to_dict())   # ticked again, not yet re-solved
report = client.post("/reports/project", json={}).text
check("ticked but not re-solved is reported as such, not as applied",
      "requested, but the model was solved without it" in report)

solve_model()
# One worker: this script has no __main__ guard, and spawned workers would
# re-run it. The pooled path is covered by test_memory_budget.py.
job = client.post("/ortho/generate", json={"imageIds": [image_id], "pixelSizeX": 10.0,
                                          "pixelSizeY": 10.0, "maxWorkers": 1}).json()["job"]
for _ in range(600):
    state = next(j for j in client.get("/jobs").json()["jobs"] if j["id"] == job["id"])
    if state["status"] in ("done", "failed"):
        break
    time.sleep(0.1)
check("the orthophoto runs with the corrections applied", state["status"] == "done",
      state.get("error") or "done")
client.post("/project/close")

print("\n" + "=" * 64)
print(f"  TOTAL {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("  FAILED: " + ", ".join(FAIL))
print("=" * 64)
sys.exit(1 if FAIL else 0)
