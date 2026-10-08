"""Build a populated demo project so the interface can be seen doing real work.

Synthesises a three-photo strip over patterned ground, measures control on it,
solves the block and orthorectifies — then leaves the project open on the
running engine.
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from urllib import error, request

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parent.parent
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8731
BASE = f"http://127.0.0.1:{PORT}"
DEMO = Path(os.environ.get("FIDUCIA_DEMO_DIR") or Path(tempfile.gettempdir()) / "fiducia-demo")

sys.path.insert(0, str(ROOT / "engine"))
from fiducia.camera import CameraModel                            # noqa: E402
from fiducia.collinearity import project_points, rotation_matrix  # noqa: E402


def call(method, path, payload=None, raw=False):
    data = json.dumps(payload).encode() if payload is not None else None
    req = request.Request(BASE + path, data=data, method=method,
                          headers={"Content-Type": "application/json"} if data else {})
    try:
        with request.urlopen(req, timeout=300) as response:
            body = response.read().decode()
            return body if raw else (json.loads(body) if body else None)
    except error.HTTPError as exc:
        raise RuntimeError(f"{method} {path}: {exc.read().decode()[:300]}") from None


def wait(job_id, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        jobs = call("GET", "/jobs?limit=60")["jobs"]
        job = next((j for j in jobs if j["id"] == job_id), None)
        if job and job["status"] in ("done", "failed", "cancelled"):
            if job["status"] != "done":
                raise RuntimeError(f"{job['label']}: {job.get('error')}")
            return job
        time.sleep(0.4)
    raise TimeoutError(job_id)


FOCAL, PITCH, COLS, ROWS, GSD = 153.690, 0.015, 1500, 1500, 0.5
HEIGHT = (FOCAL / 1000.0) * (GSD / (PITCH / 1000.0))
ORIGIN_E, ORIGIN_N = 76500.0, 3760000.0   # plausible Lo19 / Western Cape
CRS = "ZALO19"


def ground_pattern(east, north):
    """Something that reads as landscape rather than a test chart."""
    u = (east - ORIGIN_E) / 90.0
    v = (north - ORIGIN_N) / 90.0
    fields = ((np.floor(u * 1.4) + np.floor(v * 1.1)) % 3) * 38 + 58
    river = 70 * np.exp(-((v + 0.35 * np.sin(u * 0.8)) ** 2) / 0.06)
    texture = 16 * np.sin(u * 7.3 + v * 5.1) + 10 * np.cos(u * 13.7 - v * 9.4)
    roads = 45 * np.exp(-((u % 3.1 - 1.55) ** 2) / 0.004)
    return np.clip(fields + texture + roads - river, 5, 250)


DEMO.mkdir(parents=True, exist_ok=True)
for stale in DEMO.glob("*"):
    if stale.is_file():
        stale.unlink()
    else:
        import shutil
        shutil.rmtree(stale, ignore_errors=True)

camera = CameraModel(
    kind="digital", name="Generic 153mm frame camera",
    focal_mm=FOCAL, pixel_pitch_mm=PITCH, columns=COLS, rows=ROWS,
    image_scale=20000.0,
)

base = 420.0
stations = {
    "demo_strip_01": np.array([ORIGIN_E, ORIGIN_N, HEIGHT, 0.004, -0.003, 0.012]),
    "demo_strip_02": np.array([ORIGIN_E + base, ORIGIN_N + 12, HEIGHT + 9, -0.003, 0.005, -0.008]),
    "demo_strip_03": np.array([ORIGIN_E + 2 * base, ORIGIN_N - 8, HEIGHT - 6, 0.005, 0.002, 0.016]),
}

print("Synthesising imagery…")
paths = {}
for name, eo in stations.items():
    cols, rows = np.meshgrid(np.arange(COLS), np.arange(ROWS))
    film = camera.pixel_to_film(np.column_stack([cols.ravel(), rows.ravel()]))
    rot = rotation_matrix(eo[3], eo[4], eo[5])
    directions = np.column_stack(
        [film[:, 0], film[:, 1], np.full(film.shape[0], -FOCAL)]) @ rot
    t = (0.0 - eo[2]) / directions[:, 2]
    grey = ground_pattern(eo[0] + t * directions[:, 0],
                          eo[1] + t * directions[:, 1]).reshape(ROWS, COLS)

    # Three bands with a mild colour cast that differs per frame, so the
    # mosaic's colour balancing has something real to correct.
    cast = {"demo_strip_01": (1.08, 1.0, 0.9),
            "demo_strip_02": (1.0, 1.02, 1.0),
            "demo_strip_03": (0.94, 0.99, 1.1)}[name]
    stack = np.stack([np.clip(grey * c, 0, 255) for c in cast]).astype(np.uint8)

    path = DEMO / f"{name}.tif"
    with rasterio.open(path, "w", driver="GTiff", width=COLS, height=ROWS, count=3,
                       dtype="uint8", compress="DEFLATE", tiled=True,
                       blockxsize=256, blockysize=256) as sink:
        sink.write(stack)
    paths[name] = str(path)
    print(f"  {name}.tif")

print("\nConfiguring the project…")
bundle = DEMO / "Demo_strip.fidu"
try:
    call("POST", "/project/close")
except Exception:
    pass

call("POST", "/project/new", {
    "directory": str(bundle),
    "name": "Demo strip",
    "description": "Three-photo strip — bundle adjustment, orthos and mosaic",
})
call("PATCH", "/project", {"mathModel": {"kind": "aerial_digital",
                                         "exteriorSource": "gcp_tiepoints"}})
call("PATCH", "/project", {"projection": {
    "output": CRS, "gcpSource": CRS,
    "pixelSpacingX": GSD, "pixelSpacingY": GSD,
    "elevationReference": "mean_sea_level"}})
call("POST", "/camera", camera.to_dict())

added = call("POST", "/images/add", {"paths": list(paths.values())})
ids = {Path(i["path"]).stem: i["id"] for i in added["state"]["images"]}
print(f"  {len(ids)} images added")

print("\nMeasuring control...")
rng = np.random.default_rng(17)

# Place control against each photo's real footprint rather than scattering it
# over a guessed extent. Every photo needs at least four measurements, and a
# strip is only well conditioned if the end frames are tied down too.
half_ground = (COLS / 2.0) * GSD          # metres from photo centre to edge

targets = []
for name, eo in stations.items():
    cx, cy = eo[0], eo[1]
    for fx, fy in ((-0.55, -0.55), (0.55, -0.55), (-0.55, 0.55),
                   (0.55, 0.55), (0.0, 0.0)):
        targets.append((cx + fx * half_ground, cy + fy * half_ground))

# Thin near-duplicates where consecutive footprints overlap.
control = []
for east, north in targets:
    if any(abs(east - e) < 90 and abs(north - n) < 90 for e, n, _ in control):
        continue
    control.append((east, north, float(rng.uniform(40, 120))))

placed = 0
for point in control:
    point = np.asarray(point)
    measurements = []
    for name, eo in stations.items():
        film = project_points(point.reshape(1, 3), eo, FOCAL)
        pixel = camera.film_to_pixel(film)[0]
        if 25 <= pixel[0] < COLS - 25 and 25 <= pixel[1] < ROWS - 25:
            measurements.append({
                "imageId": ids[name],
                "col": float(pixel[0] + rng.normal(0, 0.4)),
                "row": float(pixel[1] + rng.normal(0, 0.4)),
            })
    if not measurements:
        continue
    call("POST", "/points/gcp", {
        "x": float(point[0]), "y": float(point[1]), "z": float(point[2]),
        "measurements": measurements,
    })
    placed += 1

from collections import Counter
state = call("GET", "/project")["state"]
per_image = Counter(o["imageId"] for o in state["observations"])
print(f"  {placed} control points")
for name, iid in ids.items():
    print(f"    {name}: {per_image[iid]} measurements")

readiness = call("GET", "/model/readiness")
if not readiness["ready"]:
    print("  BLOCKED:", [b["message"] for b in readiness["blockers"]])
    sys.exit(1)

# One check point, so the interface has an independent accuracy figure.
if len(state["gcps"]) >= 6:
    call("POST", "/model/checkpoint",
         {"pointId": state["gcps"][-1]["id"], "isCheckPoint": True})

print("\nSolving the block…")
wait(call("POST", "/model/compute", {})["job"]["id"])
model = call("GET", "/project")["state"]["model"]
print(f"  converged={model['converged']}  sigma0={model['sigma0']:.5f}  "
      f"image RMS={model['rmsImageMm'] * 1000:.1f} um")

print("\nOrthorectifying…")
wait(call("POST", "/ortho/generate", {"resampling": "bilinear"})["job"]["id"])
print(f"  {len(call('GET', '/project')['state']['orthos'])} orthos")

print("\nMosaicking…")
wait(call("POST", "/mosaic/generate", {"colorBalance": "linear",
                                       "cutlineMethod": "distance",
                                       "blendWidth": 40})["job"]["id"])

summary = call("GET", "/health")["project"]
print(f"\nDemo project ready: {bundle}")
print(f"  {summary['imageCount']} images · {summary['gcpCount']} control · "
      f"{summary['checkPointCount']} check · {summary['orthoCount']} orthos")
