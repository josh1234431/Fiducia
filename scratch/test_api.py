"""Drive the whole workflow over HTTP, exactly as the interface does.

Creates a project, sets a camera and projection, adds synthetic imagery,
measures control, solves the model, orthorectifies and mosaics -- all through
the REST API, so the server layer is validated and not just the library.
"""

import json
import shutil
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib import error, request

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parent.parent
PY = Path(sys.executable)
PORT = 8791
BASE = f"http://127.0.0.1:{PORT}"

sys.path.insert(0, str(ROOT / "engine"))
from fiducia.camera import CameraModel                    # noqa: E402
from fiducia.collinearity import project_points, rotation_matrix  # noqa: E402

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def call(method, path, payload=None, raw=False):
    data = json.dumps(payload).encode() if payload is not None else None
    req = request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with request.urlopen(req, timeout=180) as response:
            body = response.read().decode()
            return body if raw else (json.loads(body) if body else None)
    except error.HTTPError as exc:
        detail = exc.read().decode()
        raise RuntimeError(f"{method} {path} -> {exc.code}: {detail[:400]}") from None


def wait_for_job(job_id, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        jobs = call("GET", "/jobs?limit=60")["jobs"]
        job = next((j for j in jobs if j["id"] == job_id), None)
        if job and job["status"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.4)
    raise TimeoutError(f"job {job_id} did not finish")


# -- synthetic survey ------------------------------------------------------

work = Path(tempfile.mkdtemp(prefix="fiducia-api-"))
FOCAL, PITCH, COLS, ROWS, GSD = 100.0, 0.006, 1100, 850, 0.30
HEIGHT = (FOCAL / 1000.0) * (GSD / (PITCH / 1000.0))
ORIGIN_E, ORIGIN_N = 40000.0, 6100000.0
CRS = "ZALO19_WGS84"


def pattern(east, north):
    u = (east - ORIGIN_E) / 28.0
    v = (north - ORIGIN_N) / 28.0
    return np.clip(((np.floor(u) + np.floor(v)) % 2) * 95 + 45
                   + 50 * np.sin(u * 1.7) * np.cos(v * 1.3)
                   + 20 * np.sin(u * 8.5 + v * 6.7), 0, 255)


camera = CameraModel(kind="digital", name="API test frame", focal_mm=FOCAL,
                     pixel_pitch_mm=PITCH, columns=COLS, rows=ROWS)

stations = {
    "A": np.array([ORIGIN_E, ORIGIN_N, HEIGHT, 0.003, -0.002, 0.01]),
    "B": np.array([ORIGIN_E + 170, ORIGIN_N + 5, HEIGHT + 4, -0.002, 0.004, -0.008]),
}

print(f"\nWorking in {work}\n")
print("=== Synthesising imagery ===")
paths = {}
for name, eo in stations.items():
    cols, rows = np.meshgrid(np.arange(COLS), np.arange(ROWS))
    film = camera.pixel_to_film(np.column_stack([cols.ravel(), rows.ravel()]))
    rot = rotation_matrix(eo[3], eo[4], eo[5])
    directions = np.column_stack(
        [film[:, 0], film[:, 1], np.full(film.shape[0], -FOCAL)]) @ rot
    t = (0.0 - eo[2]) / directions[:, 2]
    band = pattern(eo[0] + t * directions[:, 0],
                   eo[1] + t * directions[:, 1]).reshape(ROWS, COLS).astype(np.uint8)
    path = work / f"api_{name}.tif"
    with rasterio.open(path, "w", driver="GTiff", width=COLS, height=ROWS, count=1,
                       dtype="uint8", compress="DEFLATE", tiled=True,
                       blockxsize=256, blockysize=256) as sink:
        sink.write(band, 1)
    paths[name] = str(path)
    print(f"  api_{name}.tif")

# -- start the engine ------------------------------------------------------

print("\n=== Starting the engine ===")
log = open(work / "engine.log", "w")
# FIDUCIA_ENGINE_EXE runs the suite against a packaged engine (fiducia-engine.exe).
engine_exe = os.environ.get("FIDUCIA_ENGINE_EXE")
command = [engine_exe] if engine_exe else [str(PY), str(ROOT / "engine" / "server.py")]
server = subprocess.Popen(command + ["--port", str(PORT)],
                          cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)

ready = False
for _ in range(120):
    try:
        if call("GET", "/health")["ok"]:
            ready = True
            break
    except Exception:
        time.sleep(0.5)
check("engine responds to /health", ready)
if not ready:
    server.terminate()
    sys.exit(1)

try:
    # -- project ----------------------------------------------------------
    print("\n=== Project ===")
    bundle = work / "ApiTest.fidu"
    result = call("POST", "/project/new",
                  {"directory": str(bundle), "name": "API round trip"})
    check("project created", result["summary"]["name"] == "API round trip")
    check("bundle exists on disk", (bundle / "project.json").exists())

    call("PATCH", "/project", {"mathModel": {"kind": "aerial_digital",
                                             "exteriorSource": "gcp_tiepoints"}})
    call("PATCH", "/project", {"projection": {
        "output": CRS, "gcpSource": CRS,
        "pixelSpacingX": GSD, "pixelSpacingY": GSD,
        "elevationReference": "mean_sea_level"}})

    described = call("GET", f"/projections/describe?identifier={CRS}")
    check("LO19 resolves with the right datum",
          "Lo19" in described["label"] and described["isProjected"],
          f"{described['label']} / {described['datum']}")

    # -- camera -----------------------------------------------------------
    print("\n=== Camera ===")
    call("POST", "/camera", camera.to_dict())
    state = call("GET", "/project")["state"]
    check("camera stored", abs(state["camera"]["focalMm"] - FOCAL) < 1e-9)

    fit = call("POST", "/camera/distortion-table",
               {"radii": [20, 40, 60, 80, 100],
                "distortions": [1.2, 2.6, 3.1, 1.9, -2.4]})
    check("distortion table fits", fit["rmsUm"] < 5.0,
          f"RMS {fit['rmsUm']:.3f} um over {fit['termsFitted']} terms")

    # -- images -----------------------------------------------------------
    print("\n=== Images ===")
    added = call("POST", "/images/add", {"paths": list(paths.values())})
    check("both images added", len(added["added"]) == 2)
    ids = {Path(i["path"]).stem: i["id"] for i in added["state"]["images"]}
    check("dimensions read back",
          all(i["width"] == COLS for i in added["state"]["images"]))

    tile = request.urlopen(
        f"{BASE}/images/{ids['api_A']}/tile/0/0/0.png?enhancement=linear2pct", timeout=60)
    body = tile.read()
    check("tile endpoint serves PNG",
          tile.status == 200 and body[:8] == b"\x89PNG\r\n\x1a\n", f"{len(body)} bytes")

    # -- control ----------------------------------------------------------
    print("\n=== Control points ===")
    ground = np.array([
        [ORIGIN_E - 55, ORIGIN_N - 40, 0.0],
        [ORIGIN_E + 60, ORIGIN_N + 45, 0.0],
        [ORIGIN_E + 30, ORIGIN_N - 50, 0.0],
        [ORIGIN_E - 20, ORIGIN_N + 55, 0.0],
        [ORIGIN_E + 95, ORIGIN_N - 10, 0.0],
    ])

    placed = 0
    for index, point in enumerate(ground):
        measurements = []
        for name, eo in stations.items():
            film = project_points(point.reshape(1, 3), eo, FOCAL)
            pixel = camera.film_to_pixel(film)[0]
            if 0 <= pixel[0] < COLS and 0 <= pixel[1] < ROWS:
                measurements.append({"imageId": ids[f"api_{name}"],
                                     "col": float(pixel[0]), "row": float(pixel[1])})
        if not measurements:
            continue
        call("POST", "/points/gcp", {
            "x": float(point[0]), "y": float(point[1]), "z": float(point[2]),
            "measurements": measurements,
        })
        placed += 1

    check("control points accepted", placed >= 4, f"{placed} placed")

    readiness = call("GET", "/model/readiness")
    check("readiness reports ready", readiness["ready"],
          f"blockers: {[b['message'] for b in readiness['blockers']]}")

    # -- model ------------------------------------------------------------
    print("\n=== Model ===")
    job = wait_for_job(call("POST", "/model/compute", {})["job"]["id"])
    check("adjustment completes", job["status"] == "done", job.get("error") or "")

    state = call("GET", "/project")["state"]
    model = state["model"]
    check("model converged", model["converged"], model["message"])

    worst = 0.0
    for name, eo in stations.items():
        solved = np.asarray(model["exterior"][ids[f"api_{name}"]])
        worst = max(worst, float(np.linalg.norm(solved[:3] - eo[:3])))
    check("recovers camera stations", worst < 2.0, f"worst {worst:.3f} m")

    residuals = call("GET", "/model/residuals?units=ground&show=all")["rows"]
    check("residual table populated", len(residuals) > 0,
          f"{len(residuals)} rows, worst {max(r['magnitude'] for r in residuals):.3f} m")

    # -- ortho ------------------------------------------------------------
    print("\n=== Ortho ===")
    footprint = call("POST", "/ortho/footprint", {"imageId": ids["api_A"]})
    check("footprint estimated", footprint["width"] > 100,
          f"{footprint['width']} x {footprint['height']}")

    job = wait_for_job(call("POST", "/ortho/generate",
                            {"resampling": "bilinear"})["job"]["id"])
    check("orthos generated", job["status"] == "done", job.get("error") or "")

    state = call("GET", "/project")["state"]
    check("orthos recorded in the project", len(state["orthos"]) == 2)

    ortho_path = state["orthos"][0]["path"]
    with rasterio.open(ortho_path) as ds:
        data = ds.read(1)
        rows_v, cols_v = np.nonzero(data > 0)
        pick = np.random.default_rng(3).choice(len(rows_v), size=min(2500, len(rows_v)),
                                               replace=False)
        east, north = ds.transform * (cols_v[pick] + 0.5, rows_v[pick] + 0.5)
        corr = float(np.corrcoef(data[rows_v[pick], cols_v[pick]].astype(float),
                                 pattern(np.asarray(east), np.asarray(north)))[0, 1])
    check("ortho is geometrically correct", corr > 0.93, f"correlation {corr:.4f}")

    # -- mosaic -----------------------------------------------------------
    print("\n=== Mosaic ===")
    preview = call("POST", "/mosaic/preview", {"colorBalance": "linear"})
    check("preview returns images",
          len(preview["previewPng"]) > 500 and len(preview["seamPng"]) > 200,
          f"{preview['width']}x{preview['height']}")

    job = wait_for_job(call("POST", "/mosaic/generate", {"colorBalance": "linear"})["job"]["id"])
    check("mosaic generated", job["status"] == "done", job.get("error") or "")

    # -- reports ----------------------------------------------------------
    print("\n=== Reports ===")
    report = call("POST", "/reports/project", {}, raw=True)
    check("project report generated",
          "FIDUCIA" in report and "EXTERIOR ORIENTATION" in report, f"{len(report)} chars")

    residual_report = call("POST", "/reports/residual", {"units": "ground"}, raw=True)
    check("residual report generated", "RESIDUAL REPORT" in residual_report)

    # -- autosave and recovery -------------------------------------------
    print("\n=== Autosave and portability ===")
    call("POST", "/project/snapshot", {"reason": "test"})
    snapshots = call("GET", "/project/snapshots")["snapshots"]
    check("snapshots listed", len(snapshots) >= 2, f"{len(snapshots)} snapshots")

    state = call("GET", "/project")["state"]
    check("image paths stored relatively",
          all(not Path(i["storedPath"]).is_absolute() for i in state["images"]),
          str([i["storedPath"] for i in state["images"]]))

    call("POST", "/project/close")
    reopened = call("POST", "/project/open", {"directory": str(bundle)})
    check("project reopens with everything intact",
          len(reopened["state"]["images"]) == 2
          and reopened["state"]["model"] is not None
          and len(reopened["state"]["orthos"]) == 2,
          f"{len(reopened['state']['gcps'])} GCPs, "
          f"{len(reopened['state']['orthos'])} orthos")

finally:
    server.terminate()
    try:
        server.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server.kill()
    log.close()

print("\n" + "=" * 64)
print(f"  {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("  FAILED: " + ", ".join(FAIL))
    print(f"  engine log: {work / 'engine.log'}")
print("=" * 64)
sys.exit(1 if FAIL else 0)
