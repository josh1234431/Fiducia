"""Validation for the plausibility checks: camera description, solution sanity,
live residual preview, coordinate entry, and projection agreement.

Built around a real failure: a digital frame delivered turned 90 degrees,
described with the sensor's own chip orientation, with the chip dimensions
typed into the principal-point offset fields. The adjustment "converged" on
it with 166 m residuals. Every part below checks that such a setup is now
caught, explained, and fixable, and that a correct setup still solves exactly.

"""

import math
import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))

from fiducia import plausibility                        # noqa: E402
from fiducia.camera import CameraModel                   # noqa: E402
from fiducia.collinearity import project_points, rotation_matrix  # noqa: E402
from fiducia.resection import resect                     # noqa: E402

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


work = Path(tempfile.mkdtemp(prefix="fiducia-reality-"))

# A small digital camera, delivered portrait: the sensor is 600 x 400, the
# file 400 x 600.
PITCH, FOCAL = 0.05, 60.0
FILE_W, FILE_H = 400, 600
TRUE_CAMERA = {"kind": "digital", "name": "Test", "focalMm": FOCAL, "pixelPitchMm": PITCH,
               "columns": FILE_W, "rows": FILE_H}
WRONG_CAMERA = {**TRUE_CAMERA, "columns": FILE_H, "rows": FILE_W,
                "longTrackOffsetMm": FILE_W * PITCH, "crossTrackOffsetMm": FILE_H * PITCH}

# Negative easting/northing, as South African data in a north-oriented Lo
# zone is stored.
EO = np.array([-41970.0, -3742716.0, 1200.0, math.radians(0.6), math.radians(-0.2),
               math.radians(89.5)])
camera_true = CameraModel.from_dict(TRUE_CAMERA)


def ground_at(col, row, z):
    film = camera_true.pixel_to_film([[col, row]])[0]
    direction = rotation_matrix(*EO[3:]).T @ np.array([film[0], film[1], -FOCAL])
    return EO[:3] + (z - EO[2]) / direction[2] * direction


PIXELS = {"G1": (40.0, 520.0), "G2": (350.0, 560.0), "G3": (370.0, 60.0),
          "G4": (60.0, 40.0), "G5": (210.0, 300.0)}
HEIGHTS = {"G1": 57.6, "G2": 73.6, "G3": 4.4, "G4": 10.0, "G5": 30.0}
GROUND = {k: ground_at(*PIXELS[k], HEIGHTS[k]) for k in PIXELS}

# =====================================================================
print("\n=== 1. Camera description against the images ===")
image = {"id": "img_1", "name": "portrait_frame", "online": True, "width": FILE_W, "height": FILE_H}
problems = plausibility.camera_problems(WRONG_CAMERA, [image])
codes = {p["code"]: p for p in problems}
check("a sensor described landscape for a portrait file is a blocker",
      "chip_turned" in codes and codes["chip_turned"]["severity"] == "blocker",
      codes.get("chip_turned", {}).get("message", "")[:110])
check("the fix describes the file as delivered",
      codes.get("chip_turned", {}).get("fix") == {"columns": FILE_W, "rows": FILE_H})
offset = next((p for c, p in codes.items() if c.startswith("offset_")), None)
check("a chip dimension typed as an offset is recognised as such",
      offset is not None and "chip's" in offset["message"], (offset or {}).get("message", "")[:120])
check("one fix clears every implausible offset",
      offset is not None and offset.get("fix") == {"longTrackOffsetMm": 0.0, "crossTrackOffsetMm": 0.0},
      str((offset or {}).get("fix")))
check("a correct camera raises nothing", plausibility.camera_problems(TRUE_CAMERA, [image]) == [])
check("a sub-millimetre offset is left alone",
      plausibility.camera_problems({**TRUE_CAMERA, "ppoXMm": 0.012}, [image]) == [])

# =====================================================================
print("\n=== 2. Judging a solution ===")
ids = list(PIXELS)
ground = np.array([GROUND[k] for k in ids])
film_true = np.array([camera_true.pixel_to_film([PIXELS[k]])[0] for k in ids])
fit = resect(film_true, ground, FOCAL)
verdict = plausibility.judge_solution(fit.eo, ground, fit.rms_mm, PITCH, FOCAL)
check("correct camera, exact data: good", verdict["quality"] == "good" and verdict["ok"],
      f"{verdict['rmsPx']:.2e} px")

camera_wrong = CameraModel.from_dict(WRONG_CAMERA)
film_wrong = np.array([camera_wrong.pixel_to_film([PIXELS[k]])[0] for k in ids])
bad = resect(film_wrong, ground, FOCAL)
verdict = plausibility.judge_solution(bad.eo, ground, bad.rms_mm, PITCH, FOCAL, "portrait_frame")
# The wrong camera is partly absorbed by the orientation, so its residuals
# alone may look merely poor. It must never look good; stopping it outright
# is the camera check's job (part 3).
check("the wrong camera's solution never passes as good", verdict["quality"] != "good",
      f"{verdict['quality']}: {verdict['rmsPx']:.1f} px -- which is why the camera check blocks it")

underground = fit.eo.copy()
underground[2] = -5000.0
verdict = plausibility.judge_solution(underground, ground, fit.rms_mm, PITCH, FOCAL)
check("a camera below the ground is rejected whatever the residuals",
      not verdict["ok"] and "underground" in verdict["problems"][0], verdict["problems"][0][:90])

verdict = plausibility.judge_solution(fit.eo, ground, 11 * PITCH, PITCH, FOCAL)
check("11 px residuals are flagged poor, not failed",
      verdict["ok"] and verdict["quality"] == "poor", verdict["problems"][0][:90])

# =====================================================================
print("\n=== 3. The workflow over HTTP ===")
from fastapi.testclient import TestClient  # noqa: E402
import server                              # noqa: E402

client = TestClient(server.app)
frame = work / "portrait_frame.tif"
with rasterio.open(frame, "w", driver="GTiff", width=FILE_W, height=FILE_H, count=1,
                   dtype="uint8") as ds:
    ds.write(np.random.default_rng(1).integers(0, 255, (1, FILE_H, FILE_W), dtype=np.uint8))

dem = work / "dem.tif"
dem_crs = ("+proj=tmerc +lat_0=0 +lon_0=19 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 "
           "+towgs84=0,0,0,0,0,0,0 +units=m +no_defs")
gx = [g[0] for g in GROUND.values()]
gy = [g[1] for g in GROUND.values()]
west, north = min(gx) - 200, max(gy) + 200
with rasterio.open(dem, "w", driver="GTiff", width=200, height=200, count=1, dtype="float32",
                   crs=dem_crs, transform=from_origin(west, north, 5, 5)) as ds:
    ds.write(np.full((1, 200, 200), 42.0, dtype="float32"))

client.post("/project/new", json={"directory": str(work / "Reality"), "name": "Reality"})
client.patch("/project", json={"projection": {"output": "ZALO15", "gcpSource": "ZALO15"},
                               "dem": {"referencePath": str(dem)}})
client.post("/camera", json=WRONG_CAMERA)
image_id = client.post("/images/add", json={"paths": [str(frame)]}).json()["added"][0]["id"]

readiness = client.get("/model/readiness").json()
blocker_codes = [b["code"] for b in readiness["blockers"]]
check("readiness blocks on the turned chip and the offsets",
      "chip_turned" in blocker_codes and any(c.startswith("offset_") for c in blocker_codes),
      ", ".join(blocker_codes))
check("readiness warns the control projection disagrees with the DEM",
      any(w["code"] == "crs_dem" and "central meridian 15" in w["message"]
          and "axes run" in w["message"] for w in readiness["warnings"]),
      next((w["message"][:120] for w in readiness["warnings"] if w["code"] == "crs_dem"), "none"))

for index, key in enumerate(ids[:3]):
    client.post("/points/gcp", json={"id": key, "x": GROUND[key][0], "y": GROUND[key][1],
                                     "z": GROUND[key][2],
                                     "measurements": [{"imageId": image_id, "col": PIXELS[key][0],
                                                       "row": PIXELS[key][1]}]})
preview = client.post("/model/preview", json={"imageId": image_id}).json()
check("preview waits for the fourth point", not preview["ready"] and "3 of 4" in preview["message"],
      preview["message"])

key = ids[3]
client.post("/points/gcp", json={"id": key, "x": GROUND[key][0], "y": GROUND[key][1],
                                 "z": GROUND[key][2],
                                 "measurements": [{"imageId": image_id, "col": PIXELS[key][0],
                                                   "row": PIXELS[key][1]}]})
preview = client.post("/model/preview", json={"imageId": image_id}).json()
check("with a wrong camera the preview refuses and says why",
      not preview["ready"] and preview["setupProblems"], preview["message"])

# The model step refuses outright, even called directly past the interface.
import time  # noqa: E402
response = client.post("/model/compute", json={})
check("the engine refuses to compute from a camera that contradicts its images",
      response.status_code == 400 and "camera description does not match" in response.json()["detail"]
      and client.get("/project").json()["state"]["model"] is None,
      f"{response.status_code}: {response.json().get('detail', '')[:90]}")

# Apply the offered fixes, exactly as the interface buttons would.
readiness = client.get("/model/readiness").json()
patch = {}
for item in readiness["blockers"]:
    patch.update(item.get("fix") or {})
camera_now = client.get("/project").json()["state"]["camera"]
client.post("/camera", json={**camera_now, **patch})
readiness = client.get("/model/readiness").json()
check("after the offered fixes the camera blockers are gone",
      not any(b["code"] in ("chip_turned", "chip_size") or b["code"].startswith("offset_")
              for b in readiness["blockers"]), str([b["code"] for b in readiness["blockers"]]))

preview = client.post("/model/preview", json={"imageId": image_id}).json()
check("corrected, the preview solves exactly",
      preview["ready"] and preview["quality"] == "good" and preview["rmsPx"] < 1e-3,
      f"rms {preview.get('rmsPx', float('nan')):.2e} px over {preview['count']} points")

# Move one surveyed coordinate 15 m: the preview must name it.
key = ids[4]
moved = GROUND[key] + np.array([15.0, 0.0, 0.0])
client.post("/points/gcp", json={"id": key, "x": moved[0], "y": moved[1], "z": moved[2],
                                 "measurements": [{"imageId": image_id, "col": PIXELS[key][0],
                                                   "row": PIXELS[key][1]}]})
preview = client.post("/model/preview", json={"imageId": image_id}).json()
check("a 15 m coordinate error is visible and named",
      preview["ready"] and preview["worst"] == key and preview["quality"] != "good",
      f"worst {preview['worst']}, rms {preview['rmsGroundM']:.2f} m, "
      + ", ".join(f"{p['pointId']} {p['groundM']:.1f} m" for p in preview["points"]))
before = client.get("/project").json()["state"]
client.post("/model/preview", json={"imageId": image_id})
after = client.get("/project").json()["state"]
check("the preview saved nothing",
      before["model"] == after["model"] and before["modified"] == after["modified"])

# With four points a blunder cannot be isolated; the preview must say so
# rather than point at the wrong one.
client.delete(f"/points/gcp/{ids[0]}")
preview = client.post("/model/preview", json={"imageId": image_id}).json()
check("with four points it declines to name a culprit, and says why",
      preview["worst"] is None and "fifth" in preview["note"], preview["note"][:90])

# =====================================================================
print("\n=== 4. Coordinate entry ===")
# Declared as Lo15 while the data is Lo19 E/N, Extract Z converts from the
# wrong system and misses -- which is correct: the project claims a system
# the numbers are not in, and readiness says so. The user's fix:
x0, y0 = float(GROUND["G1"][0]), float(GROUND["G1"][1])
wrong = client.get("/raster/sample", params={"path": str(dem), "x": x0, "y": y0}).json()
check("with the project in the wrong zone, heights are not read from the wrong place",
      wrong["value"] is None, str(wrong))
client.patch("/project", json={"projection": {"output": "ZALO19_EN", "gcpSource": "ZALO19_EN"}})
readiness = client.get("/model/readiness").json()
check("switching to the zone the data is in clears the warning",
      not any(w["code"] == "crs_dem" for w in readiness["warnings"]),
      str([w["code"] for w in readiness["warnings"]]))
x, y = float(GROUND["G1"][0]), float(GROUND["G1"][1])
inside = client.get("/raster/sample", params={"path": str(dem), "x": x, "y": y}).json()
check("a coordinate on the DEM samples normally", inside["value"] == 42.0, str(inside))
swapped = client.get("/raster/sample", params={"path": str(dem), "x": y, "y": x}).json()
check("a swapped pair is recognised as swapped",
      swapped["value"] is None and swapped["swappedValue"] == 42.0, str(swapped))
nowhere = client.get("/raster/sample", params={"path": str(dem), "x": 0.0, "y": 0.0}).json()
check("a coordinate simply off the DEM is not called swapped",
      nowhere["value"] is None and nowhere["swappedValue"] is None, str(nowhere))
response = client.get("/raster/sample", params={"path": str(dem), "x": "nan", "y": y})
check("a coordinate that is not a number gets a plain message, not a crash",
      response.status_code == 400 and "plain numbers" in response.json()["detail"],
      f"{response.status_code}: {response.json().get('detail', '')[:80]}")

from fiducia.geodesy import resolve_crs  # noqa: E402
from pyproj import Transformer              # noqa: E402
t = Transformer.from_crs(rasterio.crs.CRS.from_string(dem_crs).to_wkt(), resolve_crs("ZALO19_EN"),
                         always_xy=True)
same = np.allclose(t.transform(x, y), (x, y), atol=1e-6)
check("the north-oriented Lo19 preset matches plain TM data point for point", same)
check("and matches the DEM with no warning",
      plausibility.crs_differences("ZALO19_EN", rasterio.crs.CRS.from_string(dem_crs).to_wkt()) == [])
client.post("/project/close")

print("\n" + "=" * 64)
print(f"  TOTAL {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("  FAILED: " + ", ".join(FAIL))
print("=" * 64)
sys.exit(1 if FAIL else 0)
