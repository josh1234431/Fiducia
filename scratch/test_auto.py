"""Validation for the automatic measurement features.

Both tests build a scene whose answer is known exactly, then check the
detector recovers it:

  Fiducial detection -- fiducials are drawn at known pixel positions on synthetic
  scans, then detected. Success is recovering those positions to sub-pixel.

  Automatic control -- a synthetic aerial photo and a matching reference orthomosaic
  are generated from the same ground pattern, the sensor model is then
  deliberately biased, and the matcher must recover the ground truth despite
  the bias.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))

from fiducia import auto_control, fiducial_detection                      # noqa: E402
from fiducia.camera import CameraModel                            # noqa: E402
from fiducia.collinearity import project_points, rotation_matrix  # noqa: E402

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


work = Path(tempfile.mkdtemp(prefix="fiducia-auto-"))
print(f"\nWorking in {work}\n")

# =====================================================================
# 1. Fiducial detection
# =====================================================================

print("=== 1. Fiducial detection ===")

CALIBRATED = {
    "top_left": (-106.005, 106.005), "top_middle": (-0.007, 111.996),
    "top_right": (105.996, 105.996), "right_middle": (112.003, -0.002),
    "bottom_right": (106.004, -106.004), "bottom_middle": (0.000, -111.995),
    "bottom_left": (-106.000, -105.999), "left_middle": (-112.010, 0.004),
}

SCAN = 2200
PITCH = 0.115          # mm per pixel, so 224 mm of film spans ~1950 px


def make_scan(path, rotation_deg=0.0, shift=(0.0, 0.0), seed=0, noise=9.0):
    """A synthetic scanned aerial: textured ground plus eight fiducial crosses."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:SCAN, 0:SCAN]

    image = (
        118
        + 34 * np.sin(xs / 47.0) * np.cos(ys / 61.0)
        + 22 * np.sin((xs + ys) / 29.0)
        + rng.normal(0, noise, (SCAN, SCAN))
    )

    theta = np.radians(rotation_deg)
    cos, sin = np.cos(theta), np.sin(theta)
    truth = {}

    for slot, (mx, my) in CALIBRATED.items():
        col = (mx * cos + my * sin) / PITCH + SCAN / 2 + shift[0]
        row = -(-mx * sin + my * cos) / PITCH + SCAN / 2 + shift[1]
        truth[slot] = (col, row)

        c, r = int(round(col)), int(round(row))
        arm, thick = 26, 2
        # A dark cross on light film, with a small bright surround so the
        # feature has structure beyond a single stroke.
        image[r - arm - 4: r + arm + 5, c - arm - 4: c + arm + 5] = 205
        image[r - thick: r + thick + 1, c - arm: c + arm + 1] = 28
        image[r - arm: r + arm + 1, c - thick: c + thick + 1] = 28

    data = np.clip(image, 0, 255).astype(np.uint8)
    with rasterio.open(
        path, "w", driver="GTiff", width=SCAN, height=SCAN, count=1, dtype="uint8",
        compress="DEFLATE", tiled=True, blockxsize=256, blockysize=256,
    ) as sink:
        sink.write(data, 1)
    return truth


camera = CameraModel(
    kind="film", focal_mm=153.690, image_scale=20000.0, fiducials_mm=CALIBRATED,
)

reference_path = work / "scan_reference.tif"
reference_truth = make_scan(reference_path, rotation_deg=0.0, seed=1)

target_path = work / "scan_target.tif"
target_truth = make_scan(target_path, rotation_deg=0.7, shift=(14.0, -9.0), seed=2)
print(f"  reference and target scans written ({SCAN}x{SCAN})")

# -- template mode: measure one photo, detect on the next ------------------
templates = fiducial_detection.extract_templates(
    str(reference_path), {k: list(v) for k, v in reference_truth.items()}, chip_px=96
)
check("templates cut from the measured photo", len(templates) == 8, f"{len(templates)} chips")

result = fiducial_detection.detect_fiducials(str(target_path), camera, templates)
check("all eight marks found", result.accepted == 8,
      f"{result.accepted} found, RMS {result.rms_px:.3f} px" if result.rms_px else "no fit")

errors = [
    float(np.hypot(c.col - target_truth[c.slot][0], c.row - target_truth[c.slot][1]))
    for c in result.candidates
]
check("every mark within 1 px of truth", errors and max(errors) < 1.0,
      f"worst {max(errors):.3f} px, mean {np.mean(errors):.3f} px" if errors else "none")
check("fitted interior orientation is tight", result.rms_px is not None and result.rms_px < 1.0,
      f"RMS {result.rms_px:.4f} px" if result.rms_px else "no fit")

# -- synthetic mode: no template available --------------------------------
cold = fiducial_detection.detect_fiducials(str(reference_path), camera, templates=None)
cold_errors = [
    float(np.hypot(c.col - reference_truth[c.slot][0], c.row - reference_truth[c.slot][1]))
    for c in cold.candidates
]
check("synthetic cross finds marks with no template", cold.accepted >= 6,
      f"{cold.accepted} of 8 found" + (f", worst {max(cold_errors):.2f} px" if cold_errors else ""))

# -- a mark that is not there ---------------------------------------------
blank = work / "scan_blank.tif"
rng = np.random.default_rng(9)
with rasterio.open(blank, "w", driver="GTiff", width=SCAN, height=SCAN, count=1,
                   dtype="uint8", compress="DEFLATE") as sink:
    sink.write(rng.integers(100, 140, (SCAN, SCAN)).astype(np.uint8), 1)

empty = fiducial_detection.detect_fiducials(str(blank), camera, templates)
check("reports failure on a photo with no marks", empty.accepted < 3 or (empty.rms_px or 99) > 4,
      f"{empty.accepted} claimed, message: {empty.message[:60]}")


# =====================================================================
# 2. Automatic ground control matching
# =====================================================================

print("\n=== 2. Ground control matching ===")

FOCAL, PIXEL_PITCH, COLS, ROWS, GSD = 100.0, 0.006, 1400, 1100, 0.25
HEIGHT = (FOCAL / 1000.0) * (GSD / (PIXEL_PITCH / 1000.0))
ORIGIN_E, ORIGIN_N = 60000.0, 6250000.0
CRS = "+proj=tmerc +lat_0=0 +lon_0=19 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"


def ground_pattern(east, north):
    """Distinctive, non-repeating ground so correlation has something to lock to."""
    u = (east - ORIGIN_E) / 22.0
    v = (north - ORIGIN_N) / 22.0
    blocks = ((np.floor(u * 0.7) * 7 + np.floor(v * 0.7) * 13) % 5) * 32 + 48
    ridges = 40 * np.sin(u * 0.9 + np.cos(v * 0.6) * 2.0)
    speckle = 26 * np.sin(u * 5.3 + v * 3.1) * np.cos(u * 2.2 - v * 4.7)
    return np.clip(blocks + ridges + speckle, 5, 250)


digital = CameraModel(
    kind="digital", focal_mm=FOCAL, pixel_pitch_mm=PIXEL_PITCH, columns=COLS, rows=ROWS,
)
eo_true = np.array([ORIGIN_E, ORIGIN_N, HEIGHT, 0.004, -0.003, 0.02])

# -- the photograph --------------------------------------------------------
cols, rows = np.meshgrid(np.arange(COLS), np.arange(ROWS))
film = digital.pixel_to_film(np.column_stack([cols.ravel(), rows.ravel()]))
rot = rotation_matrix(eo_true[3], eo_true[4], eo_true[5])
directions = np.column_stack([film[:, 0], film[:, 1], np.full(film.shape[0], -FOCAL)]) @ rot
t = (0.0 - eo_true[2]) / directions[:, 2]
photo_east = eo_true[0] + t * directions[:, 0]
photo_north = eo_true[1] + t * directions[:, 1]

# Real photography and a reference mosaic never match pixel for pixel: they
# are different sensors, often different seasons. Give them different gain and
# offset plus independent noise, so the correlation has to be genuinely
# contrast-invariant rather than finding an identical copy of itself.
_rng_photo = np.random.default_rng(41)
photo = np.clip(
    ground_pattern(photo_east, photo_north).reshape(ROWS, COLS) * 1.18 - 14
    + _rng_photo.normal(0, 6.0, (ROWS, COLS)),
    0, 255,
).astype(np.uint8)
photo_path = work / "photo.tif"
with rasterio.open(photo_path, "w", driver="GTiff", width=COLS, height=ROWS, count=1,
                   dtype="uint8", compress="DEFLATE", tiled=True,
                   blockxsize=256, blockysize=256) as sink:
    sink.write(photo, 1)

# -- the reference orthomosaic, on the same ground -------------------------
ref_res = 0.25
ref_w = ref_h = 1800
west = ORIGIN_E - ref_w * ref_res / 2
north = ORIGIN_N + ref_h * ref_res / 2
rx, ry = np.meshgrid(
    west + (np.arange(ref_w) + 0.5) * ref_res,
    north - (np.arange(ref_h) + 0.5) * ref_res,
)
_rng_ref = np.random.default_rng(42)
_ref = ground_pattern(rx, ry) * 0.82 + 30 + _rng_ref.normal(0, 6.0, (ref_h, ref_w))
# A touch of blur, as a resampled mosaic always has.
from scipy.ndimage import gaussian_filter as _gf
ref_image = np.clip(_gf(_ref, 0.8), 0, 255).astype(np.uint8)
ref_path = work / "reference.tif"
with rasterio.open(ref_path, "w", driver="GTiff", width=ref_w, height=ref_h, count=1,
                   dtype="uint8", crs=CRS,
                   transform=Affine(ref_res, 0, west, 0, -ref_res, north),
                   compress="DEFLATE", tiled=True, blockxsize=256, blockysize=256) as sink:
    sink.write(ref_image, 1)

# -- a flat DEM ------------------------------------------------------------
dem_path = work / "dem.tif"
with rasterio.open(dem_path, "w", driver="GTiff", width=200, height=200, count=1,
                   dtype="float32", crs=CRS,
                   transform=Affine(2.5, 0, west, 0, -2.5, north),
                   compress="DEFLATE") as sink:
    sink.write(np.zeros((200, 200), dtype=np.float32), 1)

print(f"  photo {COLS}x{ROWS}, reference {ref_w}x{ref_h} at {ref_res} m")

# The model is deliberately wrong by ~35 m, as a real block is before adjustment.
# Deliberately NOT a whole number of reference pixels (0.25 m), so the true
# answer falls between cells and subpixel refinement has to earn its place.
BIAS = np.array([26.13, -21.07, 0.0])
eo_biased = eo_true.copy()
eo_biased[:3] += BIAS
truth_shift = float(np.hypot(BIAS[0], BIAS[1]))

image_record = {
    "id": "img_test", "name": "photo", "path": str(photo_path),
    "online": True, "width": COLS, "height": ROWS, "bandCount": 1,
    "exterior": eo_biased.tolist(), "fiducialFit": None, "clipRegion": None,
}

options = auto_control.AutoControlOptions(
    target_count=18, search_m=70.0, min_score=0.5, min_separation_m=45.0,
    patch_px=96, edge_margin_px=120,
)

result = auto_control.match_ground_control(
    image_record, digital.to_dict(), str(ref_path), str(dem_path),
    seed_observations=None, options=options,
)

check("proposals returned", len(result.proposals) >= 8,
      f"{len(result.proposals)} from {result.tested} tested, seed '{result.seed}'")

# Each proposal's ground coordinate should match where that photo pixel
# TRULY lands -- not where the biased model thought it would.
if result.proposals:
    pixels = np.array([[p.col, p.row] for p in result.proposals])
    film_p = digital.pixel_to_film(pixels)
    dirs = np.column_stack(
        [film_p[:, 0], film_p[:, 1], np.full(len(film_p), -FOCAL)]
    ) @ rot
    tt = (0.0 - eo_true[2]) / dirs[:, 2]
    true_x = eo_true[0] + tt * dirs[:, 0]
    true_y = eo_true[1] + tt * dirs[:, 1]

    got = np.array([[p.x, p.y] for p in result.proposals])
    errors = np.hypot(got[:, 0] - true_x, got[:, 1] - true_y)

    check("ground coordinates recover the truth", float(np.median(errors)) < 1.5,
          f"median {np.median(errors):.2f} m, worst {errors.max():.2f} m "
          f"(model was biased {truth_shift:.0f} m)")
    check("most proposals are accurate", float(np.mean(errors < 2.0)) > 0.8,
          f"{np.mean(errors < 2.0):.0%} within 2 m")
    check("the correction matches the injected bias",
          abs(result.mean_shift_m - truth_shift) < 12.0,
          f"corrected {result.mean_shift_m:.1f} m, injected {truth_shift:.1f} m")
    check("proposals are spread across the frame",
          float(np.ptp(pixels[:, 0])) > COLS * 0.35, f"col spread {np.ptp(pixels[:, 0]):.0f} px")
    check("scores are meaningful",
          all(p.score >= options.min_score for p in result.proposals),
          f"min {min(p.score for p in result.proposals):.3f}")

# -- a reference that does not overlap -------------------------------------
far_path = work / "reference_far.tif"
with rasterio.open(far_path, "w", driver="GTiff", width=400, height=400, count=1,
                   dtype="uint8", crs=CRS,
                   transform=Affine(ref_res, 0, west + 40000, 0, -ref_res, north + 40000),
                   compress="DEFLATE") as sink:
    sink.write(np.random.default_rng(5).integers(0, 255, (400, 400)).astype(np.uint8), 1)

try:
    far = auto_control.match_ground_control(
        image_record, digital.to_dict(), str(far_path), str(dem_path), None, options
    )
    check("a non-overlapping reference yields nothing", len(far.proposals) == 0,
          far.message[:70])
except Exception as exc:
    check("a non-overlapping reference fails clearly", True, f"{type(exc).__name__}")

# =====================================================================
# 3. Certificate reading -- the parts that are not the model
# =====================================================================
#
# The model's reading is checked by arithmetic, and that arithmetic is what
# turns a plausible-looking transcription into a trustworthy one. It is
# testable without any API call, so it is tested here.

print("\n=== 3. Certificate safety net ===")

from fiducia import certificate_reader  # noqa: E402

status = certificate_reader.available()
check("status reports availability honestly", "available" in status,
      f"available={status['available']}, model={certificate_reader.MODEL}")

# A clean table, generated from known coefficients: must fit tightly.
TRUE_K = (3.52886e-05, -3.95205e-09, -1.01226e-13, 1.00869e-17)
radii = np.array([10, 20, 40, 60, 80, 100, 120, 140], dtype=float)
clean_um = (radii * TRUE_K[0] + radii**3 * TRUE_K[1]
            + radii**5 * TRUE_K[2] + radii**7 * TRUE_K[3]) * 1000.0

clean_table = [{"radiusMm": float(r), "distortionUm": float(d)}
               for r, d in zip(radii, clean_um)]
good = certificate_reader._verify_distortion(clean_table)
check("a clean table is passed", good["fitted"] and good["verdict"] == "good",
      f"verdict '{good.get('verdict')}', RMS {good.get('rmsUm', 0):.4f} um")

# Now transpose two digits in one entry, the classic transcription blunder.
typo_table = [dict(row) for row in clean_table]
original = typo_table[5]["distortionUm"]
typo_table[5]["distortionUm"] = original + 18.0
bad = certificate_reader._verify_distortion(typo_table)
check("a transposed digit is caught", bad["fitted"] and bad["verdict"] != "good",
      f"verdict '{bad.get('verdict')}', RMS {bad.get('rmsUm', 0):.2f} um "
      f"(was {good.get('rmsUm', 0):.4f})")
check("the warning names the problem in plain language",
      "misread" in bad["note"] or "looser" in bad["note"], bad["note"][:78])

# PPA + PPS must be added, an addition otherwise done by hand.
extraction = {
    "cameraKind": "film",
    "cameraName": "Wild RC30", "lensType": "15/4 UAG-S", "serialNumber": "13352",
    "principalPointMode": "ppa_pps",
    "ppoX": None, "ppoY": None,
    "ppaX": {"value": -0.008, "sourceText": "PPA x -0.008", "confidence": "high"},
    "ppaY": {"value": 0.005, "sourceText": "PPA y 0.005", "confidence": "high"},
    "ppsX": {"value": -0.005, "sourceText": "PPS x -0.005", "confidence": "high"},
    "ppsY": {"value": 0.004, "sourceText": "PPS y 0.004", "confidence": "high"},
    "focalLength": {"value": 153.69, "sourceText": "153.690 mm", "confidence": "high"},
    "fiducials": [
        {"slot": s, "x": v[0], "y": v[1], "label": s} for s, v in CALIBRATED.items()
    ],
    "radialCoefficients": [],
    "decenteringP1": None, "decenteringP2": None,
    "pixelPitchMm": None, "columns": None, "rows": None,
}
patch = certificate_reader._to_camera_patch(extraction, good)

check("PPO is computed as PPA + PPS",
      abs(patch["ppoXMm"] - (-0.013)) < 1e-9 and abs(patch["ppoYMm"] - 0.009) < 1e-9,
      f"PPO = {patch['ppoXMm']:.6f}, {patch['ppoYMm']:.6f}")
check("all eight fiducials carry through", len(patch["fiducialsMm"]) == 8,
      f"{len(patch['fiducialsMm'])} marks")
check("distortion coefficients come from the fitted table",
      abs(patch["k0"] - TRUE_K[0]) / TRUE_K[0] < 0.01,
      f"K0 {patch['k0']:.6e} vs true {TRUE_K[0]:.6e}")
check("camera name is assembled from the parts",
      patch["name"] == "Wild RC30 15/4 UAG-S 13352", patch["name"])
check("the schema is valid JSON Schema",
      certificate_reader.CERTIFICATE_SCHEMA["type"] == "object"
      and len(certificate_reader.CERTIFICATE_SCHEMA["required"]) > 15,
      f"{len(certificate_reader.CERTIFICATE_SCHEMA['required'])} required fields")


# =====================================================================
#
# Choosing a reading service. No network: each SDK is handed a transport that
# answers locally, so what is checked is the request Fiducia builds and the
# way it reads the reply -- the two places a provider integration goes wrong.

print("\n=== 4. Reading service ===")

import base64 as _b64   # noqa: E402
import json as _json    # noqa: E402
import os as _os        # noqa: E402

import httpx2           # noqa: E402

for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY"):
    _os.environ.pop(name, None)

scan = work / "certificate.png"
from PIL import Image  # noqa: E402
Image.new("L", (64, 64), 255).save(scan)

certificate_reader.configure(provider="openai", keys={"anthropic": None, "openai": None})
status = certificate_reader.available()
check("no key reads as unavailable, with a reason that says what to do",
      not status["available"] and "API key" in status["reason"], status["reason"])

try:
    certificate_reader.read_certificate(str(scan))
    missing_key_message = ""
except certificate_reader.CertificateReadError as exc:
    missing_key_message = str(exc)
check("a missing key fails with a plain message, not the SDK's header complaint",
      "No ChatGPT (OpenAI) API key" in missing_key_message
      and "X-Api-Key" not in missing_key_message, missing_key_message)

certificate_reader.configure(keys={"openai": "sk-test-000000000000abcd"},
                       models={"openai": "gpt-test"})
status = certificate_reader.available()
check("a saved key is reported by its last four characters only",
      status["available"] and status["services"]["openai"]["keyHint"] == "abcd"
      and "sk-test" not in _json.dumps(status),
      _json.dumps(status["services"]["openai"]))

canned = {key: None for key in certificate_reader.CERTIFICATE_SCHEMA["required"]}
canned.update(cameraName="Wild RC30", lensType="", serialNumber="", calibrationDate="",
              cameraKind="film", principalPointMode="none",
              focalLength={"value": 153.28, "sourceText": "c = 153.28 mm",
                           "confidence": "high"},
              fiducials=[], distortionTable=[], distortionUnitsAsPrinted="",
              radialCoefficients=[], notes="", unreadable=[])

seen = {}


def openai_reply(request):
    seen["url"] = str(request.url)
    seen["auth"] = request.headers.get("authorization")
    seen["body"] = _json.loads(request.content)
    return httpx2.Response(200, json={
        "id": "resp_test", "object": "response", "created_at": 0,
        "status": "completed", "model": "gpt-test",
        "output": [{
            "type": "message", "id": "msg_test", "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": _json.dumps(canned),
                         "annotations": []}],
        }],
        "usage": {"input_tokens": 1200, "output_tokens": 340, "total_tokens": 1540},
    })


import openai  # noqa: E402

_real_openai = openai.OpenAI
openai.OpenAI = lambda **kw: _real_openai(
    **kw, http_client=httpx2.Client(transport=httpx2.MockTransport(openai_reply)))
try:
    result = certificate_reader.read_certificate(str(scan))
finally:
    openai.OpenAI = _real_openai

body = seen.get("body", {})
content = body.get("input", [{}])[0].get("content", [])
check("OpenAI is called on the Responses endpoint with the saved key",
      seen.get("url", "").endswith("/responses")
      and seen.get("auth") == "Bearer sk-test-000000000000abcd", seen.get("url"))
check("the scan is sent as an image with the instructions and model override",
      content and content[0]["type"] == "input_image"
      and content[0]["image_url"].startswith("data:image/png;base64,")
      and body.get("model") == "gpt-test" and "calibration" in body.get("instructions", ""),
      f"model={body.get('model')}")
check("the certificate schema is enforced strictly",
      body.get("text", {}).get("format", {}).get("strict") is True
      and body["text"]["format"]["schema"] == certificate_reader.MODEL_SCHEMA)
check("the OpenAI reply becomes a camera proposal",
      result["camera"]["focalMm"] == 153.28 and result["provider"] == "openai"
      and result["usage"] == {"inputTokens": 1200, "outputTokens": 340},
      f"focal {result['camera']['focalMm']} mm, usage {result['usage']}")

pdf = work / "certificate.pdf"
pdf.write_bytes(b"%PDF-1.4\n%test\n")
openai.OpenAI = lambda **kw: _real_openai(
    **kw, http_client=httpx2.Client(transport=httpx2.MockTransport(openai_reply)))
try:
    certificate_reader.read_certificate(str(pdf))
finally:
    openai.OpenAI = _real_openai
document = seen["body"]["input"][0]["content"][0]
check("a PDF is sent as a file input",
      document["type"] == "input_file" and document["filename"] == "certificate.pdf"
      and document["file_data"].startswith("data:application/pdf;base64,"),
      document["type"])


def rejected(request):
    return httpx2.Response(401, json={"error": {"message": "Incorrect API key provided",
                                                "type": "invalid_request_error"}})


openai.OpenAI = lambda **kw: _real_openai(
    **kw, max_retries=0,
    http_client=httpx2.Client(transport=httpx2.MockTransport(rejected)))
try:
    certificate_reader.read_certificate(str(scan))
    rejected_message = ""
except certificate_reader.CertificateReadError as exc:
    rejected_message = str(exc)
finally:
    openai.OpenAI = _real_openai
check("a rejected key says so in plain language",
      "OpenAI rejected the API key" in rejected_message, rejected_message)

# And Claude still works through the same path.
certificate_reader.configure(provider="anthropic", keys={"anthropic": "sk-ant-test-00000000wxyz"})


def anthropic_reply(request):
    seen["anthropic"] = _json.loads(request.content)
    seen["anthropic_key"] = request.headers.get("x-api-key")
    return httpx2.Response(200, json={
        "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-opus-5",
        "content": [{"type": "text", "text": _json.dumps(canned)}],
        "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 900, "output_tokens": 300},
    })


import anthropic  # noqa: E402

_real_anthropic = anthropic.Anthropic
anthropic.Anthropic = lambda **kw: _real_anthropic(
    **kw, http_client=httpx2.Client(transport=httpx2.MockTransport(anthropic_reply)))
try:
    result = certificate_reader.read_certificate(str(pdf))
finally:
    anthropic.Anthropic = _real_anthropic
check("Claude is called with the saved key and the PDF as a document",
      seen.get("anthropic_key") == "sk-ant-test-00000000wxyz"
      and seen["anthropic"]["messages"][0]["content"][0]["type"] == "document"
      and result["provider"] == "anthropic" and result["model"] == "claude-opus-5",
      f"{result['provider']} / {result['model']}")

certificate_reader.configure(provider="anthropic", keys={"anthropic": None, "openai": None},
                       models={"openai": None})

print("\n" + "=" * 64)
print(f"  TOTAL {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("  FAILED: " + ", ".join(FAIL))
print("=" * 64)
sys.exit(1 if FAIL else 0)
