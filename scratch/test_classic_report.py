"""Validation for the classic report layout.

Part 1 is byte-for-byte: two real reports are parsed back into plain data,
regenerated, and compared with the originals. Their location comes from
FIDUCIA_CLASSIC_REPORTS (a folder holding *_project.txt and *_residual.txt);
the part is skipped when they are not available.

Part 2 runs the adapters on a synthetic block whose geometry is exact, so the
residuals Fiducia computes for the classic columns can be checked against
values known in advance.
"""

import math
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "scratch"))

from fiducia import classic_report                        # noqa: E402
from fiducia.collinearity import project_points      # noqa: E402

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def first_difference(a: bytes, b: bytes) -> str:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            line = a[:i].count(b"\n") + 1
            return f"first difference at byte {i}, line {line}: {a[i-20:i+20]!r} vs {b[i-20:i+20]!r}"
    return f"lengths differ: {len(a)} vs {len(b)}"

from classic_reports import parse_project, parse_residual  # noqa: E402



# =====================================================================
print("\n=== 1. Byte-for-byte against real reports ===")

folder = os.environ.get("FIDUCIA_CLASSIC_REPORTS")
samples = sorted(Path(folder).glob("*_project.txt")) if folder and Path(folder).is_dir() else []
if not samples:
    print("  (skipped: set FIDUCIA_CLASSIC_REPORTS to a folder of real reports)")
for project_file in samples:
    original = project_file.read_bytes()
    doc = parse_project(original.decode(classic_report.ENCODING))
    rebuilt = classic_report.format_project_report(doc).encode(classic_report.ENCODING)
    check(f"{project_file.name} regenerates byte-for-byte", rebuilt == original,
          f"{len(original)} bytes" if rebuilt == original else first_difference(original, rebuilt))

    residual_file = project_file.with_name(project_file.name.replace("_project", "_residual"))
    if residual_file.exists():
        original = residual_file.read_bytes()
        doc = parse_residual(original.decode(classic_report.ENCODING))
        rebuilt = classic_report.format_residual_report(doc).encode(classic_report.ENCODING)

        # The report prints residuals to 3 decimals, so per-image RMS values
        # recomputed from the printed rows can differ from the originals in the
        # last digit. Every character must match except those digits, and they
        # must agree to within that rounding.
        old_lines = original.decode(classic_report.ENCODING).split("\r\n")
        new_lines = rebuilt.decode(classic_report.ENCODING).split("\r\n")
        layout_ok = len(old_lines) == len(new_lines)
        worst = 0.0
        exact = 0
        for a, b in zip(old_lines, new_lines):
            if a == b:
                exact += 1
                continue
            numbers_a = re.findall(r"-?\d+\.\d+", a)
            numbers_b = re.findall(r"-?\d+\.\d+", b)
            if (len(a) != len(b) or "RMS" not in a or len(numbers_a) != len(numbers_b)
                    or re.sub(r"-?\d+\.\d+", "#", a) != re.sub(r"-?\d+\.\d+", "#", b)):
                layout_ok = False
                print("     differs:", repr(a[:80]), "\n              ", repr(b[:80]))
                continue
            worst = max(worst, *(abs(float(x) - float(y)) for x, y in zip(numbers_a, numbers_b)))
        check(f"{residual_file.name} regenerates with identical layout",
              layout_ok and worst <= 0.0015 and len(rebuilt) == len(original),
              f"{exact}/{len(old_lines)} lines byte-identical, {len(doc['rows'])} rows, "
              f"per-image RMS within {worst:.3f} of the original")


# =====================================================================
print("\n=== 2. Residuals from a synthetic block with a known answer ===")

FOCAL = 100.0
PITCH = 0.01          # mm per pixel
COLS, ROWS = 4000, 3000
H = 1500.0            # flying height over datum

camera = {"kind": "digital", "name": "Test", "focalMm": FOCAL, "pixelPitchMm": PITCH,
          "columns": COLS, "rows": ROWS}
# Distortion is left out of the geometry (k = 0 below) so residuals stay exact;
# the report mapping is checked separately with a camera that carries it.
eos = {
    "img_a": [1000.0, 2000.0, H, math.radians(0.4), math.radians(-0.3), math.radians(1.0)],
    "img_b": [1600.0, 2000.0, H, math.radians(-0.2), math.radians(0.5), math.radians(0.8)],
}
ground = {
    "G1": (1100.0, 2150.0, 40.0), "G2": (1450.0, 1850.0, 55.0), "G3": (1300.0, 2100.0, 20.0),
    "T1": (1250.0, 1950.0, 35.0), "T2": (1400.0, 2200.0, 60.0),
}


def pixel(point, image_id):
    film = project_points(np.array([point]), np.array(eos[image_id]), FOCAL)[0]
    return (film[0] / PITCH + (COLS - 1) / 2, -film[1] / PITCH + (ROWS - 1) / 2)


BLUNDER_PX = 3.0   # one tie measurement displaced this far in columns
observations = []
for pid, xyz in ground.items():
    for image_id in eos:
        col, row = pixel(xyz, image_id)
        if pid == "T1" and image_id == "img_b":
            col += BLUNDER_PX
        observations.append({"pointId": pid, "imageId": image_id, "col": col, "row": row,
                             "kind": "tie" if pid.startswith("T") else "gcp"})

state = {
    "name": "Synthetic", "description": "known answer", "modified": "2026-09-22T10:00:00+02:00",
    "camera": camera,
    "projection": {"output": "ZALO19", "gcpSource": "ZALO19"},
    "images": [{"id": i, "name": i.upper(), "path": f"C:\\data\\{i}.tif", "width": COLS,
                "height": ROWS, "bandCount": 3, "exterior": eos[i],
                "addedAt": "2026-09-01T09:00:00+02:00"} for i in eos],
    "gcps": [{"id": pid, "x": xyz[0], "y": xyz[1], "z": xyz[2]}
             for pid, xyz in ground.items() if pid.startswith("G")],
    "tiePoints": [{"id": "T1"}, {"id": "T2"}],
    "observations": observations,
    "model": {"objectPoints": {pid: list(xyz) for pid, xyz in ground.items()
                               if pid.startswith("T")},
              "imageSigmaMm": 0.01},
    "orthos": [], "dem": {},
}

doc = classic_report.residual_document(state)
rows = {(r["pointId"], r["imageName"]): r for r in doc["rows"]}
exact = [r for r in doc["rows"] if r["pointId"] != "T1"]
check("perfect measurements give zero residuals",
      max(abs(r["resXY"]) for r in exact) < 1e-6,
      f"worst {max(abs(r['resXY']) for r in exact):.2e} m")

# A column offset of p pixels on a near-vertical photo lands p * pitch * H' / f
# metres away on the ground, where H' is the height above the point.
expected = BLUNDER_PX * PITCH * (H - ground["T1"][2]) / FOCAL
blunder = rows[("T1", "IMG_B")]
check("a displaced tie measurement shows its expected ground miss",
      abs(blunder["resXY"] - expected) / expected < 0.03,
      f"{blunder['resXY']:.4f} m vs {expected:.4f} m expected")
check("the untouched measurement of the same point stays clean",
      rows[("T1", "IMG_A")]["resXY"] < 1e-6)
check("DS is the residual in standard deviations",
      abs(blunder["dsXY"] - BLUNDER_PX) / BLUNDER_PX < 0.05,
      f"{blunder['dsXY']:.2f} sigma for a {BLUNDER_PX:.0f}-pixel error at 1 pixel sigma")
check("the worst measurement is listed first",
      classic_report.format_residual_report(doc).split("\r\n")[14].split()[0] == "T1")

gcp_rows = [r for r in doc["rows"] if r["pointId"] == "G1"]
check("a control point reads the same on every image",
      len(gcp_rows) == 2 and gcp_rows[0]["resX"] == gcp_rows[1]["resX"])

text = classic_report.residual_report(state)
lines = text.split("\r\n")
check("summary lines are the fixed classic width", all(
    len(line) == 102 for line in lines[5:10]), str([len(line) for line in lines[5:10]]))
check("every listing row is the fixed classic width",
      all(len(line) == 207 for line in lines[14:14 + len(doc["rows"])]))
check("counts are distinct points, not measurements",
      lines[5].startswith(" Active GCPs:    3") and lines[7].startswith("  Active TPs:    2"),
      lines[5][:18] + " / " + lines[7][:18])

project_text = classic_report.project_report(state, "Synthetic.fidu")
check("project report opens with the classic heading",
      project_text.startswith("Project Report for Multi image orthorectification\r\n"))
check("georeference code is projection plus datum",
      "TM          D518" in project_text)
with_k = dict(state, camera=dict(camera, k0=3.52886e-05, k1=-3.95205e-09,
                                   k2=-1.01226e-13, k3=1.00869e-17))
radial = [line for line in classic_report.project_report(with_k).split(classic_report.NEWLINE)
          if "Radial Lens Distortion" in line]
check("K0..K3 land in the odd radial terms, even terms zero",
      radial[1].endswith("R1 : 3.528860e-05") and radial[7].endswith("R7 : 1.008690e-17")
      and radial[0].endswith("R0 : 0.000000e+00") and radial[2].endswith("R2 : 0.000000e+00"),
      radial[1].strip())
check("line endings are CRLF throughout",
      "\n" not in project_text.replace("\r\n", "") and "\n" not in text.replace("\r\n", ""))

print("\n" + "=" * 64)
print(f"  TOTAL {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("  FAILED: " + ", ".join(FAIL))
print("=" * 64)
sys.exit(1 if FAIL else 0)
