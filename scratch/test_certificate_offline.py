"""Offline certificate reading, against synthetic certificates with known values.

Two layouts -- a survey-style report (PPA and PPS, a fiducial table, four
diagonals and a mean) and a continental one (PPO, fiducials across the page,
field angles, decimal commas, no mean) -- each as a PDF with text, a clean
300 dpi scan and a rough 150 dpi scan that is noisy, blurred and tilted.

Run:  python scratch/test_certificate_offline.py
"""

import json
import runpy
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "engine"))

from fiducia import certificate_reader as cr  # noqa: E402

work = Path(tempfile.mkdtemp(prefix="fiducia-certs-"))
sys.argv = ["make_samples", str(work)]
runpy.run_path(str(HERE / "certificates" / "make_samples.py"), run_name="__main__")
truth = json.loads((work / "truth.json").read_text())

cr.configure(provider="offline")
passed = failed = 0


def check(name, ok, detail=""):
    global passed, failed
    passed += bool(ok)
    failed += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  -- {detail}")


status = cr.available()
check("offline reading is available on this computer", status["available"], status["reason"])

for path in sorted(work.glob("[AB]_*.*")):
    t = truth[path.name[0]]
    print(f"\n=== {path.name} ===")
    result = cr.read_certificate(str(path))
    camera, extraction = result["camera"], result["extraction"]
    rough = "rough" in path.name

    ppo = t.get("ppo") or [t["ppa"][0] + t["pps"][0], t["ppa"][1] + t["pps"][1]]
    check("focal length", abs(camera["focalMm"] - t["focal"]) < 1e-6, camera["focalMm"])
    check("principal point", abs(camera["ppoXMm"] - ppo[0]) < 1e-9
          and abs(camera["ppoYMm"] - ppo[1]) < 1e-9, (camera["ppoXMm"], camera["ppoYMm"]))
    marks = camera["fiducialsMm"]
    check("all eight fiducials, in the right slots", len(marks) == 8 and all(
        abs(marks[s][0] - v[0]) < 1e-6 and abs(marks[s][1] - v[1]) < 1e-6
        for s, v in t["fid"].items() if s in marks), f"{len(marks)} marks")

    rows = len(extraction["distortionTable"])
    if rough and rows < t["rows"]:
        # A rough scan may lose figures, but must say which.
        check("missing distortion rows are reported, not dropped silently",
              any("could not be read" in note for note in result["review"]),
              f"{rows} of {t['rows']} rows")
    else:
        check("every distortion row", rows == t["rows"], f"{rows} of {t['rows']}")
    check("the distortion table fits", result["distortionCheck"].get("verdict") == "good",
          f"{result['distortionCheck'].get('rmsUm', 0):.2f} um")
    check("camera named", t["camera"].lower() in camera["name"].lower(), camera["name"])

print(f"\n{'=' * 64}\n  TOTAL {passed} passed, {failed} failed\n{'=' * 64}")
sys.exit(1 if failed else 0)
