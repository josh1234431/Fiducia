"""Fiducia against a reference solution, on the same real block.

A real block's classic reports hold everything the reference package was given --
camera calibration, eight fiducial measurements per photo, control with its
surveyed coordinates, tie points -- and everything it solved: the exterior
orientation of each photo and the residual of every measurement.

Here Fiducia is given exactly those inputs, with the same weights, and its
answers are compared with the reference. Agreement validates the conventions
end to end (fiducial transform, principal point, distortion, rotation order,
angle signs, units) on real data, independently of any synthetic test.

Needs the reports: FIDUCIA_CLASSIC_REPORTS pointing at the folder holding
*_project.txt and *_residual.txt.
"""

import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "scratch"))

from classic_reports import parse_project, parse_residual  # noqa: E402
from fiducia import classic_report                               # noqa: E402
from fiducia.bundle import BundleInput, Observation, adjust_block  # noqa: E402
from fiducia.camera import CameraModel, fit_fiducial_transform     # noqa: E402
from fiducia.collinearity import project_points             # noqa: E402

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


folder = os.environ.get("FIDUCIA_CLASSIC_REPORTS")
reports = sorted(Path(folder).glob("*_project.txt")) if folder else []
if not reports:
    print("skipped: set FIDUCIA_CLASSIC_REPORTS")
    sys.exit(0)

project_file = reports[0]
doc = parse_project(project_file.read_text(encoding="cp1252").replace("\r\n", "\n").replace("\n", "\r\n"))
residual_doc = parse_residual(project_file.with_name(project_file.name.replace("_project", "_residual"))
                              .read_text(encoding="cp1252").replace("\r\n", "\n").replace("\n", "\r\n"))

SLOT = {label: slot for slot, label in classic_report.FIDUCIAL_ORDER}
cam_doc = doc["camera"]
radial = cam_doc["radial"]
camera = CameraModel.from_dict({
    "kind": "film",
    "focalMm": cam_doc["focalMm"],
    "ppoXMm": cam_doc["ppoXMm"], "ppoYMm": cam_doc["ppoYMm"],
    # R1, R3, R5, R7 multiply r, r^3, r^5, r^7.
    "k0": radial[1], "k1": radial[3], "k2": radial[5], "k3": radial[7],
    "fiducialsMm": {SLOT[label]: [x, y] for label, x, y in cam_doc["fiducials"]},
    "imageScale": cam_doc.get("scale") or 0,
})

print(f"\n{project_file.name}: {len(doc['images'])} photos, f = {camera.focal_mm} mm")

# ---------------------------------------------------------------------------
print("\n=== 1. Interior orientation from the reference fiducial measurements ===")
fits = {}
for image in doc["images"]:
    measured = {SLOT[label]: (c, r) for label, c, r in image["fiducials"]}
    fit = fit_fiducial_transform(measured, camera.fiducials_mm)
    fits[image["name"]] = fit
    print(f"  {image['name']}: affine RMS {fit.rms_px:.3f} px, "
          f"scan {camera.pixel_scale_mm(fit) * 1000:.1f} um/px")
check("fiducials fit to under 2 px on every photo (the app's own warning level)",
      all(f.rms_px < 2.0 for f in fits.values()),
      ", ".join(f"{f.rms_px:.2f}" for f in fits.values()))
pixel_mm = float(np.mean([camera.pixel_scale_mm(f) for f in fits.values()]))

# ---------------------------------------------------------------------------
print("\n=== 2. The block adjustment, with the reference weights ===")
observations, control = [], {}
gcp_sigma_px = doc["images"][0]["gcps"][0]["pixelSigma"]
tp_sigma_px = doc["images"][0]["tiePoints"][0]["pixelSigma"]
control_sigma = doc["images"][0]["gcps"][0]["xySigma"]
for image in doc["images"]:
    fit = fits[image["name"]]
    for g in image["gcps"]:
        film = camera.pixel_to_film([[g["col"], g["row"]]], fit)[0]
        observations.append(Observation(image["name"], g["id"], float(film[0]), float(film[1]), 1.0))
        control[g["id"]] = (g["x"], g["y"], g["z"])
    for t in image["tiePoints"]:
        film = camera.pixel_to_film([[t["col"], t["row"]]], fit)[0]
        # Weight multiplies the standardised residual: sigma ratio, not squared.
        observations.append(Observation(image["name"], t["id"], float(film[0]), float(film[1]),
                                        gcp_sigma_px / tp_sigma_px))

names = [i["name"] for i in doc["images"]]
data = BundleInput(
    image_ids=names, focal_by_image={n: camera.focal_mm for n in names},
    observations=observations, control=control, control_sigma_m=control_sigma,
    image_sigma_mm=gcp_sigma_px * pixel_mm, image_scale=camera.image_scale, robust=False,
)
result = adjust_block(data)
print(f"  {len(observations)} observations, {len(control)} control, "
      f"converged {result.converged} in {result.iterations} iterations, "
      f"sigma0 {result.sigma0:.3f}")
check("the block converges", result.converged, result.message[:80])

# ---------------------------------------------------------------------------
print("\n=== 3. Exterior orientation: Fiducia vs the reference ===")
print("  Each difference, and in brackets that difference in units of the parameter's")
print("  own standard deviation from Fiducia's adjustment. Two sound programs on the")
print("  same data should differ by no more than the solution's own uncertainty.")
labels = ["X0", "Y0", "Z0", "omega", "phi", "kappa"]
worst_sigma, worst_px = 0.0, 0.0
for image in doc["images"]:
    ours_eo = result.eo[image["name"]]
    theirs_eo = np.array(image["exterior"][:3] + [math.radians(a) for a in image["exterior"][3:]])
    d = ours_eo - theirs_eo
    d[3:] = (d[3:] + np.pi) % (2 * np.pi) - np.pi
    sig = np.array(result.eo_sigma[image["name"]])
    in_sigma = np.abs(d) / sig
    worst_sigma = max(worst_sigma, float(in_sigma.max()))

    def shown(k, v):
        return v if k < 3 else math.degrees(v) * 3600

    def unit(k):
        return "m" if k < 3 else '"'

    print(f"  {image['name']}")
    print("    difference  " + "  ".join(
        f"{labels[k]:>5} {shown(k, d[k]):+8.2f}{unit(k)} ({in_sigma[k]:3.1f})" for k in range(6)))
    print("    precision   " + "  ".join(
        f"{labels[k]:>5} {shown(k, sig[k]):8.2f}{unit(k)}      " for k in range(6)))
    pts = np.array([[g["x"], g["y"], g["z"]] for g in image["gcps"]])
    a = project_points(pts, ours_eo, camera.focal_mm)
    b = project_points(pts, theirs_eo, camera.focal_mm)
    worst_px = max(worst_px, float(np.max(np.hypot(*(a - b).T))) / pixel_mm)

print("\n  Across the whole photo: how far apart the two solutions put each corner on the")
print("  ground, against how uncertain Fiducia says each photo's corners are.")
from fiducia.bundle import corner_uncertainty      # noqa: E402
from fiducia.collinearity import rotation_matrix  # noqa: E402
corner_ok = True
for image in doc["images"]:
    ours_eo = result.eo[image["name"]]
    theirs_eo = np.array(image["exterior"][:3] + [math.radians(a) for a in image["exterior"][3:]])
    z = float(np.mean([g["z"] for g in image["gcps"]]))
    half = max(abs(v) for _, x, y in cam_doc["fiducials"] for v in (x, y)) * 0.95

    def to_ground(eo, film_xy):
        direction = rotation_matrix(*eo[3:]).T @ np.array([film_xy[0], film_xy[1], -camera.focal_mm])
        return eo[:2] + (z - eo[2]) / direction[2] * direction[:2]

    apart = max(np.linalg.norm(to_ground(ours_eo, c) - to_ground(theirs_eo, c))
                for c in ((half, half), (half, -half), (-half, half), (-half, -half)))
    uncertain = corner_uncertainty(ours_eo, result.eo_covariance[image["name"]],
                                   camera.focal_mm, half, half, z)
    corner_ok &= apart < 3 * uncertain
    print(f"  {image['name']}: corners {apart:6.2f} m apart; Fiducia's corner uncertainty "
          f"{uncertain:6.2f} m (1 sigma) -> {apart / uncertain:.1f} sigma")
check("frame-wide differences are within 3x the reported corner uncertainty", corner_ok)

check("every orientation parameter agrees within 3 standard deviations", worst_sigma < 3.0,
      f"worst {worst_sigma:.2f} sigma")
check("at the control the two solutions differ by under a pixel", worst_px < 1.0,
      f"worst {worst_px:.3f} px")

# ---------------------------------------------------------------------------
print("\n=== 4. Residuals: Fiducia's classic report vs the reference ===")
state = {
    "camera": camera.to_dict(),
    "images": [{"id": i["name"], "name": i["name"], "exterior": result.eo[i["name"]].tolist(),
                "fiducialFit": fits[i["name"]].to_dict()} for i in doc["images"]],
    "gcps": [{"id": k, "x": v[0], "y": v[1], "z": v[2]} for k, v in control.items()],
    "observations": [{"pointId": g["id"], "imageId": i["name"], "col": g["col"], "row": g["row"]}
                     for i in doc["images"] for g in i["gcps"]]
                    + [{"pointId": t["id"], "imageId": i["name"], "col": t["col"], "row": t["row"],
                        "kind": "tie"} for i in doc["images"] for t in i["tiePoints"]],
    "model": {"objectPoints": {k: v.tolist() for k, v in result.object_points.items()},
              "imageSigmaMm": gcp_sigma_px * pixel_mm},
}
ours = {(r["pointId"], r["imageName"]): r for r in classic_report.residual_document(state)["rows"]}
theirs = {(r["pointId"], r["imageName"]): r for r in residual_doc["rows"]}
common = sorted(set(ours) & set(theirs))
check("every reference residual row has a Fiducia counterpart",
      len(common) == len(theirs), f"{len(common)} of {len(theirs)}")

for kind in ("gcp", "tie"):
    keys = [k for k in common if theirs[k]["kind"] == kind]
    if not keys:
        continue
    diff_xy = np.array([ours[k]["resXY"] - theirs[k]["resXY"] for k in keys])
    diff_z = np.array([ours[k]["resZ"] - theirs[k]["resZ"] for k in keys])
    ground = np.array([np.hypot(ours[k]["groundX"] - theirs[k]["groundX"],
                                ours[k]["groundY"] - theirs[k]["groundY"]) for k in keys])
    corr = np.corrcoef([ours[k]["resXY"] for k in keys], [theirs[k]["resXY"] for k in keys])[0, 1]
    print(f"  {kind:<4} n={len(keys):3d}  |d Res XY| median {np.median(np.abs(diff_xy)):.3f} m, "
          f"worst {np.max(np.abs(diff_xy)):.3f} m;  |d Res Z| median {np.median(np.abs(diff_z)):.3f} m;  "
          f"ground coords differ by median {np.median(ground):.3f} m;  correlation {corr:.3f}")
    if kind == "gcp":
        # Magnitudes depend on the relative weighting, which the reference does
        # not print precisely; the pattern -- which point, which way -- must match.
        signs = [np.sign(ours[k][axis]) == np.sign(theirs[k][axis]) for k in keys
                 for axis in ("resX", "resY", "resZ") if abs(theirs[k][axis]) > 0.005]
        check("control corrections follow the reference pattern, including its Z sign",
              corr > 0.95 and np.mean(signs) > 0.9,
              f"correlation {corr:.3f}, same sign on {np.mean(signs):.0%} of components")
    else:
        tie_ids = sorted({k[0] for k in keys})
        sigmas = []
        for pid in tie_ids:
            k = next(key for key in keys if key[0] == pid)
            s = result.point_sigma[pid]
            dx = ours[k]["groundX"] - theirs[k]["groundX"]
            dy = ours[k]["groundY"] - theirs[k]["groundY"]
            dz = ours[k]["groundZ"] - theirs[k]["groundZ"]
            sigmas.append(max(abs(dx) / s[0], abs(dy) / s[1], abs(dz) / s[2]))
        sigmas = np.array(sigmas)
        check("tie-point ground coordinates agree within their own precision",
              np.mean(sigmas < 3) > 0.9,
              f"{np.mean(sigmas < 3):.0%} within 3 sigma, median {np.median(sigmas):.2f} sigma")
        check("tie-point residuals follow the reference", corr > 0.85, f"correlation {corr:.3f}")

rms = lambda values: math.sqrt(sum(v * v for v in values) / len(values))  # noqa: E731
for kind, label in (("gcp", "control"), ("tie", "tie points")):
    ours_rms = [rms([ours[k][axis] for k in common if theirs[k]["kind"] == kind])
                for axis in ("resX", "resY", "resZ")]
    theirs_rms = [rms([theirs[k][axis] for k in common if theirs[k]["kind"] == kind])
                  for axis in ("resX", "resY", "resZ")]
    print(f"  {label:<11} RMS X/Y/Z  Fiducia {ours_rms[0]:.3f} {ours_rms[1]:.3f} {ours_rms[2]:.3f}   "
          f"reference {theirs_rms[0]:.3f} {theirs_rms[1]:.3f} {theirs_rms[2]:.3f}")

print("\n" + "=" * 64)
print(f"  TOTAL {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("  FAILED: " + ", ".join(FAIL))
print("=" * 64)
sys.exit(1 if FAIL else 0)
