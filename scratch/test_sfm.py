"""Validation for structure from motion without positions.

Part 1 is a synthetic rock face photographed from the ground: the true
cameras, points and lens are known, the measurements carry noise, wrong
observations and wholly false tracks. Structure from motion has no
positions, so its result is compared with the truth after the one
similarity transform (scale, rotation, shift) it cannot know.

Part 2, when the public Brighton Beach drone photos are on this computer
(FIDUCIA_BRIGHTON pointing at their folder), solves them with their GPS
ignored and compares the cameras with the GPS-based solution of the same
photos (FIDUCIA_BRIGHTON_PROJECT, a solved .fidu project).
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))

from fiducia import sfm  # noqa: E402

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


rng = np.random.default_rng(11)

# ---------------------------------------------------------------------------
print("\n=== 1. A rock face from the ground ===")
WIDTH, HEIGHT = 6000, 4000
TRUE = sfm.Intrinsics(focal=3600.0, cx=WIDTH / 2, cy=HEIGHT / 2, k1=-0.06, k2=0.015)

# The face: 60 m wide, 25 m high, bulging and gullied, with a ledge of scree
# in front. World axes: x along the face, y into it, z up.
n_points = 6000
u = rng.uniform(0, 60, n_points)
v = rng.uniform(0, 25, n_points)
depth = 3.0 * np.sin(u / 7.0) + 2.0 * np.cos(v / 5.0) + 0.4 * np.sin(u * 1.3) * np.cos(v * 1.1)
face = np.column_stack([u, depth, v])
scree = np.column_stack([rng.uniform(0, 60, 1500), rng.uniform(-8, -1, 1500), rng.uniform(-1.5, 0, 1500)])
points = np.vstack([face, scree])


def look_at(centre, target):
    """World-to-camera rotation (x right, y down, z forward) looking at a target."""
    forward = (target - centre) / np.linalg.norm(target - centre)
    right = np.cross(forward, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.vstack([right, down, forward])


# 24 stations along an arc 22-32 m out from the face, at the foot and from a
# path higher up, each aimed at a point on the face with a little jitter.
cameras = []
for k in range(24):
    s = k / 23
    centre = np.array([-6 + 72 * s + rng.normal(0, 1.5), -26 - 6 * math.sin(math.pi * s) + rng.normal(0, 1),
                       1.6 if k % 2 == 0 else 9.0 + rng.normal(0, 0.5)])
    target = np.array([np.clip(centre[0] + rng.normal(0, 6), 5, 55), 0.0, rng.uniform(6, 18)])
    R = look_at(centre, target)
    cameras.append((R, -R @ centre, centre))

tracks, truth_index = [], []
for p, X in enumerate(points):
    obs = []
    for c, (R, t, centre) in enumerate(cameras):
        cam = R @ X + t
        if cam[2] <= 1 or cam[2] > 80:
            continue
        px = TRUE.project(cam.reshape(1, 3))[0]
        if 0 <= px[0] < WIDTH and 0 <= px[1] < HEIGHT:
            obs.append((c, px[0] + rng.normal(0, 0.5), px[1] + rng.normal(0, 0.5)))
    if len(obs) >= 2:
        # About one observation in twenty is a wrong match, off by 20-200 px.
        noisy = []
        for c, x, y in obs:
            if rng.random() < 0.05:
                angle = rng.uniform(0, 2 * math.pi)
                shift = rng.uniform(20, 200)
                x, y = x + shift * math.cos(angle), y + shift * math.sin(angle)
            noisy.append((c, x, y))
        tracks.append(noisy)
        truth_index.append(p)
# And some tracks that are false from end to end.
for _ in range(int(0.03 * len(tracks))):
    chosen = rng.choice(len(cameras), size=3, replace=False)
    tracks.append([(int(c), rng.uniform(0, WIDTH), rng.uniform(0, HEIGHT)) for c in chosen])
    truth_index.append(-1)
print(f"  {len(points):,} points, {len(cameras)} photos, {len(tracks):,} tracks, "
      f"{np.mean([len(t) for t in tracks]):.1f} photos per track")

# The pairs' matches, as matching would give them, for choosing where to start.
pair_matches = {}
for track in tracks:
    for a in range(len(track)):
        for b in range(a + 1, len(track)):
            (i, xa, ya), (j, xb, yb) = sorted([track[a], track[b]])
            pair_matches.setdefault((i, j), ([], []))
            pair_matches[(i, j)][0].append((xa, ya))
            pair_matches[(i, j)][1].append((xb, yb))
pair_matches = {k: (np.array(a), np.array(b)) for k, (a, b) in pair_matches.items()}

started = time.time()
# The focal length from metadata is rarely exact: start 4% off.
result = sfm.reconstruct([(WIDTH, HEIGHT)] * len(cameras), tracks, pair_matches, TRUE.focal * 1.04)
elapsed = time.time() - started
s = result["summary"]
check("every photo placed", s["placed"] == len(cameras), f"{s['placed']} of {len(cameras)} in {elapsed:.1f} s")

placed = sorted(result["poses"])
estimated = np.array([result["poses"][i]["centre"] for i in placed])
true_centres = np.array([cameras[i][2] for i in placed])
scale, R, shift = sfm.similarity(estimated, true_centres)
aligned = (scale * (R @ estimated.T)).T + shift
centre_error = np.linalg.norm(aligned - true_centres, axis=1)
check("camera positions", np.sqrt(np.mean(centre_error ** 2)) < 0.05,
      f"RMS {np.sqrt(np.mean(centre_error ** 2)) * 100:.1f} cm, worst {centre_error.max() * 100:.1f} cm, "
      f"over a scene 70 m across")

angle_errors = []
for i in placed:
    r_est = sfm._rotation(np.array(result["poses"][i]["rvec"]))
    # world-to-camera, expressed in the true world: R_est @ R^T
    r_aligned = r_est @ R.T
    delta = r_aligned @ cameras[i][0].T
    angle_errors.append(math.degrees(math.acos(np.clip((np.trace(delta) - 1) / 2, -1, 1))))
check("camera orientations", max(angle_errors) < 0.05, f"worst {max(angle_errors) * 3600:.0f} arc-seconds")

ids = result["pointTracks"]
real = [n for n, t in enumerate(ids) if truth_index[t] >= 0]
est_pts = (scale * (R @ result["points"][real].T)).T + shift
true_pts = points[[truth_index[ids[n]] for n in real]]
point_error = np.linalg.norm(est_pts - true_pts, axis=1)
check("points on the face", np.median(point_error) < 0.03,
      f"median {np.median(point_error) * 100:.1f} cm, 95% within {np.percentile(point_error, 95) * 100:.1f} cm, "
      f"{len(real):,} points")
false_kept = sum(1 for t in ids if truth_index[t] < 0)
false_total = sum(1 for t in truth_index if t < 0)
check("false tracks rejected", false_kept <= 0.02 * false_total, f"{false_kept} of {false_total} kept")

lens = result["lens"]
check("focal length recovered from the photos", s["focalSource"] == "photos" and abs(lens["focal"] / TRUE.focal - 1) < 0.002,
      f"{lens['focal']:.1f} px against {TRUE.focal:.1f}, started at {TRUE.focal * 1.04:.1f}")
check("lens distortion recovered", abs(lens["k1"] - TRUE.k1) < 0.005 and abs(lens["k2"] - TRUE.k2) < 0.01,
      f"k1 {lens['k1']:.4f} (true {TRUE.k1}), k2 {lens['k2']:.4f} (true {TRUE.k2})")
check("reprojection near the noise", s["reprojectionRmsPx"] < 1.0,
      f"{s['reprojectionRmsPx']:.2f} px RMS for 0.5 px noise per axis")


# ---------------------------------------------------------------------------
photos_dir = os.environ.get("FIDUCIA_BRIGHTON")
project_dir = os.environ.get("FIDUCIA_BRIGHTON_PROJECT")
if photos_dir and project_dir and Path(photos_dir).exists() and Path(project_dir).exists():
    print("\n=== 2. Real drone photos, GPS ignored ===")
    from fiducia import drone

    state = json.load(open(Path(project_dir) / "project.json", encoding="utf-8"))
    images = [im for im in state["images"] if (state.get("model") or {}).get("exterior", {}).get(im["id"])]
    photos = []
    for im in images:
        local = Path(photos_dir) / Path(im["path"]).name
        photos.append({"id": im["id"], "path": str(local if local.exists() else im["path"])})
    metas = [drone.read_metadata(p["path"]) for p in photos]
    camera = drone.camera_from_metadata(metas)["camera"]
    focal_px = camera["focalMm"] / camera["pixelPitchMm"]
    started = time.time()
    tracks_r, pairs_r, match_summary = sfm.match_photos(photos, focal_px)
    real_result = sfm.reconstruct([(im["width"], im["height"]) for im in images], tracks_r, pairs_r, focal_px)
    elapsed = time.time() - started
    rs = real_result["summary"]
    check("every drone photo placed without GPS", rs["placed"] == len(photos),
          f"{rs['placed']} of {len(photos)}, {rs['points']:,} points, {elapsed:.0f} s")
    placed = sorted(real_result["poses"])
    est = np.array([real_result["poses"][i]["centre"] for i in placed])
    ref = np.array([state["model"]["exterior"][images[i]["id"]][:3] for i in placed])
    scale_r, R_r, shift_r = sfm.similarity(est, ref)
    delta = (scale_r * (R_r @ est.T)).T + shift_r - ref
    err = np.linalg.norm(delta, axis=1)
    horizontal = np.sqrt(np.mean(delta[:, 0] ** 2 + delta[:, 1] ** 2))
    vertical = np.sqrt(np.mean(delta[:, 2] ** 2))
    # The flight's ground pixel, from the GPS-based solution: 1.7 cm from about
    # 38.5 m above the ground. (198 m is the drone's height above sea level,
    # not above the ground.)
    block = state.get("droneBlock") or {}
    gsd = block.get("gsdM") or (block.get("flyingHeightM", 38.5) * camera["pixelPitchMm"] / camera["focalMm"])
    height = block.get("flyingHeightM") or gsd * camera["focalMm"] / camera["pixelPitchMm"]
    print(f"  cameras against the GPS-based solution, after scale, rotation and shift: "
          f"horizontal {horizontal * 100:.1f} cm ({horizontal / gsd:.1f} px), vertical {vertical * 100:.1f} cm "
          f"({vertical / gsd:.1f} px), worst {err.max() * 100:.1f} cm; {np.sqrt(np.mean(err ** 2)) / height * 100:.2f}% "
          f"of the {height:.1f} m flying height; ground pixel {gsd * 100:.1f} cm")
    # Both solutions carry error: the GPS one is held by GPS of a few metres
    # and the photos' own fit. Agreement to half a percent of the flying height
    # says the two describe the same block.
    check("cameras agree with the GPS-based solution", np.sqrt(np.mean(err ** 2)) < 0.005 * height,
          f"RMS {np.sqrt(np.mean(err ** 2)) * 100:.1f} cm = {np.sqrt(np.mean(err ** 2)) / gsd:.1f} ground pixels, "
          f"{np.sqrt(np.mean(err ** 2)) / height * 100:.2f}% of the flying height")
    check("focal length judged from the evidence", rs["focalSource"] == "metadata",
          f"kept the {rs['focalSource']} focal length: flat ground seen from above cannot fix it")
    check("reprojection on real photos", rs["reprojectionRmsPx"] < 1.0, f"{rs['reprojectionRmsPx']:.2f} px RMS")
    print(f"  lens: focal {real_result['lens']['focal']:.0f} px (metadata {focal_px:.0f}), "
          f"k1 {real_result['lens']['k1']:.4f}, k2 {real_result['lens']['k2']:.4f}; "
          f"{match_summary['pairsMatched']} of {match_summary['pairsTried']} pairs matched")
else:
    print("\n  (part 2 skipped: set FIDUCIA_BRIGHTON and FIDUCIA_BRIGHTON_PROJECT)")

print("\n" + "=" * 64)
print(f"  {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("  FAILED: " + ", ".join(FAIL))
print("=" * 64)
sys.exit(1 if FAIL else 0)
