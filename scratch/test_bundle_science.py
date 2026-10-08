"""Statistical honesty of the bundle adjustment, on synthetic blocks.

A block is built whose truth is known, measured with Gaussian noise of a
known standard deviation, and adjusted. A sound adjustment then:

* reports a variance factor (sigma-0) close to 1 when the a-priori
  precisions are right,
* has redundancy numbers that sum to its degrees of freedom,
* recovers camera errors by self-calibration to within their reported
  standard deviations,
* solves from GNSS camera positions alone, without ground control,
* finds a blundered measurement by data snooping,
* keeps its precision at Lo-zone coordinates in the millions of metres.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))

from fiducia.bundle import (BundleInput, Observation, adjust_block,  # noqa: E402
                            apply_additional_parameters)
from fiducia.collinearity import project_points  # noqa: E402

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  - {detail}" if detail else ""))


FOCAL = 100.5            # mm, a large-format digital frame
PIXEL = 0.0052           # mm
HALF_W, HALF_H = 13080 / 2 * PIXEL, 20010 / 2 * PIXEL
HEIGHT = 3000.0          # m above ground
ORIGIN = np.array([-42000.0, -3743000.0, 40.0])   # Lo19 east/north magnitudes


def make_block(seed=0, strips=2, per_strip=5, true_aps=None, radius=None):
    """Two strips at 60% forward and 30% side overlap over rolling terrain."""
    rng = np.random.default_rng(seed)
    ground_w = 2 * HALF_W / FOCAL * HEIGHT     # across track, m
    ground_h = 2 * HALF_H / FOCAL * HEIGHT     # along track, m
    base = 0.4 * ground_h
    spacing = 0.7 * ground_w

    images = {}
    for s in range(strips):
        for k in range(per_strip):
            img = f"s{s}p{k}"
            images[img] = np.array([
                ORIGIN[0] + s * spacing + rng.normal(0, 5),
                ORIGIN[1] + k * base + rng.normal(0, 5),
                ORIGIN[2] + HEIGHT + rng.normal(0, 10),
                rng.normal(0, 0.01), rng.normal(0, 0.01), rng.normal(0, 0.01),
            ])

    def terrain(x, y):
        return ORIGIN[2] + 40 * np.sin((x - ORIGIN[0]) / 900) + 30 * np.cos((y - ORIGIN[1]) / 700)

    xs = rng.uniform(ORIGIN[0] - 0.45 * ground_w, ORIGIN[0] + (strips - 1) * spacing + 0.45 * ground_w, 900)
    ys = rng.uniform(ORIGIN[1] - 0.45 * ground_h, ORIGIN[1] + (per_strip - 1) * base + 0.45 * ground_h, 900)
    points = {f"P{i:04d}": np.array([x, y, terrain(x, y)]) for i, (x, y) in enumerate(zip(xs, ys))}

    radius = radius or float(np.hypot(HALF_W, HALF_H))
    truth = {}
    for img, eo in images.items():
        for pid, xyz in points.items():
            xy = project_points(xyz[None, :], eo, FOCAL)[0]
            if abs(xy[0]) < HALF_W * 0.97 and abs(xy[1]) < HALF_H * 0.97:
                if true_aps:
                    xy = xy + apply_additional_parameters(xy[None, :], true_aps, FOCAL, radius)[0]
                truth[(img, pid)] = xy
    return images, points, truth, radius


def observe(truth, sigma_mm, seed=1):
    rng = np.random.default_rng(seed)
    return [Observation(img, pid, float(xy[0] + rng.normal(0, sigma_mm)),
                        float(xy[1] + rng.normal(0, sigma_mm)))
            for (img, pid), xy in truth.items()]


def control_split(points, n_control=12, n_check=10, seed=2):
    rng = np.random.default_rng(seed)
    ids = list(points)
    rng.shuffle(ids)
    return ids[:n_control], ids[n_control:n_control + n_check]


SIGMA = 0.5 * PIXEL


print("\n=== 1. Variance factor and redundancy with correct weights ===")
images, points, truth, _ = make_block()
obs = observe(truth, SIGMA)
control_ids, check_ids = control_split(points)
rng = np.random.default_rng(9)
control = {pid: tuple(points[pid] + rng.normal(0, [0.05, 0.05, 0.08])) for pid in control_ids}
control.update({pid: tuple(points[pid]) for pid in check_ids})
data = BundleInput(
    image_ids=list(images), focal_by_image={i: FOCAL for i in images}, observations=obs,
    control=control, check_points=set(check_ids), control_sigma_m=0.05, control_sigma_z_m=0.08,
    image_sigma_mm=SIGMA, robust=False,
)
result = adjust_block(data)
check("converged", result.converged, result.message)
check("sigma0 within 10% of 1", abs(result.sigma0 - 1.0) < 0.1, f"sigma0 = {result.sigma0:.3f}")
check("redundancy numbers sum to the degrees of freedom",
      result.redundancy_total is not None
      and abs(result.redundancy_total - result.degrees_of_freedom) < 0.01 * result.degrees_of_freedom,
      f"sum r = {result.redundancy_total}, dof = {result.degrees_of_freedom}")
errors = np.array([result.eo[i][:3] - images[i][:3] for i in images])
sigmas = np.array([result.eo_sigma[i][:3] for i in images])
z = (errors / sigmas).ravel()
check("camera position errors consistent with reported sigma (|z| rms 0.6-1.6)",
      0.6 < float(np.sqrt(np.mean(z ** 2))) < 1.6, f"rms z = {np.sqrt(np.mean(z ** 2)):.2f}")
check("check point RMS below 0.3 m", result.rms_check_m < 0.3,
      f"{result.rms_check_m:.3f} m, per axis {np.round(result.rms_check_xyz_m, 3)}")
check("no suspects in clean data", len(result.suspects) <= 1, f"{len(result.suspects)} flagged")


print("\n=== 2. Self-calibration recovers injected camera errors ===")
true_aps = {"df": 0.030, "dx0": 0.012, "dy0": -0.008, "k1": 4e-5 * 60, "p1": 0.0, "p2": 0.0}
images, points, truth, radius = make_block(seed=3, strips=3, true_aps=true_aps)
obs = observe(truth, SIGMA, seed=4)
control_ids, check_ids = control_split(points, 20, 12, seed=5)
control = {pid: tuple(points[pid]) for pid in control_ids + check_ids}
base = dict(image_ids=list(images), focal_by_image={i: FOCAL for i in images}, observations=obs,
            control=control, check_points=set(check_ids), control_sigma_m=0.03,
            image_sigma_mm=SIGMA, robust=False)
plain = adjust_block(BundleInput(**base))
calibrated = adjust_block(BundleInput(**base, self_calibration="radial"))
check("uncorrected camera inflates sigma0", plain.sigma0 > 1.5, f"sigma0 = {plain.sigma0:.2f}")
check("self-calibration brings sigma0 back to ~1", abs(calibrated.sigma0 - 1.0) < 0.15,
      f"sigma0 = {calibrated.sigma0:.3f}")
for name in ("df", "dx0", "dy0", "k1"):
    value, sigma = calibrated.additional_parameters[name]
    truth_value = true_aps.get(name, 0.0) * (radius / radius)
    check(f"{name} recovered within 3 sigma", sigma and abs(value - truth_value) < 3 * sigma + 1e-9,
          f"{value:.5f} +/- {sigma:.5f}, truth {truth_value:.5f}")
check("self-calibration improves check points", calibrated.rms_check_m < plain.rms_check_m,
      f"{plain.rms_check_m:.3f} -> {calibrated.rms_check_m:.3f} m")


print("\n=== 3. GNSS-assisted, no ground control ===")
images, points, truth, _ = make_block(seed=6)
obs = observe(truth, SIGMA, seed=7)
_, check_ids = control_split(points, 0, 15, seed=8)
rng = np.random.default_rng(10)
gnss = {i: [*(images[i][:3] + rng.normal(0, [0.05, 0.05, 0.08])), None, None, None] for i in images}
result = adjust_block(BundleInput(
    image_ids=list(images), focal_by_image={i: FOCAL for i in images}, observations=obs,
    control={pid: tuple(points[pid]) for pid in check_ids}, check_points=set(check_ids),
    image_sigma_mm=SIGMA, eo_observations=gnss, eo_sigma_xy_m=0.05, eo_sigma_z_m=0.08,
    robust=False,
))
check("solves from camera positions alone", result.converged, result.message)
check("sigma0 ~1", abs(result.sigma0 - 1.0) < 0.15, f"{result.sigma0:.3f}")
check("check points within 0.5 m", result.rms_check_m < 0.5,
      f"{result.rms_check_m:.3f} m, per axis {np.round(result.rms_check_xyz_m, 3)}")

print("\n=== 3b. GNSS with a constant offset, solved as a shift ===")
offset = np.array([0.8, -0.5, 1.2])
shifted = {i: [*(np.array(v[:3]) + offset), None, None, None] for i, v in gnss.items()}
control_ids, check_ids2 = control_split(points, 4, 12, seed=11)
result = adjust_block(BundleInput(
    image_ids=list(images), focal_by_image={i: FOCAL for i in images}, observations=obs,
    control={pid: tuple(points[pid]) for pid in control_ids + check_ids2},
    check_points=set(check_ids2), control_sigma_m=0.03, image_sigma_mm=SIGMA,
    eo_observations=shifted, eo_sigma_xy_m=0.05, eo_sigma_z_m=0.08, gnss_shift=True, robust=False,
))
check("shift recovered to 0.15 m", result.gnss_shift_m is not None
      and np.allclose(result.gnss_shift_m, offset, atol=0.15), f"{np.round(result.gnss_shift_m, 3)}")


print("\n=== 4. Data snooping finds blunders ===")
images, points, truth, _ = make_block(seed=12)
obs = observe(truth, SIGMA, seed=13)
control_ids, check_ids = control_split(points, 14, 8, seed=14)
control = {pid: tuple(points[pid]) for pid in control_ids + check_ids}
rays = {}
for o in obs:
    rays[o.point_id] = rays.get(o.point_id, 0) + 1
bad_tie = next(o for o in obs if o.point_id not in control and rays[o.point_id] >= 3)
bad_tie.x_mm += 12 * PIXEL
bad_gcp = control_ids[0]
control[bad_gcp] = (control[bad_gcp][0], control[bad_gcp][1], control[bad_gcp][2] + 4.0)
ties = {o.point_id for o in obs if o.point_id not in control}
setup = dict(image_ids=list(images), focal_by_image={i: FOCAL for i in images},
             control_sigma_m=0.05, image_sigma_mm=SIGMA, robust=True, automatic_ties=ties)
clean = adjust_block(BundleInput(**setup, observations=observe(truth, SIGMA, seed=13),
                                 control={pid: tuple(points[pid]) for pid in control_ids + check_ids},
                                 check_points=set(check_ids)))
result = adjust_block(BundleInput(**setup, observations=obs, control=control,
                                  check_points=set(check_ids)))
flagged = [s["pointId"] for s in result.suspects]
check("blundered tie removed automatically", bad_tie.point_id in result.rejected_ties,
      f"rejected {list(result.rejected_ties)[:5]}")
check("blundered control height flagged first", flagged[:1] == [bad_gcp], f"flagged {flagged[:5]}")
check("flagged control is not removed automatically",
      bad_gcp not in result.rejected_ties and bad_gcp not in result.single_ray_points)
# The operator's response: make the flagged point a check point.
fixed = adjust_block(BundleInput(**setup, observations=obs, control=control,
                                 check_points=set(check_ids) | {bad_gcp}))
# Check points seen on one photo check position only (dZ is NaN).
clean_check = np.sqrt(np.nanmean(np.square(np.array(
    [clean.control_residuals_m[p] for p in check_ids if p in clean.control_residuals_m],
    dtype=float)), axis=0))
fixed_check = np.sqrt(np.nanmean(np.square(np.array(
    [fixed.control_residuals_m[p] for p in check_ids if p in fixed.control_residuals_m],
    dtype=float)), axis=0))
finite = np.isfinite(clean_check) & np.isfinite(fixed_check)
check("with the flagged point excluded the block matches the clean one (within 20%)",
      finite[:2].all() and np.all(fixed_check[finite] <= 1.2 * clean_check[finite] + 0.02),
      f"clean {np.round(clean_check, 3)}, after {np.round(fixed_check, 3)}")
check("and its own residual shows the blunder (~4 m in height)",
      abs(fixed.control_residuals_m[bad_gcp][2] - 4.0) < 0.5,
      f"dZ = {fixed.control_residuals_m[bad_gcp][2]:.2f} m")


print(f"\n{'=' * 64}\n  TOTAL {len(PASS)} passed, {len(FAIL)} failed\n{'=' * 64}")
if FAIL:
    print("  FAILED:", ", ".join(FAIL))
    sys.exit(1)
