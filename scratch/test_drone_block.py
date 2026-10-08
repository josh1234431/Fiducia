"""A synthetic drone block with known truth.

Photos flown in a lawnmower pattern over rolling ground by a consumer camera
with real lens distortion, measured to half a pixel, placed by metre-level
GNSS and nothing else -- the situation the drone workflow is built for. The
adjustment must recover the lens, fit to the noise it was given, and shape
the ground correctly; its absolute placement can only be as good as the GNSS.

    python scratch/test_drone_block.py              the checks
    python scratch/test_drone_block.py time 8 16    time a block of 8 strips x 16 photos
"""

import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))

from fiducia.bundle import BundleInput, Observation, adjust_block, apply_additional_parameters  # noqa: E402
from fiducia.collinearity import project_points  # noqa: E402

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


FOCAL = 3.61                 # mm
PITCH = 0.00156              # mm
COLS, ROWS = 4000, 3000
HEIGHT = 60.0                # m above the ground
GROUND = 100.0               # m, mean ground height
NOISE_PX = 0.5
GNSS_XY, GNSS_Z = 1.5, 2.0   # m
LENS = {"dx0": 0.02, "dy0": -0.08, "k1": 0.012, "k2": 0.0, "k3": 0.01, "p1": 0.002, "p2": -0.001}


def terrain(x, y):
    return GROUND + 5.0 * np.sin(x / 40.0) + 3.0 * np.cos(y / 30.0)


def make_block(strips, per_strip, seed=7, points_per_photo=150):
    rng = np.random.default_rng(seed)
    half_w, half_h = COLS / 2 * PITCH, ROWS / 2 * PITCH
    across = 2 * half_w * HEIGHT / FOCAL       # footprint along x
    along = 2 * half_h * HEIGHT / FOCAL        # footprint along y
    side_step, forward_step = across * 0.35, along * 0.25   # 65% side, 75% forward overlap

    truth_eo, observed_eo = {}, {}
    for s in range(strips):
        for k in range(per_strip):
            img = f"P{s:02d}_{k:03d}"
            x, y = s * side_step, k * forward_step
            z = GROUND + HEIGHT + rng.normal(0, 0.5)
            angles = rng.normal(0, math.radians(2.0), 3)
            angles[2] += math.pi if s % 2 else 0.0      # back and forth
            truth_eo[img] = np.array([x, y, z, *angles])
            observed_eo[img] = [x + rng.normal(0, GNSS_XY), y + rng.normal(0, GNSS_XY),
                                z + rng.normal(0, GNSS_Z), 0.0, 0.0,
                                float(angles[2] + rng.normal(0, math.radians(3.0)))]

    xs = np.array([e[0] for e in truth_eo.values()])
    ys = np.array([e[1] for e in truth_eo.values()])
    n_points = points_per_photo * len(truth_eo)
    px = rng.uniform(xs.min() - across / 2, xs.max() + across / 2, n_points)
    py = rng.uniform(ys.min() - along / 2, ys.max() + along / 2, n_points)
    ground = np.column_stack([px, py, terrain(px, py)])

    radius = math.hypot(half_w, half_h)
    by_point = {}
    for img, eo in truth_eo.items():
        near = np.flatnonzero((np.abs(px - eo[0]) < across) & (np.abs(py - eo[1]) < across))
        film = project_points(ground[near], eo, FOCAL)
        film = film + apply_additional_parameters(film, LENS, FOCAL, radius)
        inside = (np.abs(film[:, 0]) < half_w * 0.98) & (np.abs(film[:, 1]) < half_h * 0.98)
        for j, (fx, fy) in zip(near[inside], film[inside]):
            by_point.setdefault(int(j), []).append(
                (img, fx + rng.normal(0, NOISE_PX * PITCH), fy + rng.normal(0, NOISE_PX * PITCH)))

    observations, kept = [], {}
    for j, rays in by_point.items():
        if len(rays) < 2:
            continue
        kept[f"T{j}"] = ground[j]
        observations += [Observation(image_id=i, point_id=f"T{j}", x_mm=fx, y_mm=fy) for i, fx, fy in rays]

    data = BundleInput(
        image_ids=list(truth_eo), focal_by_image={i: FOCAL for i in truth_eo},
        observations=observations, control={}, check_points={}, control_sigma_m=0.5,
        image_sigma_mm=NOISE_PX * PITCH, tie_sigma_mm=NOISE_PX * PITCH,
        eo_observations=observed_eo, eo_sigma_xy_m=GNSS_XY, eo_sigma_z_m=GNSS_Z,
        self_calibration="lens", lens_priors="consumer", ground_z=GROUND, robust=True,
        auto_reject_ties=True, automatic_ties=set(kept),
    )
    return data, truth_eo, kept


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "time":
        strips, per_strip = int(sys.argv[2]), int(sys.argv[3])
        t = time.time()
        data, _, points = make_block(strips, per_strip)
        print(f"block: {strips * per_strip} photos, {len(points)} points, "
              f"{len(data.observations)} observations (built in {time.time() - t:.0f}s)")
        t = time.time()
        result = adjust_block(data)
        print(f"solved in {time.time() - t:.1f}s: converged {result.converged}, "
              f"sigma0 {result.sigma0:.3f}, {result.iterations} evaluations, {result.message}")
        return

    print("Synthetic drone block, 4 strips x 10 photos, GNSS only")
    data, truth_eo, truth_points = make_block(4, 10)
    t = time.time()
    result = adjust_block(data)
    elapsed = time.time() - t
    check("converges", result.converged, f"{result.message}, {elapsed:.0f}s")
    check("sigma0 near 1", 0.8 < result.sigma0 < 1.3, f"{result.sigma0:.3f}")
    rms_px = result.rms_image_mm / PITCH
    check("fits to the measurement noise", rms_px < 0.8, f"{rms_px:.2f} px RMS")

    # Shape, not placement: with GNSS alone the whole block can sit a little
    # shifted, turned and scaled. Remove the best similarity transform, then
    # the ground must be right.
    ids = [p for p in truth_points if p in result.object_points]
    solved = np.array([result.object_points[p] for p in ids])
    true = np.array([truth_points[p] for p in ids])
    mu_s, mu_t = solved.mean(axis=0), true.mean(axis=0)
    a, b = solved - mu_s, true - mu_t
    u, sv, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    rotation = (u @ np.diag([1, 1, d]) @ vt).T
    scale = float((sv * [1, 1, d]).sum() / (a ** 2).sum())
    error = scale * a @ rotation.T + mu_t - true
    horizontal = float(np.sqrt(np.mean(np.sum(error[:, :2] ** 2, axis=1))))
    vertical = float(np.sqrt(np.mean(error[:, 2] ** 2)))
    check("ground shape within 5 cm horizontally", horizontal < 0.05, f"{horizontal * 100:.1f} cm RMS")
    check("ground shape within 10 cm vertically", vertical < 0.10, f"{vertical * 100:.1f} cm RMS")
    tilt = math.degrees(math.acos(min(1.0, (np.trace(rotation) - 1) / 2)))
    shift = float(np.linalg.norm(mu_s - mu_t))
    check("placement within the GNSS's reach", shift < 2.0 and tilt < 1.0 and abs(scale - 1) < 0.01,
          f"offset {shift:.2f} m, tilt {tilt:.2f} deg, scale {scale:.4f}")

    # The lens as a whole: k1, k2 and k3 trade against one another, so judge
    # the distortion they describe, in pixels across the frame.
    half_w, half_h = COLS / 2 * PITCH, ROWS / 2 * PITCH
    gx, gy = np.meshgrid(np.linspace(-half_w, half_w, 21), np.linspace(-half_h, half_h, 15))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    solved_lens = {k: v[0] for k, v in result.additional_parameters.items()}
    true_field = apply_additional_parameters(grid, LENS, FOCAL, math.hypot(half_w, half_h))
    solved_field = apply_additional_parameters(grid, solved_lens, FOCAL, result.format_radius_mm)
    worst = float(np.max(np.hypot(*(solved_field - true_field).T))) / PITCH
    largest = float(np.max(np.hypot(*true_field.T))) / PITCH
    check("recovers the lens distortion", worst < 1.0,
          f"worst {worst:.2f} px across the frame, of up to {largest:.0f} px")

    print(f"\n  TOTAL {len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
