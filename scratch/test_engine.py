"""End-to-end validation of the Fiducia engine against synthetic data.

Every test builds a scene whose answer is known exactly, runs the engine's
solver, and checks it recovers the truth. If the collinearity conventions,
the Jacobian signs, or the interior orientation were wrong, these fail.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))

from fiducia.camera import CameraModel, fit_fiducial_transform, fit_radial_from_table
from fiducia.collinearity import project_points, rotation_matrix, collinearity_jacobian
from fiducia.resection import resect
from fiducia.bundle import BundleInput, Observation, adjust_block

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


# A representative 153 mm aerial mapping camera.
FOCAL = 153.690
SCALE = 20000.0
FLYING_HEIGHT = FOCAL / 1000.0 * SCALE   # 3073.8 m


def make_ground(n=12, seed=0, extent=1500.0, relief=180.0):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-extent, extent, n)
    ys = rng.uniform(-extent, extent, n)
    zs = rng.uniform(0, relief, n)
    return np.column_stack([xs + 50000.0, ys + 6200000.0, zs])


print("\n=== 1. Rotation matrix ===")
R = rotation_matrix(0.03, -0.02, 1.1)
check("orthonormal", np.allclose(R @ R.T, np.eye(3), atol=1e-12),
      f"max dev {np.abs(R @ R.T - np.eye(3)).max():.2e}")
check("determinant +1", abs(np.linalg.det(R) - 1.0) < 1e-12)


print("\n=== 2. Analytic Jacobian vs finite differences ===")
eo = np.array([50000.0, 6200000.0, FLYING_HEIGHT, 0.02, -0.015, 0.8])
ground = make_ground(6, seed=1)
d_eo, d_ground = collinearity_jacobian(ground, eo, FOCAL)

numeric = np.zeros_like(d_eo)
for k in range(6):
    step = 1e-4 if k < 3 else 1e-7
    plus, minus = eo.copy(), eo.copy()
    plus[k] += step
    minus[k] -= step
    numeric[:, :, k] = (project_points(ground, plus, FOCAL)
                        - project_points(ground, minus, FOCAL)) / (2 * step)

err = np.abs(numeric - d_eo).max()
check("EO Jacobian matches finite differences", err < 1e-4, f"max error {err:.3e}")

numeric_g = np.zeros_like(d_ground)
for k in range(3):
    step = 1e-3
    plus, minus = ground.copy(), ground.copy()
    plus[:, k] += step
    minus[:, k] -= step
    numeric_g[:, :, k] = (project_points(plus, eo, FOCAL)
                          - project_points(minus, eo, FOCAL)) / (2 * step)
err_g = np.abs(numeric_g - d_ground).max()
check("Ground Jacobian matches finite differences", err_g < 1e-5, f"max error {err_g:.3e}")


print("\n=== 3. Distortion round trip ===")
cam = CameraModel(
    kind="film", focal_mm=FOCAL,
    ppo_x_mm=-0.013, ppo_y_mm=0.009,
    k0=3.52886e-05, k1=-3.95205e-09, k2=-1.01226e-13, k3=1.00869e-17,
    image_scale=SCALE,
)
pts = np.column_stack([np.linspace(-110, 110, 25), np.linspace(105, -105, 25)])
back = cam.apply_distortion(cam.remove_distortion(pts))
err = np.abs(back - pts).max()
check("remove -> apply is identity", err < 1e-7, f"max error {err:.2e} mm")

r = np.hypot(pts[:, 0], pts[:, 1])
delta = np.abs(cam.remove_distortion(pts) - pts).max()
check("distortion magnitude is physical", 1e-4 < delta < 0.2,
      f"{delta * 1000:.1f} um at r={r.max():.0f}mm")


print("\n=== 4. Distortion table fit ===")
# Synthesise a certificate table from known coefficients, then recover them.
radii = np.array([10, 20, 40, 60, 80, 100, 120, 140], dtype=float)
true = dict(k0=3.52886e-05, k1=-3.95205e-09, k2=-1.01226e-13, k3=1.00869e-17)
dr_mm = (radii * true["k0"] + radii**3 * true["k1"]
         + radii**5 * true["k2"] + radii**7 * true["k3"])
fit = fit_radial_from_table(radii, dr_mm * 1000.0)
recovered = np.array([fit["k0"], fit["k1"], fit["k2"], fit["k3"]])
expected = np.array([true["k0"], true["k1"], true["k2"], true["k3"]])
rel = np.abs(recovered - expected) / np.maximum(np.abs(expected), 1e-30)
check("recovers K0..K3 from a table", fit["rmsUm"] < 0.02,
      f"fit RMS {fit['rmsUm']:.4f} um, max rel err {rel.max():.2%}")


print("\n=== 5. Fiducial interior orientation ===")
# Calibrated marks from the screenshot of the real camera.
calibrated = {
    "top_left": (-106.005, 106.005), "top_middle": (-0.007, 111.996),
    "top_right": (105.996, 105.996), "right_middle": (112.003, -0.002),
    "bottom_right": (106.004, -106.004), "bottom_middle": (0.000, -111.995),
    "bottom_left": (-106.000, -105.999), "left_middle": (-112.010, 0.004),
}
# Simulate a scan: 15 um pixels, slight rotation and offset.
pixel_size, theta = 0.015, np.radians(0.4)
cos, sin = np.cos(theta), np.sin(theta)
measured = {}
for slot, (mx, my) in calibrated.items():
    col = (mx * cos + my * sin) / pixel_size + 7000
    row = -(-mx * sin + my * cos) / pixel_size + 7000
    measured[slot] = (col, row)

fit = fit_fiducial_transform(measured, calibrated)
check("affine fit is near-exact on clean marks", fit.rms_px < 0.01,
      f"RMS {fit.rms_px:.5f} px, {fit.transform_kind}")

cam.fiducials_mm = calibrated
round_trip = cam.film_to_pixel(cam.pixel_to_film([[7000, 7000]], fit), fit)
check("pixel -> film -> pixel round trip", np.abs(round_trip - 7000).max() < 1e-4,
      f"error {np.abs(round_trip - 7000).max():.2e} px")

# A deliberately mis-clicked mark must show up.
bad = dict(measured)
bad["top_right"] = (bad["top_right"][0] + 40, bad["top_right"][1] - 25)
bad_fit = fit_fiducial_transform(bad, calibrated)
check("a blundered mark raises the residual", bad_fit.max_px > 5.0,
      f"max residual {bad_fit.max_px:.2f} px")


print("\n=== 6. Space resection recovers known orientation ===")
truth = np.array([50120.0, 6200080.0, FLYING_HEIGHT, 0.018, -0.011, 0.62])
ground = make_ground(8, seed=3)
film = project_points(ground, truth, FOCAL)

result = resect(film, ground, FOCAL, image_scale=SCALE)
pos_err = np.linalg.norm(result.eo[:3] - truth[:3])
ang_err = np.degrees(np.abs(result.eo[3:] - truth[3:])).max()
check("converged", result.converged, result.message)
check("perspective centre within 1 mm", pos_err < 1e-3, f"{pos_err * 1000:.4f} mm")
check("angles within 1e-5 deg", ang_err < 1e-5, f"{ang_err:.2e} deg")
check("residual is essentially zero", result.rms_mm < 1e-6, f"{result.rms_mm:.2e} mm")

# With realistic measurement noise the solution should stay close.
rng = np.random.default_rng(7)
noisy = film + rng.normal(0, 0.010, film.shape)      # 10 um, a good operator
noisy_result = resect(noisy, ground, FOCAL, image_scale=SCALE)
noisy_pos = np.linalg.norm(noisy_result.eo[:3] - truth[:3])
check("robust to 10 um measuring noise", noisy_pos < 5.0,
      f"centre off by {noisy_pos:.2f} m, sigma0 {noisy_result.sigma0:.4f}")

# Off-north imagery — the case that breaks a naive initial guess.
rotated = truth.copy()
rotated[5] = 2.9
film_rot = project_points(ground, rotated, FOCAL)
rot_result = resect(film_rot, ground, FOCAL, image_scale=SCALE)
rot_err = np.linalg.norm(rot_result.eo[:3] - rotated[:3])
check("handles kappa = 166 deg", rot_err < 1e-2, f"centre off by {rot_err:.5f} m")


print("\n=== 7. Bundle block adjustment, 3-photo strip ===")
# A three-photo strip with 60% forward overlap.
base = 800.0
truths = {
    "photo_a": np.array([49200.0, 6200000.0, FLYING_HEIGHT, 0.012, -0.008, 0.05]),
    "photo_b": np.array([49200.0 + base, 6200040.0, FLYING_HEIGHT + 18, -0.006, 0.010, 0.03]),
    "photo_c": np.array([49200.0 + 2 * base, 6199960.0, FLYING_HEIGHT - 12, 0.009, 0.004, -0.02]),
}

rng = np.random.default_rng(11)
control_pts, tie_pts = {}, {}
for i in range(6):
    control_pts[f"G{i:04d}"] = np.array([
        49100.0 + rng.uniform(0, 1800), 6200000.0 + rng.uniform(-700, 700), rng.uniform(20, 190)
    ])
for i in range(40):
    tie_pts[f"T{i:04d}"] = np.array([
        49100.0 + rng.uniform(0, 1800), 6200000.0 + rng.uniform(-700, 700), rng.uniform(20, 190)
    ])

all_points = {**control_pts, **tie_pts}
observations = []
half_format = 105.0   # mm, so a point must land on the film to be seen

for image_id, eo_true in truths.items():
    ids = list(all_points.keys())
    coords = np.array([all_points[p] for p in ids])
    projected = project_points(coords, eo_true, FOCAL)
    for point_id, (x, y) in zip(ids, projected):
        if not np.isfinite(x) or abs(x) > half_format or abs(y) > half_format:
            continue
        observations.append(Observation(
            image_id=image_id, point_id=point_id,
            x_mm=x + rng.normal(0, 0.008),
            y_mm=y + rng.normal(0, 0.008),
        ))

seen = {}
for o in observations:
    seen.setdefault(o.point_id, 0)
    seen[o.point_id] += 1
multi = sum(1 for v in seen.values() if v > 1)
print(f"  ({len(observations)} observations, {len(seen)} points, {multi} multi-ray)")

data = BundleInput(
    image_ids=list(truths.keys()),
    focal_by_image={k: FOCAL for k in truths},
    observations=observations,
    control={k: tuple(v) for k, v in control_pts.items()},
    check_points=set(),
    control_sigma_m=0.3,
    image_sigma_mm=0.008,
    image_scale=SCALE,
)

result = adjust_block(data)
check("converged", result.converged, result.message)

worst_pos, worst_ang = 0.0, 0.0
for image_id, eo_true in truths.items():
    solved = result.eo[image_id]
    worst_pos = max(worst_pos, float(np.linalg.norm(solved[:3] - eo_true[:3])))
    worst_ang = max(worst_ang, float(np.degrees(np.abs(solved[3:] - eo_true[3:])).max()))

check("all perspective centres within 3 m", worst_pos < 3.0, f"worst {worst_pos:.3f} m")
check("all angles within 0.05 deg", worst_ang < 0.05, f"worst {worst_ang:.4f} deg")
check("image residual near the noise floor", result.rms_image_mm < 0.03,
      f"RMS {result.rms_image_mm * 1000:.2f} um (noise was 8 um)")

tie_err = [
    float(np.linalg.norm(result.object_points[p] - tie_pts[p]))
    for p in tie_pts if p in result.object_points
]
check("tie points triangulate within 3 m", np.max(tie_err) < 3.0,
      f"worst {np.max(tie_err):.3f} m, mean {np.mean(tie_err):.3f} m")


print("\n=== 8. A blundered GCP is exposed, not absorbed ===")
blunder_control = {k: v.copy() for k, v in control_pts.items()}
blunder_control["G0002"] = blunder_control["G0002"] + np.array([25.0, -18.0, 4.0])

data_bad = BundleInput(
    image_ids=list(truths.keys()),
    focal_by_image={k: FOCAL for k in truths},
    observations=observations,
    control={k: tuple(v) for k, v in blunder_control.items()},
    check_points=set(),
    control_sigma_m=0.3,
    image_sigma_mm=0.008,
    image_scale=SCALE,
)
bad_result = adjust_block(data_bad)
residual_of_blunder = np.linalg.norm(bad_result.control_residuals_m.get("G0002", (0, 0, 0)))
others = [
    np.linalg.norm(v) for k, v in bad_result.control_residuals_m.items() if k != "G0002"
]
check("the bad point carries the largest residual",
      residual_of_blunder > max(others) * 2,
      f"G0002 {residual_of_blunder:.2f} m vs next worst {max(others):.2f} m")
check("robust weighting keeps good control clean",
      max(others) < 3.0, f"next worst good point {max(others):.2f} m")
names = [s_["pointId"] for s_ in bad_result.suspects]
check("blunder detection names G0002", names[:1] == ["G0002"],
      f"suspects: {names or 'none'}")
if bad_result.suspects:
    print("     -> " + bad_result.suspects[0]["note"])

print("\n" + "=" * 62)
print(f"  {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("  FAILED: " + ", ".join(FAIL))
print("=" * 62)
sys.exit(1 if FAIL else 0)
