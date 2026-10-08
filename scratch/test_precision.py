"""Are the reported precisions true?

A bundle adjustment reports a standard deviation for every parameter. That is
only worth reporting if it is right, and the test of that is Monte Carlo:
solve one block many times over with fresh random measurement noise of known
size, and compare the actual scatter of the answers with the precision the
adjustment claimed. They should agree to within sampling error.

Also checks that per-point control precisions are honoured: a loosely known
control point must pull on the solution less than a well-surveyed one.
"""

import math
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))

from fiducia.bundle import BundleInput, Observation, adjust_block  # noqa: E402
from fiducia.collinearity import project_points                    # noqa: E402

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


rng = np.random.default_rng(2026)
FOCAL = 153.0
SIGMA_IMG = 0.008        # mm: 8 um, a good film measurement
SIGMA_CTRL = 0.05        # m

truth = {
    "P1": np.array([1000.0, 2000.0, 3060.0, 0.010, -0.006, 0.02]),
    "P2": np.array([1920.0, 2010.0, 3055.0, -0.008, 0.012, 0.01]),
    "P3": np.array([2840.0, 1995.0, 3065.0, 0.004, 0.007, -0.015]),
}
names = list(truth)

control = {f"G{i}": np.array([x, y, z]) for i, (x, y, z) in enumerate([
    (900, 1500, 40), (900, 2500, 55), (1900, 1450, 35), (1900, 2550, 60),
    (2950, 1500, 45), (2950, 2500, 50)])}
ties = {f"T{i}": np.array([rng.uniform(800, 3050), rng.uniform(1400, 2600), rng.uniform(20, 70)])
        for i in range(40)}
points = {**control, **ties}

# Which photo sees which point: inside the frame, with a margin.
visible = {}
for name, eo in truth.items():
    ids = list(points)
    film = project_points(np.array([points[p] for p in ids]), eo, FOCAL)
    visible[name] = [p for p, f in zip(ids, film) if np.all(np.abs(f) < 105)]


def solve(noise=True):
    obs = []
    for name in names:
        ids = visible[name]
        film = project_points(np.array([points[p] for p in ids]), truth[name], FOCAL)
        if noise:
            film = film + rng.normal(0, SIGMA_IMG, film.shape)
        obs += [Observation(name, p, float(f[0]), float(f[1])) for p, f in zip(ids, film)]
    ctrl = {p: tuple(xyz + (rng.normal(0, SIGMA_CTRL, 3) if noise else 0)) for p, xyz in control.items()}
    return adjust_block(BundleInput(names, {n: FOCAL for n in names}, obs, ctrl,
                                    control_sigma_m=SIGMA_CTRL, image_sigma_mm=SIGMA_IMG,
                                    robust=False))


print("\n=== 1. The block, and its reported precision ===")
exact = solve(noise=False)
check("exact data recovers the truth", max(np.max(np.abs(exact.eo[n] - truth[n])) for n in names) < 1e-6)
check("precision is reported for every photo and point",
      set(exact.eo_sigma) == set(names) and set(exact.point_sigma) >= set(ties))

TRIALS = 150
solutions = {n: [] for n in names}
tie_solutions = {p: [] for p in list(ties)[:10]}
reported = {n: [] for n in names}
reported_tie = {p: [] for p in tie_solutions}
sigma0s = []
covariances = {n: [] for n in names}
for _ in range(TRIALS):
    r = solve()
    for n in names:
        covariances[n].append(np.array(r.eo_covariance[n]))
    sigma0s.append(r.sigma0)
    for n in names:
        solutions[n].append(r.eo[n])
        reported[n].append(r.eo_sigma[n])
    for p in tie_solutions:
        tie_solutions[p].append(r.object_points[p])
        reported_tie[p].append(r.point_sigma[p])

print(f"\n=== 2. Monte Carlo, {TRIALS} noisy solves ===")
check("sigma0 averages 1 when the weights are right", abs(np.mean(sigma0s) - 1) < 0.08,
      f"mean sigma0 {np.mean(sigma0s):.3f}")

labels = ["X0 m", "Y0 m", "Z0 m", "omega\"", "phi\"", "kappa\""]
ratios = []
print(f"  {'photo':<6}" + "".join(f"{l:>20}" for l in labels))
for n in names:
    actual = np.std(np.array(solutions[n]), axis=0, ddof=1)
    claimed = np.sqrt(np.mean(np.square(np.array(reported[n])), axis=0))
    ratios.extend(actual / claimed)
    cells = []
    for k in range(6):
        f = 206264.8 if k >= 3 else 1.0
        cells.append(f"{actual[k] * f:8.3f}/{claimed[k] * f:<8.3f}")
    print(f"  {n:<6}" + "".join(f"{c:>20}" for c in cells))
print("  (actual scatter / reported sigma)")
ratios = np.array(ratios)
# With 150 trials the sampling error of a standard deviation is about 6%.
check("reported orientation precision matches the actual scatter",
      np.all((ratios > 0.8) & (ratios < 1.2)),
      f"ratio actual/reported between {ratios.min():.2f} and {ratios.max():.2f}")

tie_ratios = []
for p in tie_solutions:
    actual = np.std(np.array(tie_solutions[p]), axis=0, ddof=1)
    claimed = np.sqrt(np.mean(np.square(np.array(reported_tie[p])), axis=0))
    tie_ratios.extend(actual / claimed)
tie_ratios = np.array(tie_ratios)
check("reported tie-point precision matches the actual scatter",
      np.all((tie_ratios > 0.75) & (tie_ratios < 1.25)),
      f"ratio between {tie_ratios.min():.2f} and {tie_ratios.max():.2f}")

bias = max(np.max(np.abs(np.mean(np.array(solutions[n]), axis=0) - truth[n])
                  / np.sqrt(np.mean(np.square(np.array(reported[n])), axis=0))) for n in names)
check("the estimates are unbiased", bias < 0.35, f"worst mean offset {bias:.2f} sigma")

print("\n=== 3. Corner uncertainty on the ground ===")
from fiducia.bundle import corner_uncertainty  # noqa: E402
from fiducia.collinearity import rotation_matrix  # noqa: E402


def corner_ground(eo, corner, z=45.0):
    direction = rotation_matrix(*eo[3:]).T @ np.array([corner[0], corner[1], -FOCAL])
    t = (z - eo[2]) / direction[2]
    return eo[:2] + t * direction[:2]


worst_ratio = []
for n in names:
    positions = {c: np.array([corner_ground(e, c) for e in solutions[n]])
                 for c in ((105, 105), (105, -105), (-105, 105), (-105, -105))}
    actual = max(float(np.sqrt(np.trace(np.cov(p.T)))) for p in positions.values())
    # The covariance each solve reports, averaged (a noise-free solve has
    # sigma0 = 0 and so reports none).
    predicted = corner_uncertainty(truth[n], np.mean(covariances[n], axis=0), FOCAL, 105, 105, 45.0)
    worst_ratio.append(actual / predicted)
    print(f"  {n}: corners scatter {actual:.3f} m, predicted {predicted:.3f} m")
check("predicted corner uncertainty matches the actual scatter",
      all(0.8 < r < 1.2 for r in worst_ratio), ", ".join(f"{r:.2f}" for r in worst_ratio))
mean_cov = np.mean(covariances["P1"], axis=0)
diagonal_only = corner_uncertainty(truth["P1"], np.diag(np.diag(mean_cov)), FOCAL, 105, 105, 45.0)
full = corner_uncertainty(truth["P1"], mean_cov, FOCAL, 105, 105, 45.0)
check("correlations matter: ignoring them would overstate it", diagonal_only > 1.5 * full,
      f"{diagonal_only:.3f} m from standard deviations alone vs {full:.3f} m with the full covariance")

print("\n=== 4. Single-photo resection precision ===")
import server  # noqa: E402
from fiducia.resection import resect  # noqa: E402

ids = [p for p in visible["P1"] if p in control] + [p for p in visible["P1"] if p in ties][:6]
ground = np.array([points[p] for p in ids])
film_true = project_points(ground, truth["P1"], FOCAL)
solved, claimed = [], []
for _ in range(300):
    noisy = film_true + rng.normal(0, SIGMA_IMG, film_true.shape)
    r = resect(noisy, ground, FOCAL)
    cov = server._resection_covariance(r, noisy, ground, FOCAL)
    solved.append(r.eo)
    claimed.append(np.sqrt(np.diag(np.array(cov))))
ratio = np.std(np.array(solved), axis=0, ddof=1) / np.sqrt(np.mean(np.square(np.array(claimed)), axis=0))
check("reported resection precision matches the actual scatter",
      np.all((ratio > 0.8) & (ratio < 1.25)), f"ratio between {ratio.min():.2f} and {ratio.max():.2f}, "
      f"{len(ids)} points")

print("\n=== 5. Per-point control precision is honoured ===")
# Displace one control point by 1 m. Declared at 5 cm it should drag the
# solution; declared at 5 m it should barely matter.
def displaced(sigma_for_g0):
    obs = []
    for name in names:
        ids = visible[name]
        film = project_points(np.array([points[p] for p in ids]), truth[name], FOCAL)
        obs += [Observation(name, p, float(f[0]), float(f[1])) for p, f in zip(ids, film)]
    ctrl = {p: tuple(xyz) for p, xyz in control.items()}
    ctrl["G0"] = tuple(control["G0"] + np.array([1.0, 0, 0]))
    r = adjust_block(BundleInput(names, {n: FOCAL for n in names}, obs, ctrl,
                                 control_sigma_m=SIGMA_CTRL, image_sigma_mm=SIGMA_IMG, robust=False,
                                 control_sigma_by_point={"G0": sigma_for_g0}))
    return np.linalg.norm(r.eo["P1"][:3] - truth["P1"][:3])

tight, loose = displaced(0.05), displaced(5.0)
check("a loosely surveyed point pulls the solution far less", loose < tight / 5,
      f"photo P1 moved {tight:.3f} m with G0 at 5 cm, {loose:.3f} m with G0 at 5 m")

print("\n" + "=" * 64)
print(f"  TOTAL {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("  FAILED: " + ", ".join(FAIL))
print("=" * 64)
sys.exit(1 if FAIL else 0)
