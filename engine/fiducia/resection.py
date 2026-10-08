"""Single-photo space resection.

Recovers the six exterior orientation parameters of one photo from three or
more ground control points. Used on its own for single-image orthorectification
and as the initialiser that gives the block
adjustment a starting point good enough to converge.

The hard part of resection is never the least squares -- it is the starting
approximation. A photogrammetric Gauss-Newton diverges readily if kappa is out
by more than about 30 degrees, which is exactly what happens with imagery that
is not flown north-up. So the initial guess here is built properly: a planar
similarity between image and ground fixes kappa and scale, flying height comes
from the photo scale, and only then does the iteration start.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .collinearity import collinearity_jacobian, project_points, rotation_matrix

__all__ = ["ResectionResult", "resect", "initial_exterior_orientation"]


@dataclass
class ResectionResult:
    eo: np.ndarray                 # X0, Y0, Z0, omega, phi, kappa
    converged: bool
    iterations: int
    rms_mm: float
    residuals_mm: np.ndarray       # (N, 2)
    sigma0: float                  # reference variance of unit weight
    message: str

    def to_dict(self) -> dict:
        return {
            "eo": self.eo.tolist(),
            "converged": self.converged,
            "iterations": self.iterations,
            "rmsMm": float(self.rms_mm),
            "sigma0": float(self.sigma0),
            "message": self.message,
        }


def initial_exterior_orientation(
    film_mm: np.ndarray,
    ground: np.ndarray,
    focal_mm: float,
    image_scale: float = 0.0,
) -> np.ndarray:
    """Build a starting exterior orientation from a planar similarity fit.

    Solves ``ground_xy ~ s * R(kappa) * film_xy + t`` in closed form, which
    gives an honest kappa and photo scale even for imagery flown at an
    arbitrary heading. Flying height then follows from the scale.
    """
    film = np.atleast_2d(np.asarray(film_mm, dtype=float))
    grd = np.atleast_2d(np.asarray(ground, dtype=float))
    n = film.shape[0]

    film_c = film - film.mean(axis=0)
    ground_c = grd[:, :2] - grd[:, :2].mean(axis=0)

    # Closed-form 2D Helmert: the complex-number formulation, which is both
    # shorter and better conditioned than stacking a design matrix.
    z_img = film_c[:, 0] + 1j * film_c[:, 1]
    z_grd = ground_c[:, 0] + 1j * ground_c[:, 1]
    denom = np.sum(np.abs(z_img) ** 2)
    if denom < 1e-12 or n < 2:
        scale, kappa = 1.0, 0.0
    else:
        alpha = np.sum(z_grd * np.conj(z_img)) / denom
        scale = float(np.abs(alpha))
        kappa = float(np.angle(alpha))

    mean_z = float(np.mean(grd[:, 2]))

    if scale > 1e-9:
        # `scale` is ground metres per film millimetre, so the photo scale
        # number is 1000 * scale and the flying height above mean ground is
        # f[mm] * scale -> millimetres * (metres per millimetre) = metres.
        height = focal_mm * scale
    elif image_scale:
        height = (focal_mm / 1000.0) * image_scale
    else:
        height = 1000.0

    centroid = grd[:, :2].mean(axis=0)
    # The similarity's angle IS the collinearity kappa, with no sign change.
    # (An earlier version negated it. Gauss-Newton usually recovered from the
    # resulting error, which is how it went unnoticed -- but for a photo
    # flown near 90 or 270 degrees the negated seed is 180 degrees out and
    # the solver settled in a false minimum with plausible residuals.
    # scratch/test_plausibility.py sweeps every heading to keep it honest.)
    return np.array([centroid[0], centroid[1], mean_z + height, 0.0, 0.0, kappa])


def resect(
    film_mm: np.ndarray,
    ground: np.ndarray,
    focal_mm: float,
    weights: np.ndarray | None = None,
    initial: np.ndarray | None = None,
    image_scale: float = 0.0,
    max_iterations: int = 60,
    tolerance: float = 1e-9,
) -> ResectionResult:
    """Least-squares space resection, started from every quarter-turn.

    Collinearity has false minima: a solution rotated half a turn can settle
    with residuals that look entirely reasonable. One seed, however good,
    leaves that to chance; so the solve is started at the closed-form heading
    and at 90, 180 and 270 degrees from it, and the best solution that puts
    the camera above the ground is kept. Four solves of a handful of points
    cost a few milliseconds. With an explicit ``initial`` only that start is
    used.
    """
    film = np.atleast_2d(np.asarray(film_mm, dtype=float))
    grd = np.atleast_2d(np.asarray(ground, dtype=float))
    if initial is not None:
        return _resect_from(film, grd, focal_mm, weights, np.asarray(initial, dtype=float),
                            max_iterations, tolerance)
    if film.shape[0] < 3:
        raise ValueError(f"Space resection needs at least 3 control points, got {film.shape[0]}")

    seed = initial_exterior_orientation(film, grd, focal_mm, image_scale)
    top = float(np.max(grd[:, 2]))
    best = None
    for turn in (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi):
        start = seed.copy()
        start[5] = seed[5] + turn
        result = _resect_from(film, grd, focal_mm, weights, start, max_iterations, tolerance)
        if not np.isfinite(result.rms_mm) or result.eo[2] <= top:
            continue
        key = (not result.converged, result.rms_mm)
        if best is None or key < (not best.converged, best.rms_mm):
            best = result
    if best is None:
        # Nothing physically valid: return the primary attempt, whose result
        # the caller's own checks will reject with a reason.
        return _resect_from(film, grd, focal_mm, weights, seed, max_iterations, tolerance)
    best.eo[5] = float(np.arctan2(np.sin(best.eo[5]), np.cos(best.eo[5])))
    return best


def _resect_from(
    film_mm: np.ndarray,
    ground: np.ndarray,
    focal_mm: float,
    weights: np.ndarray | None,
    initial: np.ndarray,
    max_iterations: int,
    tolerance: float,
) -> ResectionResult:
    """Least-squares space resection from one starting orientation.

    ``film_mm`` is ``(N, 2)`` of distortion-corrected, principal-point-relative
    focal-plane coordinates. ``ground`` is ``(N, 3)`` of map coordinates.
    ``weights`` is an optional per-point weight, used to down-weight points the
    operator has flagged rather than deleting them outright.

    Damping is Levenberg-style: the step is shortened whenever it would make
    the residual worse, which keeps a poor initial kappa from throwing the
    solution into a mirror-image minimum.
    """
    film = np.atleast_2d(np.asarray(film_mm, dtype=float))
    grd = np.atleast_2d(np.asarray(ground, dtype=float))
    n = film.shape[0]

    if n < 3:
        raise ValueError(f"Space resection needs at least 3 control points, got {n}")

    if weights is None:
        weights = np.ones(n)
    weights = np.asarray(weights, dtype=float)
    w = np.repeat(weights, 2)

    eo = np.asarray(initial, dtype=float).copy()

    def residual_vector(params: np.ndarray) -> np.ndarray:
        predicted = project_points(grd, params, focal_mm)
        return (film - predicted).ravel()

    lam = 1e-6
    previous_cost = np.inf
    converged = False
    iteration = 0
    residual = residual_vector(eo)

    for iteration in range(1, max_iterations + 1):
        d_eo, _ = collinearity_jacobian(grd, eo, focal_mm)
        jac = -d_eo.reshape(2 * n, 6)

        valid = np.isfinite(residual) & np.isfinite(jac).all(axis=1)
        if valid.sum() < 6:
            return ResectionResult(
                eo, False, iteration, float("nan"), np.full((n, 2), np.nan), float("nan"),
                "Points project behind the camera -- check the control coordinates",
            )

        jw = jac[valid] * np.sqrt(w[valid])[:, None]
        rw = residual[valid] * np.sqrt(w[valid])

        normal = jw.T @ jw
        gradient = jw.T @ rw
        cost = float(rw @ rw)

        for _ in range(12):
            try:
                step = np.linalg.solve(normal + lam * np.diag(np.diag(normal) + 1e-12), -gradient)
            except np.linalg.LinAlgError:
                lam *= 10
                continue

            candidate = eo + step
            candidate_residual = residual_vector(candidate)
            cand_valid = np.isfinite(candidate_residual)
            candidate_cost = float(
                np.sum((candidate_residual[cand_valid] ** 2) * w[cand_valid])
            )
            if np.isfinite(candidate_cost) and candidate_cost <= cost:
                eo = candidate
                residual = candidate_residual
                lam = max(lam * 0.3, 1e-12)
                break
            lam *= 10
        else:
            break

        if abs(previous_cost - cost) < tolerance * max(cost, 1.0):
            converged = True
            break
        previous_cost = cost

    residuals = residual.reshape(n, 2)
    finite = np.isfinite(residuals).all(axis=1)
    dof = max(2 * int(finite.sum()) - 6, 1)
    weighted_sq = float(np.sum((residuals[finite] ** 2).sum(axis=1) * weights[finite]))
    rms = float(np.sqrt(np.mean((residuals[finite] ** 2).sum(axis=1)))) if finite.any() else float("nan")

    return ResectionResult(
        eo=eo,
        converged=converged,
        iterations=iteration,
        rms_mm=rms,
        residuals_mm=residuals,
        sigma0=float(np.sqrt(weighted_sq / dof)),
        message="Converged" if converged else "Stopped before convergence tolerance",
    )
