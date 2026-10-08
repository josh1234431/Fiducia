"""Bundle block adjustment.

Simultaneously solves the exterior orientation of every photo in the block and
the ground coordinates of every tie point, from image observations of ground
control points, check points and tie points -- optionally with GNSS/IMU
observations of the camera positions and attitudes, and with self-calibration
of the camera.

Why this exists in the shape it does
------------------------------------
The normal equation matrix of a block adjustment is enormous but extremely
sparse: an observation of point j on photo i touches only photo i's six
exterior orientation unknowns and point j's three coordinate unknowns. Solving
it densely is what makes a legacy adjustment crawl on a block of any size. Here
the sparsity pattern is handed to ``scipy.optimize.least_squares`` explicitly,
so the trust-region solver factors a sparse Jacobian and a block of a few dozen
photos with thousands of tie points stays interactive.

Every kind of measurement is an observation with its own a-priori standard
deviation: image coordinates of control points and of tie points (separately,
since one is a careful click and the other an automatic match), control
coordinates (horizontal and vertical separately, since a height read off a DEM
is rarely as good as a surveyed position), and camera positions and attitudes
from GNSS/IMU. Nothing is a hard constraint, so a single mis-identified point
degrades the solution gracefully and shows up in the statistics instead of
warping the block.

Statistics follow the usual geodetic conventions. The variance factor
(sigma-0) is the weighted residual sum of squares over the redundancy, so 1.0
means the a-priori precisions were right. Precisions come from the inverse
normal matrix scaled by sigma-0 squared. Blunders are found by Baarda's data
snooping: each residual divided by its own standard deviation, which depends
on how well the rest of the block checks that observation (its redundancy
number). An observation nobody else checks has redundancy near zero and can
hide any error -- which is reported too, as low reliability.

Coordinates are reduced to a local origin before solving. Projected
coordinates are in the millions of metres, and differencing them in single
steps loses the precision the adjustment is meant to deliver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.sparse import lil_matrix

from .collinearity import project_points, rotation_matrix
from .resection import initial_exterior_orientation, resect

__all__ = ["Observation", "BundleInput", "BundleResult", "adjust_block",
           "SELF_CALIBRATION_SETS", "LENS_PRIORS", "AP_NAMES", "apply_additional_parameters"]

# Additional parameters for self-calibration, in the order they are carried.
# Δx = dx0 + x df/f + x (k1 ρ² + k2 ρ⁴ + k3 ρ⁶) + R [p1 (ρ² + 2ξ²) + 2 p2 ξη] + b1 x + b2 y
# Δy = dy0 + y df/f + y (k1 ρ² + k2 ρ⁴ + k3 ρ⁶) + R [p2 (ρ² + 2η²) + 2 p1 ξη]
# with (ξ, η) = (x, y) / R and ρ² = ξ² + η²: the Brown model with radial
# distance normalised by R, the half-diagonal of the measured format, so every
# coefficient is a plain number of similar size. b1, b2 are affinity and
# non-orthogonality.
AP_NAMES = ("df", "dx0", "dy0", "k1", "k2", "k3", "p1", "p2", "b1", "b2")
SELF_CALIBRATION_SETS = {
    "none": (),
    "interior": ("df", "dx0", "dy0"),
    "radial": ("df", "dx0", "dy0", "k1", "k2"),
    "brown": ("df", "dx0", "dy0", "k1", "k2", "k3", "p1", "p2"),
    # Brown with the focal length held: for a near-nadir block without control,
    # where focal length and the depth of the whole scene trade one for the other.
    "lens": ("dx0", "dy0", "k1", "k2", "k3", "p1", "p2"),
    "full": AP_NAMES,
}
# Loose a-priori standard deviations: they only keep a parameter the block
# cannot determine from wandering, and are far wider than any real effect.
AP_PRIOR_SIGMA = {"df": 0.1, "dx0": 0.1, "dy0": 0.1,   # mm
                  "k1": 2e-3, "k2": 2e-3, "k3": 2e-3, "p1": 2e-3, "p2": 2e-3,
                  "b1": 2e-3, "b2": 2e-3}
# A consumer camera (drone, phone, compact): a wide-angle lens whose
# distortion reaches the edge of the frame in whole percent, and a principal
# point off-centre by tens of pixels. The focal length stays as tightly held
# as a metric camera's: a flat nadir block cannot tell it from flying height,
# and with only GNSS heights to separate them a looser prior lets the two
# wander together instead of converging.
AP_PRIOR_SIGMA_CONSUMER = {"df": 0.1, "dx0": 0.1, "dy0": 0.1,
                           "k1": 0.1, "k2": 0.1, "k3": 0.1, "p1": 0.01, "p2": 0.01,
                           "b1": 0.01, "b2": 0.01}
LENS_PRIORS = {"metric": AP_PRIOR_SIGMA, "consumer": AP_PRIOR_SIGMA_CONSUMER}

# Critical value of Baarda's w-test at a significance of 0.1%.
W_CRITICAL = 3.29


def apply_additional_parameters(xy: np.ndarray, values: dict, focal_mm: float,
                                radius_mm: float) -> np.ndarray:
    """The self-calibration correction (Δx, Δy) at focal-plane positions."""
    xy = np.atleast_2d(np.asarray(xy, dtype=float))
    x, y = xy[:, 0], xy[:, 1]
    R = max(float(radius_mm), 1e-6)
    xi, eta = x / R, y / R
    rho2 = xi * xi + eta * eta
    get = lambda name: float(values.get(name, 0.0) or 0.0)  # noqa: E731
    radial = get("k1") * rho2 + get("k2") * rho2 ** 2 + get("k3") * rho2 ** 3
    scale = get("df") / focal_mm if focal_mm else 0.0
    dx = (get("dx0") + x * scale + x * radial
          + R * (get("p1") * (rho2 + 2 * xi * xi) + 2 * get("p2") * xi * eta)
          + get("b1") * x + get("b2") * y)
    dy = (get("dy0") + y * scale + y * radial
          + R * (get("p2") * (rho2 + 2 * eta * eta) + 2 * get("p1") * xi * eta))
    return np.column_stack([dx, dy])


@dataclass
class Observation:
    """One measurement of one point on one photo, in focal-plane millimetres."""

    image_id: str
    point_id: str
    x_mm: float
    y_mm: float
    # A multiplier on the observation's weight, for down-weighting a point
    # without deleting it. 1 means "as precise as its class".
    weight: float = 1.0


@dataclass
class BundleInput:
    image_ids: list[str]
    focal_by_image: dict[str, float]
    observations: list[Observation]
    # point_id -> (X, Y, Z) for control and check points
    control: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    # Points excluded from the solution but still measured, for independent
    # accuracy assessment, conventionally "changing a point to a check point".
    check_points: set[str] = field(default_factory=set)
    # Standard deviation of control coordinates in ground units; smaller means
    # the adjustment trusts the surveyed coordinate more. Horizontal, then
    # vertical (None: the same as horizontal).
    control_sigma_m: float = 0.5
    control_sigma_z_m: Optional[float] = None
    # Per-point standard deviations, where the control has its own (a survey
    # file with a precision column): one number for all three axes, or
    # (sx, sy, sz). A point surveyed to 2 cm and one read off a map to 1 m
    # must not be weighted alike.
    control_sigma_by_point: dict = field(default_factory=dict)
    # Image measurement precision, in focal-plane mm: control and check point
    # measurements, then automatic tie points (None: the same).
    image_sigma_mm: float = 0.010
    tie_sigma_mm: Optional[float] = None
    image_scale: float = 0.0
    fixed_eo: dict[str, np.ndarray] = field(default_factory=dict)
    # GNSS/IMU: observed (X0, Y0, Z0, omega, phi, kappa) per photo, any entry
    # None if not observed, with standard deviations (horizontal m, vertical
    # m, attitude radians); a sigma of None leaves that part unobserved.
    eo_observations: dict = field(default_factory=dict)
    eo_sigma_xy_m: Optional[float] = None
    eo_sigma_z_m: Optional[float] = None
    eo_sigma_angle_rad: Optional[float] = None
    # Approximate ground height, for a block placed by GNSS alone (no control).
    # Its tie points start where their rays meet this level: intersecting rays
    # from metre-level positions and rough attitudes over a short base puts
    # many of them metres off, some behind the cameras.
    ground_z: Optional[float] = None
    # One constant offset between the GNSS positions and the block, solved
    # for: a datum or antenna-offset error common to the whole flight.
    gnss_shift: bool = False
    # Self-calibration: a key of SELF_CALIBRATION_SETS.
    self_calibration: str = "none"
    # How loosely the self-calibration parameters are held: a key of LENS_PRIORS.
    lens_priors: str = "metric"
    # Robust estimation. With a quadratic loss a single blundered point drags
    # the whole block towards it and spreads its error over every other point,
    # which is precisely what makes blunder hunting so tedious in legacy
    # software. A soft-L1 loss caps the influence of large residuals, so the
    # block stays where the good observations put it and the bad point is left
    # sitting on an obvious residual.
    robust: bool = True
    robust_scale: float = 3.0        # in multiples of the expected sigma
    blunder_threshold: float = 3.5   # MAD-standardised residual, when snooping is too costly
    # Automatic tie points that fail data snooping are removed and the block
    # solved again, as production triangulation does. Control and check
    # points are only ever flagged: they are survey data, and whether a
    # coordinate or a measurement is wrong is the operator's call.
    auto_reject_ties: bool = True
    # Point ids that are automatic tie points (eligible for rejection).
    automatic_ties: set = field(default_factory=set)


@dataclass
class BundleResult:
    eo: dict[str, np.ndarray]
    object_points: dict[str, np.ndarray]
    residuals_mm: dict[tuple[str, str], tuple[float, float]]
    converged: bool
    iterations: int
    sigma0: float
    rms_image_mm: float
    rms_control_m: float
    rms_check_m: float
    degrees_of_freedom: int
    message: str
    per_image_rms_mm: dict[str, float] = field(default_factory=dict)
    per_point_rms_mm: dict[str, float] = field(default_factory=dict)
    control_residuals_m: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    # Points whose standardised residual marks them as probable blunders,
    # worst first, with a plain-language reason the interface can show.
    suspects: list[dict] = field(default_factory=list)
    # Precision of the solution, from the covariance of the adjustment:
    # standard deviations of X0, Y0, Z0 (m) and omega, phi, kappa (radians)
    # for every adjusted photo, and of X, Y, Z (m) for every point. Scaled by
    # sigma0, so they reflect the fit actually achieved, not only the
    # a-priori weights. None where the geometry does not determine them.
    eo_sigma: dict[str, list[float]] = field(default_factory=dict)
    point_sigma: dict[str, list[float]] = field(default_factory=dict)
    # Full 6 x 6 covariance of each photo's orientation. Position and tilt are
    # strongly correlated -- an error in one is largely offset by the other --
    # so anything propagated from the orientation needs the whole block, not
    # the standard deviations alone.
    eo_covariance: dict[str, list[list[float]]] = field(default_factory=dict)
    # Root mean square of control and check residuals per axis (X, Y, Z).
    rms_control_xyz_m: tuple = (float("nan"),) * 3
    rms_check_xyz_m: tuple = (float("nan"),) * 3
    # Self-calibration: name -> (value, standard deviation), and the format
    # radius the coefficients are normalised by.
    additional_parameters: dict = field(default_factory=dict)
    format_radius_mm: float = 0.0
    gnss_shift_m: Optional[list] = None
    # GNSS/IMU residuals per photo: observed minus adjusted.
    eo_observation_residuals: dict = field(default_factory=dict)
    # Data snooping: per point the largest |w| and the smallest redundancy
    # number of its observations; and the block's mean redundancy.
    snooping: dict = field(default_factory=dict)
    mean_redundancy: Optional[float] = None
    redundancy_total: Optional[float] = None
    # Points left out because a single ray cannot place them.
    single_ray_points: list = field(default_factory=list)
    # Of those, check points checked horizontally at their surveyed height.
    horizontal_checks: list = field(default_factory=list)
    # Automatic tie points removed by data snooping, with their test value.
    rejected_ties: dict = field(default_factory=dict)


def _seed_object_point(
    observations: list[Observation],
    eo: dict[str, np.ndarray],
    focal_by_image: dict[str, float],
    fallback_z: float,
    on_plane: bool = False,
) -> np.ndarray:
    """Triangulate a tie point from two or more rays by linear intersection.

    With only one ray available, the point is dropped onto the fallback
    elevation along that ray, which is enough to start the iteration.
    """
    rows, rhs, origins, directions = [], [], [], []

    for obs in observations:
        params = eo.get(obs.image_id)
        if params is None:
            continue
        rot = rotation_matrix(params[3], params[4], params[5])
        direction = rot.T @ np.array([obs.x_mm, obs.y_mm, -focal_by_image[obs.image_id]], dtype=float)
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            continue
        direction = direction / norm
        origin = np.asarray(params[:3], dtype=float)
        origins.append(origin)
        directions.append(direction)
        projector = np.eye(3) - np.outer(direction, direction)
        rows.append(projector)
        rhs.append(projector @ origin)

    if not rows:
        return np.array([0.0, 0.0, fallback_z])

    if on_plane:
        drops = [o + (fallback_z - o[2]) / d[2] * d for o, d in zip(origins, directions)
                 if d[2] < -1e-3]
        if drops:
            return np.mean(drops, axis=0)

    if len(rows) == 1:
        origin, direction = origins[0], directions[0]
        if abs(direction[2]) < 1e-9:
            return np.array([origin[0], origin[1], fallback_z])
        return origin + (fallback_z - origin[2]) / direction[2] * direction

    try:
        return np.linalg.solve(np.sum(rows, axis=0), np.sum(rhs, axis=0))
    except np.linalg.LinAlgError:
        return np.array([origins[0][0], origins[0][1], fallback_z])


def _control_sigmas(data: BundleInput, pid: str) -> np.ndarray:
    default_z = data.control_sigma_z_m if data.control_sigma_z_m else data.control_sigma_m
    value = data.control_sigma_by_point.get(pid)
    if isinstance(value, (list, tuple)) and len(value) == 3:
        sigmas = [float(v) if v else d for v, d in
                  zip(value, (data.control_sigma_m, data.control_sigma_m, default_z))]
    elif value:
        sigmas = [float(value)] * 3
    else:
        sigmas = [data.control_sigma_m, data.control_sigma_m, default_z]
    return np.maximum(np.asarray(sigmas, dtype=float), 1e-6)


def adjust_block(data: BundleInput, max_iterations: int = 80) -> BundleResult:
    """Run the block adjustment and return the solved block plus diagnostics."""
    image_ids = list(data.image_ids)
    image_index = {img: i for i, img in enumerate(image_ids)}

    by_point: dict[str, list[Observation]] = {}
    for obs in data.observations:
        if obs.image_id in image_index:
            by_point.setdefault(obs.point_id, []).append(obs)

    # A point seen on one photo only, and not held by control coordinates,
    # has three unknowns and two equations: anywhere along its ray fits. It
    # contributes nothing and makes the normal matrix singular, so it is left
    # out (and reported) -- a check point among them cannot be checked.
    active_control = {pid for pid in data.control if pid not in data.check_points}
    single_ray = sorted(pid for pid, obs_list in by_point.items()
                        if pid not in active_control and len({o.image_id for o in obs_list}) < 2)
    for pid in single_ray:
        del by_point[pid]
    dropped = set(single_ray)
    original_observations = list(data.observations)
    single_ray_obs = [o for o in data.observations if o.point_id in dropped]
    data = BundleInput(**{**data.__dict__, "observations": [
        o for o in data.observations if o.point_id not in dropped]})

    if not by_point:
        raise ValueError("No image observations to adjust")

    # GNSS/IMU observations, only for photos in the block and only the parts
    # that are both observed and given a precision.
    eo_sigma_parts = [data.eo_sigma_xy_m, data.eo_sigma_xy_m, data.eo_sigma_z_m,
                      data.eo_sigma_angle_rad, data.eo_sigma_angle_rad, data.eo_sigma_angle_rad]
    eo_obs: dict[str, list[tuple[int, float, float]]] = {}
    for img, values in (data.eo_observations or {}).items():
        if img not in image_index or img in data.fixed_eo or values is None:
            continue
        parts = [(k, float(values[k]), float(eo_sigma_parts[k]))
                 for k in range(6)
                 if k < len(values) and values[k] is not None and eo_sigma_parts[k]]
        if parts:
            eo_obs[img] = parts
    observed_positions = [img for img, parts in eo_obs.items()
                          if sum(1 for k, _, _ in parts if k < 3) == 3]

    solved_control = {
        pid: xyz for pid, xyz in data.control.items() if pid not in data.check_points
    }
    if len(solved_control) < 3 and len(observed_positions) < 3:
        raise ValueError(
            f"Block adjustment needs at least 3 active ground control points, "
            f"got {len(solved_control)}, or observed camera positions on at least "
            "3 photos (GNSS-assisted)."
        )

    # Each photo carries six unknowns, and every image measurement supplies two
    # equations, so fewer than three measurements can never determine an
    # orientation -- unless the orientation itself was observed. Seeding such
    # a photo from a neighbour and adjusting anyway produces a solution that
    # converges on nonsense and reports success.
    counts: dict[str, int] = {}
    for obs in data.observations:
        if obs.image_id in image_index:
            counts[obs.image_id] = counts.get(obs.image_id, 0) + 1

    starved = sorted(
        (img for img in image_ids
         if img not in data.fixed_eo and img not in eo_obs and counts.get(img, 0) < 3),
        key=lambda img: counts.get(img, 0),
    )
    if starved:
        detail = ", ".join(f"{img} ({counts.get(img, 0)})" for img in starved)
        raise ValueError(
            "These images have too few measurements to determine an orientation "
            f"(3 is the minimum, 4 or more is advisable): {detail}. "
            "Add control or tie points on them, or remove them from the block."
        )

    # -- local origin ------------------------------------------------------
    anchors = [np.asarray(v, float) for v in data.control.values()]
    anchors += [np.array([v[k] for k in range(3)], float)
                for v in (data.eo_observations or {}).values()
                if v is not None and all(v[k] is not None for k in range(3))]
    origin = np.round(np.mean(anchors, axis=0)) if anchors else np.zeros(3)

    def local(xyz):
        return np.asarray(xyz, dtype=float) - origin

    control = {pid: local(xyz) for pid, xyz in data.control.items()}
    solved_local = {pid: control[pid] for pid in solved_control}
    fixed_eo = {img: np.concatenate([local(v[:3]), np.asarray(v[3:6], float)])
                for img, v in data.fixed_eo.items()}
    eo_obs_local = {
        img: [(k, value - origin[k] if k < 3 else value, sigma) for k, value, sigma in parts]
        for img, parts in eo_obs.items()
    }

    control_z = [xyz[2] for xyz in control.values()]
    fallback_z = float(np.mean(control_z)) if control_z else -1000.0
    seed_on_plane = data.ground_z is not None and not control
    if seed_on_plane:
        fallback_z = float(data.ground_z) - float(origin[2])

    # -- initialise each photo -------------------------------------------
    eo: dict[str, np.ndarray] = {}
    unresolved: list[str] = []
    for img in image_ids:
        if img in fixed_eo:
            eo[img] = fixed_eo[img].copy()
            continue
        film, ground = [], []
        for pid, obs_list in by_point.items():
            if pid not in control:
                continue
            for obs in obs_list:
                if obs.image_id == img:
                    film.append((obs.x_mm, obs.y_mm))
                    ground.append(control[pid])
        if len(film) >= 3:
            try:
                eo[img] = resect(np.array(film), np.array(ground), data.focal_by_image[img],
                                 image_scale=data.image_scale).eo
                continue
            except Exception:
                pass
        observed = data.eo_observations.get(img) if data.eo_observations else None
        if observed is not None and all(v is not None for v in observed[:6]):
            eo[img] = np.concatenate([local(observed[:3]), np.asarray(observed[3:6], float)])
            continue
        unresolved.append(img)

    # Photos without enough direct control are placed from their tie points:
    # points they share with photos already placed are triangulated from
    # those, and the photo is resected from the result. Repeated, this walks
    # along strips and across them. (Copying a neighbour's orientation
    # instead leaves the photo a whole base length out, and the adjustment
    # can settle in a false minimum from there.)
    progressed = True
    while unresolved and progressed:
        progressed = False
        for img in list(unresolved):
            film, ground = [], []
            for pid, obs_list in by_point.items():
                mine = [o for o in obs_list if o.image_id == img]
                if not mine:
                    continue
                if pid in control:
                    xyz = control[pid]
                else:
                    placed = [o for o in obs_list if o.image_id in eo]
                    if not placed:
                        continue
                    xyz = _seed_object_point(placed, eo, data.focal_by_image, fallback_z)
                film.append((mine[0].x_mm, mine[0].y_mm))
                ground.append(xyz)
            if len(film) < 4:
                continue
            try:
                result = resect(np.array(film), np.array(ground), data.focal_by_image[img],
                                image_scale=data.image_scale)
            except Exception:
                continue
            if np.isfinite(result.eo).all():
                eo[img] = result.eo
                unresolved.remove(img)
                progressed = True

    for img in unresolved:
        anchor = None
        shared_best = 0
        for other, params in eo.items():
            shared = sum(
                1
                for obs_list in by_point.values()
                if any(o.image_id == img for o in obs_list)
                and any(o.image_id == other for o in obs_list)
            )
            if shared > shared_best:
                shared_best, anchor = shared, params
        eo[img] = (
            anchor.copy()
            if anchor is not None
            else np.array([0.0, 0.0, fallback_z + 1000.0, 0.0, 0.0, 0.0])
        )

    # -- initialise object points -----------------------------------------
    point_ids = sorted(by_point.keys())
    objects: dict[str, np.ndarray] = {}
    for pid in point_ids:
        if pid in control:
            objects[pid] = control[pid].copy()
        else:
            objects[pid] = _seed_object_point(by_point[pid], eo, data.focal_by_image, fallback_z,
                                              on_plane=seed_on_plane)

    point_index = {pid: i for i, pid in enumerate(point_ids)}
    n_points = len(point_ids)

    free_images = [img for img in image_ids if img not in fixed_eo]
    free_image_index = {img: i for i, img in enumerate(free_images)}
    n_free = len(free_images)

    observations = [o for o in data.observations if o.image_id in image_index]

    # Self-calibration: which parameters, and the radius they are normalised by.
    ap_names = list(SELF_CALIBRATION_SETS.get(data.self_calibration or "none", ()))
    prior_sigma = LENS_PRIORS.get(data.lens_priors or "metric", AP_PRIOR_SIGMA)
    n_ap = len(ap_names)
    format_radius = float(max((np.hypot(o.x_mm, o.y_mm) for o in observations), default=1.0))
    focal_ref = float(np.median(list(data.focal_by_image.values()))) if data.focal_by_image else 1.0
    n_shift = 3 if (data.gnss_shift and eo_obs) else 0

    # -- weights -----------------------------------------------------------
    tie_sigma = data.tie_sigma_mm or data.image_sigma_mm
    obs_sigma = np.array([
        (data.image_sigma_mm if o.point_id in data.control else tie_sigma) for o in observations
    ], dtype=float)
    obs_weight = np.array([o.weight for o in observations], dtype=float) / np.maximum(obs_sigma, 1e-9)

    control_ids = [pid for pid in point_ids if pid in solved_local]
    control_targets = np.array([solved_local[pid] for pid in control_ids], dtype=float).reshape(-1, 3)
    control_weights = np.array([1.0 / _control_sigmas(data, pid) for pid in control_ids],
                               dtype=float).reshape(-1, 3)
    control_slots = np.array([point_index[pid] for pid in control_ids], dtype=int)

    eo_rows = [(img, k, value, sigma) for img, parts in eo_obs_local.items()
               for k, value, sigma in parts]

    # -- parameter layout --------------------------------------------------
    obj_base = 6 * n_free
    ap_base = obj_base + 3 * n_points
    shift_base = ap_base + n_ap
    n_param = shift_base + n_shift

    def unpack_eo(vec: np.ndarray) -> dict[str, np.ndarray]:
        out = {}
        for img in image_ids:
            if img in fixed_eo:
                out[img] = fixed_eo[img]
            else:
                i = free_image_index[img]
                out[img] = vec[6 * i: 6 * i + 6]
        return out

    def ap_values(vec: np.ndarray) -> dict:
        return {name: float(vec[ap_base + k]) for k, name in enumerate(ap_names)}

    grouped: dict[str, list[int]] = {}
    for idx, obs in enumerate(observations):
        grouped.setdefault(obs.image_id, []).append(idx)
    group_cache = {
        img: (
            np.array(indices),
            np.array([(observations[i].x_mm, observations[i].y_mm) for i in indices]),
            obs_weight[indices],
            [point_index[observations[i].point_id] for i in indices],
        )
        for img, indices in grouped.items()
    }

    n_image_rows = 2 * len(observations)
    n_control_rows = 3 * len(control_ids)
    n_eo_rows = len(eo_rows)
    n_resid = n_image_rows + n_control_rows + n_eo_rows + n_ap

    def predict(vec: np.ndarray, img: str, slots, eo_map, aps) -> np.ndarray:
        obj_flat = vec[obj_base:ap_base].reshape(n_points, 3)
        predicted = project_points(obj_flat[slots], eo_map[img], data.focal_by_image[img])
        if n_ap:
            predicted = predicted + apply_additional_parameters(
                predicted, aps, data.focal_by_image[img], format_radius)
        return predicted

    def residuals(vec: np.ndarray) -> np.ndarray:
        eo_map = unpack_eo(vec)
        aps = ap_values(vec) if n_ap else {}
        obj_flat = vec[obj_base:ap_base].reshape(n_points, 3)
        out = np.empty(n_resid)

        for img, (indices, measured, weights, slots) in group_cache.items():
            diff = (measured - predict(vec, img, slots, eo_map, aps)) * weights[:, None]
            diff = np.nan_to_num(diff, nan=1e6, posinf=1e6, neginf=-1e6)
            out[2 * indices] = diff[:, 0]
            out[2 * indices + 1] = diff[:, 1]

        cursor = n_image_rows
        if control_ids:
            block = (control_targets - obj_flat[control_slots]) * control_weights
            out[cursor: cursor + n_control_rows] = block.ravel()
        cursor += n_control_rows

        shift = vec[shift_base: shift_base + 3] if n_shift else np.zeros(3)
        for row, (img, k, value, sigma) in enumerate(eo_rows):
            adjusted = eo_map[img][k] + (shift[k] if k < 3 else 0.0)
            difference = value - adjusted
            if k == 5:   # kappa wraps
                difference = float(np.arctan2(np.sin(difference), np.cos(difference)))
            out[cursor + row] = difference / sigma
        cursor += n_eo_rows

        for k, name in enumerate(ap_names):
            out[cursor + k] = -vec[ap_base + k] / prior_sigma[name]
        return out

    # -- analytic Jacobian --------------------------------------------------
    # The collinearity derivatives, chained through the self-calibration,
    # photo by photo in whole arrays. By finite differences the same matrix
    # costs a full pass over the block for every group of unknowns that
    # share no observation, which grows with the block.
    def analytic_jacobian(vec: np.ndarray):
        from scipy.sparse import coo_matrix

        from .collinearity import collinearity_jacobian

        eo_map = unpack_eo(vec)
        aps = ap_values(vec) if n_ap else {}
        obj_flat = vec[obj_base:ap_base].reshape(n_points, 3)
        rows, cols, vals = [], [], []
        for img, (indices, _, weights, slots) in group_cache.items():
            focal = data.focal_by_image[img]
            ground = obj_flat[slots]
            eo = eo_map[img]
            d_eo, d_ground = collinearity_jacobian(ground, eo, focal)
            if n_ap:
                film = project_points(ground, eo, focal)
                # d(delta)/d(film): the lens correction moves with the point.
                h = 1e-6
                grad = np.empty((len(indices), 2, 2))
                for axis in (0, 1):
                    step = np.zeros(2)
                    step[axis] = h
                    grad[:, :, axis] = (apply_additional_parameters(film + step, aps, focal, format_radius)
                                        - apply_additional_parameters(film - step, aps, focal, format_radius)) / (2 * h)
                chain = np.eye(2)[None] + grad
                d_eo = np.einsum("nij,njk->nik", chain, d_eo)
                d_ground = np.einsum("nij,njk->nik", chain, d_ground)
                basis = [apply_additional_parameters(film, {name: 1.0}, focal, format_radius)
                         for name in ap_names]
            scale = -weights[:, None]
            d_eo = np.nan_to_num(d_eo * scale[:, :, None])
            d_ground = np.nan_to_num(d_ground * scale[:, :, None])
            slot_cols = obj_base + 3 * np.asarray(slots)
            for half in (0, 1):
                r = 2 * indices + half
                if img in free_image_index:
                    base = 6 * free_image_index[img]
                    for k in range(6):
                        rows.append(r)
                        cols.append(np.full(r.size, base + k))
                        vals.append(d_eo[:, half, k])
                for k in range(3):
                    rows.append(r)
                    cols.append(slot_cols + k)
                    vals.append(d_ground[:, half, k])
                for k in range(n_ap):
                    rows.append(r)
                    cols.append(np.full(r.size, ap_base + k))
                    vals.append(np.nan_to_num(-weights * basis[k][:, half]))
        for j in range(len(control_ids)):
            for k in range(3):
                rows.append(np.array([n_image_rows + 3 * j + k]))
                cols.append(np.array([obj_base + 3 * control_slots[j] + k]))
                vals.append(np.array([-control_weights[j, k]]))
        for row, (img, k, _, sigma) in enumerate(eo_rows):
            r = n_image_rows + n_control_rows + row
            rows.append(np.array([r]))
            cols.append(np.array([6 * free_image_index[img] + k]))
            vals.append(np.array([-1.0 / sigma]))
            if n_shift and k < 3:
                rows.append(np.array([r]))
                cols.append(np.array([shift_base + k]))
                vals.append(np.array([-1.0 / sigma]))
        for k, name in enumerate(ap_names):
            rows.append(np.array([n_image_rows + n_control_rows + n_eo_rows + k]))
            cols.append(np.array([ap_base + k]))
            vals.append(np.array([-1.0 / prior_sigma[name]]))
        return coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n_resid, n_param)).tocsc()

    # -- sparsity pattern --------------------------------------------------
    # Built in one pass from coordinate lists: element by element it takes
    # longer than the adjustment itself on a block of any size.
    from scipy.sparse import coo_matrix

    rows_list, cols_list = [], []
    obs_rows = np.arange(len(observations))
    obs_point_col = np.array([obj_base + 3 * point_index[o.point_id] for o in observations],
                             dtype=np.int64)
    obs_image_col = np.array([6 * free_image_index[o.image_id] if o.image_id in free_image_index
                              else -1 for o in observations], dtype=np.int64)
    for half in (0, 1):
        r = 2 * obs_rows + half
        free = obs_image_col >= 0
        for k in range(6):
            rows_list.append(r[free])
            cols_list.append(obs_image_col[free] + k)
        for k in range(3):
            rows_list.append(r)
            cols_list.append(obs_point_col + k)
        for k in range(n_ap):
            rows_list.append(r)
            cols_list.append(np.full(r.size, ap_base + k))
    for j, pid in enumerate(control_ids):
        pcol = obj_base + 3 * point_index[pid]
        rows_list.append(n_image_rows + 3 * j + np.arange(3))
        cols_list.append(pcol + np.arange(3))
    for row, (img, k, _, _) in enumerate(eo_rows):
        r = n_image_rows + n_control_rows + row
        rows_list.append(np.array([r]))
        cols_list.append(np.array([6 * free_image_index[img] + k]))
        if n_shift and k < 3:
            rows_list.append(np.array([r]))
            cols_list.append(np.array([shift_base + k]))
    for k in range(n_ap):
        rows_list.append(np.array([n_image_rows + n_control_rows + n_eo_rows + k]))
        cols_list.append(np.array([ap_base + k]))
    all_rows = np.concatenate(rows_list) if rows_list else np.zeros(0, np.int64)
    all_cols = np.concatenate(cols_list) if cols_list else np.zeros(0, np.int64)
    sparsity = coo_matrix((np.ones(all_rows.size, dtype=np.int8), (all_rows, all_cols)),
                          shape=(n_resid, n_param)).tocsr()
    sparsity.data[:] = 1

    # The points' unknowns, eliminated when solving (see schur.py).
    point_block = (obj_base, ap_base) if n_points else None

    x0 = np.zeros(n_param)
    for img in free_images:
        x0[6 * free_image_index[img]: 6 * free_image_index[img] + 6] = eo[img]
    for pid in point_ids:
        x0[obj_base + 3 * point_index[pid]: obj_base + 3 * point_index[pid] + 3] = objects[pid]

    # Residuals are already divided by their sigma, so the robust scale is in
    # units of "expected standard deviations": soft-L1 starts damping
    # anything beyond a few of them.
    solution = _levenberg_marquardt(
        residuals, x0, sparsity,
        loss="soft_l1" if data.robust else "linear",
        scale=data.robust_scale if data.robust else 1.0,
        max_iterations=max_iterations, point_block=point_block, jac_fun=analytic_jacobian,
    )
    iterations = int(solution.nfev)

    if data.robust and solution.success:
        # Soft-L1 caps a blunder's pull but never removes it: a control height
        # wrong by metres, weighted at centimetres, still drags the block. A
        # redescending loss (Cauchy) lets a gross error lose its influence
        # altogether, but is not convex, so it starts from the soft-L1
        # solution rather than from the approximations.
        second = _levenberg_marquardt(residuals, solution.x, sparsity, loss="cauchy",
                                      scale=data.robust_scale, max_iterations=max_iterations,
                                      point_block=point_block, jac_fun=analytic_jacobian)
        iterations += int(second.nfev)
        if second.success and np.isfinite(second.x).all():
            solution = second

    vec = solution.x
    eo_local = unpack_eo(vec)
    obj_local = {pid: vec[obj_base + 3 * point_index[pid]: obj_base + 3 * point_index[pid] + 3]
                 for pid in point_ids}
    aps = ap_values(vec) if n_ap else {}

    # -- statistics on the plain least-squares problem ----------------------
    #
    # The robust loss decides where the solution lands; the statistics are
    # those of ordinary least squares at that solution, the convention every
    # adjustment report follows. (The robust cost underestimates sigma-0
    # exactly when there is a blunder, which is when it matters.)
    standardised = residuals(vec)
    dof = max(n_resid - n_param, 1)
    sigma0 = float(np.sqrt(float(standardised @ standardised) / dof))

    jacobian = None
    try:
        from scipy.optimize._numdiff import approx_derivative

        jacobian = analytic_jacobian(vec)
    except Exception:  # noqa: BLE001
        jacobian = solution.jac

    # Redundancy numbers need the whole normal matrix factorised, which only a
    # small block affords; precision comes from the reduced system always.
    normal_lu = None
    if n_param * n_resid <= 40_000_000:
        normal_lu, _ = _factorise(jacobian)
    eo_sigma, point_sigma, eo_covariance, ap_sigma = _precision(
        jacobian, sigma0, free_images, point_ids, point_index, obj_base, ap_base, ap_names)

    redundancy = _redundancy(jacobian, normal_lu, n_param, n_resid)

    eo_final = {img: np.concatenate([v[:3] + origin, v[3:]]) for img, v in eo_local.items()}
    eo_final = {img: np.asarray(v, dtype=float) for img, v in eo_final.items()}
    obj_final = {pid: np.asarray(v + origin, dtype=float) for pid, v in obj_local.items()}

    # -- residuals for the report -------------------------------------------
    residual_map: dict[tuple[str, str], tuple[float, float]] = {}
    per_image: dict[str, list[float]] = {}
    per_point: dict[str, list[float]] = {}

    for img, (indices, measured, _, slots) in group_cache.items():
        predicted = predict(vec, img, slots, eo_local, aps)
        for k, i in enumerate(indices):
            obs = observations[i]
            dx = float(obs.x_mm - predicted[k, 0])
            dy = float(obs.y_mm - predicted[k, 1])
            if not np.isfinite(dx) or not np.isfinite(dy):
                dx = dy = float("nan")
            residual_map[(obs.image_id, obs.point_id)] = (dx, dy)
            mag = float(np.hypot(dx, dy))
            if np.isfinite(mag):
                per_image.setdefault(img, []).append(mag)
                per_point.setdefault(obs.point_id, []).append(mag)

    all_mag = [m for values in per_image.values() for m in values]
    rms_image = float(np.sqrt(np.mean(np.square(all_mag)))) if all_mag else float("nan")

    control_residuals: dict[str, tuple[float, float, float]] = {}
    control_xyz, check_xyz = [], []
    for pid, xyz in data.control.items():
        if pid not in obj_final:
            continue
        diff = np.asarray(xyz, dtype=float) - obj_final[pid]
        control_residuals[pid] = (float(diff[0]), float(diff[1]), float(diff[2]))
        (check_xyz if pid in data.check_points else control_xyz).append(diff)

    # A check point measured on one photo cannot be triangulated, but it can
    # still check the block horizontally: its ray, from the adjusted
    # orientation, meets its own surveyed height at the position the block
    # says it is. Its height is not checked (dZ is NaN).
    horizontal_checks = []
    for obs in single_ray_obs:
        pid = obs.point_id
        if pid not in data.check_points or pid not in data.control or pid in control_residuals:
            continue
        params = eo_final.get(obs.image_id)
        if params is None:
            continue
        film = np.array([[obs.x_mm, obs.y_mm]])
        if n_ap:
            film = film - apply_additional_parameters(
                film, aps, data.focal_by_image[obs.image_id], format_radius)
        rot = rotation_matrix(params[3], params[4], params[5])
        direction = rot.T @ np.array([film[0, 0], film[0, 1], -data.focal_by_image[obs.image_id]])
        surveyed = np.asarray(data.control[pid], dtype=float)
        if abs(direction[2]) < 1e-12:
            continue
        t = (surveyed[2] - params[2]) / direction[2]
        ground = params[:3] + t * direction
        diff = (float(surveyed[0] - ground[0]), float(surveyed[1] - ground[1]), float("nan"))
        control_residuals[pid] = diff
        check_xyz.append(np.array(diff))
        horizontal_checks.append(pid)

    def rms3(rows):
        if not rows:
            return (float("nan"),) * 3
        a = np.asarray(rows, dtype=float)
        with np.errstate(invalid="ignore"):
            return tuple(float(np.sqrt(np.nanmean(a[:, k] ** 2))) if np.isfinite(a[:, k]).any()
                         else float("nan") for k in range(3))

    def rms_norm(rows):
        if not rows:
            return 0.0
        return float(np.sqrt(np.mean(np.nansum(np.square(rows), axis=1))))

    eo_obs_residuals: dict[str, list] = {}
    shift = vec[shift_base: shift_base + 3] if n_shift else np.zeros(3)
    for img, parts in eo_obs_local.items():
        entry = [None] * 6
        for k, value, _ in parts:
            difference = value - (eo_local[img][k] + (shift[k] if k < 3 else 0.0))
            if k == 5:
                difference = float(np.arctan2(np.sin(difference), np.cos(difference)))
            entry[k] = float(difference)
        eo_obs_residuals[img] = entry

    # A solver can satisfy its own stopping tolerance while sitting in a
    # physically meaningless minimum. Residuals an order of magnitude beyond
    # the expected measurement precision mean the geometry is wrong, not that
    # the measurements were sloppy -- so the result is reported as not
    # converged regardless of what the optimiser thought.
    converged = bool(solution.success)
    message = str(solution.message)
    tolerance = max(data.image_sigma_mm, tie_sigma) * 25.0
    if np.isfinite(rms_image) and rms_image > tolerance:
        converged = False
        message = (
            f"Image residual of {rms_image * 1000:.0f} um is far beyond the expected "
            f"{data.image_sigma_mm * 1000:.0f} um. The block has settled on an "
            "implausible geometry -- check for mis-identified control, an incorrect "
            "focal length, or control points entered in the wrong coordinate system."
        )

    # -- blunder detection -------------------------------------------------
    suspects: list[dict] = []
    snooping: dict = {}
    mean_redundancy = None
    if redundancy is not None:
        mean_redundancy = float(np.mean(redundancy))
        # Pope's tau: each residual over its own a-posteriori standard
        # deviation, sigma0 sqrt(r). With the a-priori sigma instead (Baarda's
        # w) every point fails when the a-priori precisions were optimistic;
        # this asks only which observations disagree with the rest.
        w = standardised / (max(sigma0, 1e-9) * np.sqrt(np.maximum(redundancy, 1e-9)))
        for idx, obs in enumerate(observations):
            rows = [2 * idx, 2 * idx + 1]
            worst = float(np.max(np.abs(w[rows])))
            entry = snooping.setdefault(obs.point_id, {"maxW": 0.0, "minRedundancy": 1.0})
            entry["maxW"] = max(entry["maxW"], worst)
            entry["minRedundancy"] = min(entry["minRedundancy"], float(np.min(redundancy[rows])))
        for j, pid in enumerate(control_ids):
            rows = [n_image_rows + 3 * j + k for k in range(3)]
            worst = float(np.max(np.abs(w[rows])))
            entry = snooping.setdefault(pid, {"maxW": 0.0, "minRedundancy": 1.0})
            entry["maxW"] = max(entry["maxW"], worst)
            entry["controlW"] = worst
            entry["minRedundancy"] = min(entry["minRedundancy"], float(np.min(redundancy[rows])))

        for pid, entry in snooping.items():
            if entry["maxW"] >= W_CRITICAL:
                is_control = pid in solved_local
                kind = "control" if is_control else ("check" if pid in data.control else "tie")
                if is_control:
                    value = float(np.linalg.norm(control_residuals.get(pid, (0, 0, 0))))
                    unit, note = "m", (
                        "Fails the data-snooping test (w = {score:.1f}, critical 3.29): its "
                        "coordinate or its measurement disagrees with the rest of the block "
                        "by more than its precision allows. Ground residual {value:.2f} m.")
                else:
                    value = float(np.sqrt(np.mean(np.square(per_point.get(pid, [np.nan])))))
                    unit, note = "mm", (
                        "Fails the data-snooping test (w = {score:.1f}, critical 3.29). "
                        "Most likely a mismatched or mis-measured point. Image residual "
                        "{value:.4f} mm.")
                suspects.append({
                    "pointId": pid, "kind": kind, "residual": value, "unit": unit,
                    "standardised": float(entry["maxW"]),
                    "note": note.format(value=value, score=entry["maxW"]),
                })
        weak = [pid for pid, entry in snooping.items()
                if entry["minRedundancy"] < 0.05 and pid in solved_local]
        for pid in weak:
            snooping[pid]["unchecked"] = True
    else:
        suspects = _mad_suspects(control_residuals, per_point, data, solved_local)

    suspects.sort(key=lambda s: s["standardised"], reverse=True)

    # Remove automatic tie points that fail the test and solve again. All at
    # once is safe for tie points: each is checked by many others, so one
    # blunder does not hide behind another the way control can. At most a
    # tenth of them go, so a wrong a-priori precision cannot strip the block.
    if data.auto_reject_ties and snooping:
        failing = sorted(
            ((entry["maxW"], pid) for pid, entry in snooping.items()
             if pid in data.automatic_ties and entry["maxW"] >= W_CRITICAL),
            reverse=True)
    elif data.auto_reject_ties:
        # Too large to snoop (drone blocks): the robust screen's suspects, but
        # only those also well beyond the tie points' own precision, so the
        # tail of a sharp distribution is not mistaken for blunders.
        failing = sorted(
            ((s["standardised"], s["pointId"]) for s in suspects
             if s["kind"] == "tie" and s["pointId"] in data.automatic_ties
             and s["residual"] > 3.0 * tie_sigma),
            reverse=True)
    else:
        failing = []
    if failing:
        limit = max(1, len(data.automatic_ties) // 10)
        rejected = {pid: float(w) for w, pid in failing[:limit]}
        again = adjust_block(BundleInput(**{
            **data.__dict__,
            "observations": [o for o in original_observations if o.point_id not in rejected],
            "auto_reject_ties": False,
        }), max_iterations)
        again.rejected_ties = rejected
        return again

    return BundleResult(
        eo=eo_final,
        object_points=obj_final,
        residuals_mm=residual_map,
        converged=converged,
        iterations=iterations,
        sigma0=sigma0,
        rms_image_mm=rms_image,
        rms_control_m=rms_norm(control_xyz),
        rms_check_m=rms_norm(check_xyz),
        degrees_of_freedom=dof,
        message=message,
        per_image_rms_mm={
            k: float(np.sqrt(np.mean(np.square(v)))) for k, v in per_image.items()
        },
        per_point_rms_mm={
            k: float(np.sqrt(np.mean(np.square(v)))) for k, v in per_point.items()
        },
        control_residuals_m=control_residuals,
        suspects=suspects,
        eo_sigma=eo_sigma,
        point_sigma=point_sigma,
        eo_covariance=eo_covariance,
        rms_control_xyz_m=rms3(control_xyz),
        rms_check_xyz_m=rms3(check_xyz),
        additional_parameters={name: (aps[name], ap_sigma.get(name)) for name in ap_names},
        format_radius_mm=format_radius if n_ap else 0.0,
        gnss_shift_m=[float(v) for v in shift] if n_shift else None,
        eo_observation_residuals=eo_obs_residuals,
        snooping=snooping,
        mean_redundancy=mean_redundancy,
        redundancy_total=float(np.sum(redundancy)) if redundancy is not None else None,
        single_ray_points=single_ray,
        horizontal_checks=horizontal_checks,
    )


class _Solution:
    """What the solver returns: the shape adjust_block reads."""

    def __init__(self, x, success, message, nfev, jac):
        self.x, self.success, self.message, self.nfev, self.jac = x, success, message, nfev, jac


def _robust_weights(residual: np.ndarray, loss: str, scale: float) -> np.ndarray:
    """IRLS weights of a robust loss, for residuals already in sigma units."""
    if loss == "linear":
        return np.ones_like(residual)
    z = (residual / scale) ** 2
    if loss == "soft_l1":
        return 1.0 / np.sqrt(1.0 + z)
    if loss == "cauchy":
        return 1.0 / (1.0 + z)
    raise ValueError(loss)


def _robust_cost(residual: np.ndarray, loss: str, scale: float) -> float:
    if loss == "linear":
        return 0.5 * float(residual @ residual)
    z = (residual / scale) ** 2
    if loss == "soft_l1":
        rho = 2.0 * (np.sqrt(1.0 + z) - 1.0)
    else:
        rho = np.log1p(z)
    return 0.5 * scale * scale * float(np.sum(rho))


def _levenberg_marquardt(fun, x0: np.ndarray, sparsity, loss: str = "linear",
                         scale: float = 1.0, max_iterations: int = 100,
                         point_block: Optional[tuple] = None, jac_fun=None) -> _Solution:
    """Damped Gauss-Newton on the sparse normal equations.

    The classical bundle solver: each step solves (J'WJ + lambda D) dx = -J'Wr
    by sparse direct factorisation, which copes with the poor conditioning of
    a block -- millimetre image residuals beside metre ground coordinates,
    points seen on two photos beside points seen on ten -- where an iterative
    inner solver crawls. Robust losses enter as iteratively reweighted least
    squares. The Jacobian is by finite differences, grouped by the sparsity
    pattern so it costs a few dozen residual evaluations, not one per unknown.
    """
    from scipy.optimize._numdiff import approx_derivative
    from scipy.sparse import csc_matrix, diags
    from scipy.sparse.linalg import splu

    x = np.asarray(x0, dtype=float).copy()
    residual = fun(x)
    cost = _robust_cost(residual, loss, scale)
    lam = 1e-3
    evaluations = 1
    jac = None
    message = "Stopped at the iteration limit"
    success = False
    # A robust loss on a finite-difference Jacobian keeps finding decreases of
    # a part in a million or less long after the block has settled; five of
    # them in a row is convergence, not progress.
    stalled = 0
    for _ in range(max_iterations):
        if jac_fun is not None:
            jac = jac_fun(x)
        else:
            jac = csc_matrix(approx_derivative(fun, x, method="2-point", sparsity=sparsity,
                                               f0=residual))
        weights = _robust_weights(residual, loss, scale)
        jw = diags(weights) @ jac
        normal = (jac.T @ jw).tocsc()
        gradient = jac.T @ (weights * residual)
        damping = normal.diagonal().copy()
        damping[damping <= 0] = 1.0
        if np.max(np.abs(gradient) / np.sqrt(damping)) < 1e-10 * max(1.0, np.sqrt(2 * cost)):
            success, message = True, "Converged: gradient vanished"
            break
        improved = False
        for _ in range(12):
            try:
                damped = (normal + diags(lam * damping)).tocsc()
                if point_block:
                    from .schur import ReducedSystem

                    step = ReducedSystem(damped, *point_block).solve(-gradient)
                else:
                    step = splu(damped).solve(-gradient)
                if not np.isfinite(step).all():
                    raise RuntimeError("non-finite step")
            except (RuntimeError, np.linalg.LinAlgError, ValueError):
                lam *= 10.0
                continue
            candidate = x + step
            candidate_residual = fun(candidate)
            evaluations += 1
            candidate_cost = _robust_cost(candidate_residual, loss, scale)
            if np.isfinite(candidate_cost) and candidate_cost <= cost:
                improved = True
                relative = (cost - candidate_cost) / max(cost, 1e-300)
                x, residual, cost = candidate, candidate_residual, candidate_cost
                lam = max(lam / 5.0, 1e-12)
                break
            lam *= 10.0
        if not improved:
            success, message = True, "Converged: no further decrease"
            break
        small_step = np.max(np.abs(step) / (np.abs(x) + 1.0)) < 1e-12
        if relative < 1e-12 or small_step:
            success, message = True, "Converged"
            break
        stalled = stalled + 1 if relative < 1e-6 else 0
        if stalled >= 5:
            success, message = True, "Converged: cost no longer changing"
            break
    return _Solution(x, success, message, evaluations, jac)


def _mad_suspects(control_residuals, per_point, data, solved_local) -> list[dict]:
    """Fallback blunder screen for blocks too large for full data snooping.

    Standardises each point's residual against the robust spread of all of
    them (median absolute deviation, which a blunder cannot inflate the way
    it inflates a standard deviation).
    """
    suspects: list[dict] = []

    def flag(values: dict[str, float], kind: str, unit: str, note: str) -> None:
        if len(values) < 4:
            return
        magnitudes = np.array(list(values.values()), dtype=float)
        finite = magnitudes[np.isfinite(magnitudes)]
        if finite.size < 4:
            return
        median = float(np.median(finite))
        spread = max(float(np.median(np.abs(finite - median))) * 1.4826, 1e-9)
        for point_id, magnitude in values.items():
            if not np.isfinite(magnitude):
                continue
            score = (magnitude - median) / spread
            if score >= data.blunder_threshold:
                suspects.append({
                    "pointId": point_id, "kind": kind, "residual": float(magnitude),
                    "unit": unit, "standardised": float(score),
                    "note": note.format(value=magnitude, score=score),
                })

    flag({pid: float(np.linalg.norm(v)) for pid, v in control_residuals.items()
          if pid in solved_local},
         "control", "m",
         "Ground residual of {value:.2f} m is {score:.1f} times the spread of the "
         "other control points. Check the surveyed coordinate and the measured position.")
    flag({k: float(np.sqrt(np.mean(np.square(v)))) for k, v in per_point.items()
          if k not in data.control},
         "tie", "mm",
         "Image residual of {value:.3f} mm is {score:.1f} times the spread of the "
         "other tie points. Most likely a mismatched point.")
    return suspects


def _factorise(jacobian):
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import splu

    try:
        J = csc_matrix(jacobian)
        normal = (J.T @ J).tocsc()
        return splu(normal), normal
    except Exception:  # noqa: BLE001  -- singular or otherwise unfactorisable
        return None, None


def _redundancy(jacobian, lu, n_param: int, n_resid: int, max_cells: int = 40_000_000):
    """Redundancy numbers r_i = 1 - (J (J^T J)^-1 J^T)_ii of every observation.

    Their sum is the degrees of freedom. Skipped (None) on blocks so large
    that the dense intermediate would not fit comfortably in memory.
    """
    if lu is None or n_param * n_resid > max_cells:
        return None
    from scipy.sparse import csr_matrix

    try:
        J = csr_matrix(jacobian)
        solved = lu.solve(J.T.toarray())            # (n_param, n_resid)
        hat = np.asarray(J.multiply(solved.T).sum(axis=1)).ravel()
    except Exception:  # noqa: BLE001
        return None
    return np.clip(1.0 - hat, 0.0, 1.0)


def _precision(jacobian, sigma0, free_images, point_ids, point_index, obj_base, ap_base,
               ap_names, max_points: int = 200_000):
    """Precision of the adjusted parameters.

    Covariance = sigma0^2 (J^T J)^-1, with J the Jacobian of the standardised
    residuals at the solution, taken from the reduced camera system (see
    schur.py): each photo's full 6 x 6 block, the self-calibration's
    variances, and three variances per point, without ever inverting the
    whole normal matrix. On a singular system -- a block whose datum its
    control does not fix -- nothing is reported rather than something
    meaningless.
    """
    from scipy.sparse import csc_matrix

    from .schur import ReducedSystem

    eo_sigma: dict[str, list[float]] = {}
    point_sigma: dict[str, list[float]] = {}
    eo_covariance: dict[str, list[list[float]]] = {}
    ap_sigma: dict[str, float] = {}
    try:
        J = csc_matrix(jacobian)
        normal = (J.T @ J).tocsr()
        system = ReducedSystem(normal, obj_base, ap_base)
    except Exception:  # noqa: BLE001  -- singular or otherwise unfactorisable
        return eo_sigma, point_sigma, eo_covariance, ap_sigma

    scale = sigma0 * sigma0

    def root(value: float) -> float:
        return float(np.sqrt(value)) if np.isfinite(value) and value >= 0 else float("nan")

    for i, img in enumerate(free_images):
        block = system.covariance_block(range(6 * i, 6 * i + 6)) * scale
        block = 0.5 * (block + block.T)
        eo_covariance[img] = block.tolist()
        eo_sigma[img] = [root(block[k, k]) for k in range(6)]
    if ap_names:
        block = system.covariance_block(range(ap_base, ap_base + len(ap_names)))
        for k, name in enumerate(ap_names):
            ap_sigma[name] = root(scale * block[k, k])
    if point_ids:
        variances = system.point_variances(max_points)
        if variances is not None:
            for pid in point_ids:
                v = variances[point_index[pid]]
                point_sigma[pid] = [root(scale * v[k]) for k in range(3)]
    return eo_sigma, point_sigma, eo_covariance, ap_sigma


def corner_uncertainty(eo, covariance, focal_mm: float, half_width_mm: float,
                       half_height_mm: float, ground_z: float) -> float:
    """How far a photo's corners could land on the ground, 1 sigma, in metres.

    Each frame corner is taken to the ground at ``ground_z`` through the
    orientation, and the orientation's covariance is carried through to the
    ground position by first-order propagation (Jacobian by central
    differences). The largest of the four is returned: the worst place in
    the photo, which is what an orthophoto inherits.
    """
    eo = np.asarray(eo, dtype=float)
    cov = np.asarray(covariance, dtype=float)

    def ground_of(params, film_xy):
        rot = rotation_matrix(*params[3:])
        direction = rot.T @ np.array([film_xy[0], film_xy[1], -focal_mm])
        t = (ground_z - params[2]) / direction[2]
        return params[:2] + t * direction[:2]

    worst = 0.0
    steps = np.array([0.01, 0.01, 0.01, 1e-6, 1e-6, 1e-6])
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        corner = (sx * half_width_mm, sy * half_height_mm)
        jac = np.zeros((2, 6))
        for k in range(6):
            up, down = eo.copy(), eo.copy()
            up[k] += steps[k]
            down[k] -= steps[k]
            jac[:, k] = (ground_of(up, corner) - ground_of(down, corner)) / (2 * steps[k])
        ground_cov = jac @ cov @ jac.T
        worst = max(worst, float(np.sqrt(max(np.trace(ground_cov), 0.0))))
    return worst
