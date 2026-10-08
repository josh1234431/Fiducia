"""Automatic tie point collection.

Matching runs in two stages, as in every production aerial triangulation
package:

1. **Find.** Candidate correspondences are found on reduced copies of the
   photographs, where a search is cheap and robust:

   ``ncc``
       Normalised cross-correlation. Interest points are found in one image
       and template-matched into the other, seeded by a prediction of where
       each point falls. Robust on aerial photography with similar
       illumination.

   ``fbm``
       Feature-based matching using AKAZE descriptors. Scale- and rotation-
       invariant, so it copes with imagery at different orientations or
       resolutions where NCC fails outright, and it needs no prediction.

   Both finish with a RANSAC fundamental-matrix test, which discards matches
   that are individually convincing but geometrically impossible.

2. **Refine.** Each surviving match is measured again at full resolution by
   least-squares matching (Gruen, 1985): the other photo's patch is warped
   into this photo's geometry through the pair's homography, then an affine
   geometric and a linear radiometric correction are solved for. That makes
   the position precise to about a tenth of a pixel and gives its standard
   deviation from the fit itself. Reduced-image matching alone is only good
   to about one reduced pixel -- eight full-resolution pixels on a large
   digital frame -- which is far too coarse for an adjustment that expects
   half a pixel.

Points are matched from one photo into every photo that overlaps it, so a
feature seen on three photos becomes one three-ray tie point rather than
three unrelated two-ray ones. Multi-ray points are what carry scale along a
strip.

Image coordinates throughout have whole numbers at pixel centres.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

__all__ = ["TiePointMatch", "TiePoint", "TiePointOptions", "collect_tie_points",
           "collect_multi", "least_squares_match", "refine_point", "MATCH_METHODS",
           "REFINE_METHODS"]

MATCH_METHODS = ("ncc", "fbm", "sift")   # sift: drone blocks, see drone.py
REFINE_METHODS = ("lsm", "ncc", "none")


@dataclass
class TiePointMatch:
    left_image: str
    right_image: str
    left_col: float
    left_row: float
    right_col: float
    right_row: float
    score: float
    sigma_px: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "leftImage": self.left_image,
            "rightImage": self.right_image,
            "leftCol": self.left_col,
            "leftRow": self.left_row,
            "rightCol": self.right_col,
            "rightRow": self.right_row,
            "score": self.score,
            "sigmaPx": self.sigma_px,
        }


@dataclass
class TiePoint:
    """One ground feature measured on two or more photos."""

    # (image_id, col, row, sigma_px or None), the master photo first.
    observations: list = field(default_factory=list)
    score: float = 0.0

    @property
    def rays(self) -> int:
        return len(self.observations)


@dataclass
class TiePointOptions:
    method: str = "ncc"
    target_count: int = 60
    template_size: int = 31
    search_radius: int = 220
    min_score: float = 0.75
    working_max_px: int = 2400
    ransac_threshold: float = 3.0
    min_separation_px: int = 40
    bands: Optional[Sequence[int]] = None
    # Full-resolution refinement: least-squares matching, correlation alone,
    # or none (positions from the reduced image only).
    refine: str = "lsm"
    # Refined matches poorer than this are dropped, in full-resolution pixels.
    max_sigma_px: float = 0.5


# -- reduced images ----------------------------------------------------------


def _load_working_image(path: str, max_px: int) -> tuple[np.ndarray, tuple[float, float]]:
    """A reduced single-band view and its (x, y) reduction factors.

    A reduced pixel i covers full-resolution pixels [i s, (i + 1) s), so its
    centre is at (i + 0.5) s - 0.5 in full-resolution pixel-centre
    coordinates; see :func:`_to_full`.
    """
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as dataset:
        factor = max(1.0, max(dataset.width, dataset.height) / max_px)
        out_w = max(1, int(dataset.width / factor))
        out_h = max(1, int(dataset.height / factor))
        band = 1 if dataset.count < 3 else 2   # green is the sharpest channel
        data = dataset.read(
            band, out_shape=(out_h, out_w), resampling=Resampling.average
        ).astype(np.float32)
        scale = (dataset.width / out_w, dataset.height / out_h)

    finite = data[np.isfinite(data)]
    if finite.size:
        lo, hi = np.percentile(finite, [2, 98])
        span = max(float(hi - lo), 1e-6)
        data = np.clip((data - lo) / span, 0, 1) * 255.0
    return data.astype(np.uint8), (float(scale[0]), float(scale[1]))


def _to_full(col, row, scale):
    return (np.asarray(col, float) + 0.5) * scale[0] - 0.5, (np.asarray(row, float) + 0.5) * scale[1] - 0.5


def _to_reduced(col, row, scale):
    return (np.asarray(col, float) + 0.5) / scale[0] - 0.5, (np.asarray(row, float) + 0.5) / scale[1] - 0.5


def _detect_interest_points(image: np.ndarray, count: int, separation: int) -> np.ndarray:
    """Shi-Tomasi corners, spread out so matches are not all in one field."""
    import cv2

    corners = cv2.goodFeaturesToTrack(
        image,
        maxCorners=max(count * 4, 200),
        qualityLevel=0.01,
        minDistance=max(8, separation),
        blockSize=7,
    )
    if corners is None:
        return np.empty((0, 2), dtype=float)
    return corners.reshape(-1, 2).astype(float)


def _parabola(values: np.ndarray, index: tuple[int, int]) -> tuple[float, float]:
    """Subpixel peak of a 2-D surface by separate parabolas through the peak."""
    my, mx = index
    dx = dy = 0.0
    if 0 < mx < values.shape[1] - 1:
        a, b, c = values[my, mx - 1], values[my, mx], values[my, mx + 1]
        denom = a - 2 * b + c
        if abs(denom) > 1e-9:
            dx = float(np.clip(0.5 * (a - c) / denom, -1.0, 1.0))
    if 0 < my < values.shape[0] - 1:
        a, b, c = values[my - 1, mx], values[my, mx], values[my + 1, mx]
        denom = a - 2 * b + c
        if abs(denom) > 1e-9:
            dy = float(np.clip(0.5 * (a - c) / denom, -1.0, 1.0))
    return dx, dy


def _match_ncc(
    left: np.ndarray,
    right: np.ndarray,
    points: np.ndarray,
    options: TiePointOptions,
    predictor: Optional[Callable[[float, float], tuple[float, float]]],
) -> list[tuple[float, float, float, float, float]]:
    import cv2

    half = max(5, options.template_size // 2)
    results = []

    for col, row in points:
        c, r = int(round(col)), int(round(row))
        if c - half < 0 or r - half < 0 or c + half >= left.shape[1] or r + half >= left.shape[0]:
            continue
        template = left[r - half: r + half + 1, c - half: c + half + 1]
        if float(template.std()) < 6.0:
            continue  # featureless patch -- correlation would be meaningless

        if predictor is not None:
            pc, pr = predictor(c, r)
            # Predicted off the other photo: this point is outside the overlap.
            if not (np.isfinite(pc) and np.isfinite(pr)) or not (
                    half <= pc < right.shape[1] - half and half <= pr < right.shape[0] - half):
                continue
        else:
            pc, pr = c, r

        radius = options.search_radius
        c0 = int(max(0, pc - radius))
        r0 = int(max(0, pr - radius))
        c1 = int(min(right.shape[1], pc + radius))
        r1 = int(min(right.shape[0], pr + radius))
        if c1 - c0 < template.shape[1] or r1 - r0 < template.shape[0]:
            continue

        window = right[r0:r1, c0:c1]
        response = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
        _, best, _, location = cv2.minMaxLoc(response)
        if best < options.min_score:
            continue

        mx, my = location
        dx, dy = _parabola(response, (my, mx))
        # The template is centred on the integer pixel (c, r).
        results.append((float(c), float(r), c0 + mx + dx + half, r0 + my + dy + half, float(best)))

    return results


def _detect_akaze(image: np.ndarray):
    import cv2

    return cv2.AKAZE_create().detectAndCompute(image, None)


def _match_fbm(left_features, right_features) -> list[tuple[float, float, float, float, float]]:
    import cv2

    kp_left, desc_left = left_features
    kp_right, desc_right = right_features
    if desc_left is None or desc_right is None or len(kp_left) < 8 or len(kp_right) < 8:
        return []

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = matcher.knnMatch(desc_left, desc_right, k=2)

    results = []
    for pair in raw:
        if len(pair) < 2:
            continue
        best, second = pair
        # Lowe's ratio test: a match is only trustworthy if it is clearly
        # better than the next best candidate.
        if best.distance > 0.75 * second.distance:
            continue
        lp = kp_left[best.queryIdx].pt
        rp = kp_right[best.trainIdx].pt
        score = 1.0 - min(best.distance / 256.0, 1.0)
        results.append((lp[0], lp[1], rp[0], rp[1], score))

    return results


def _geometric_filter(
    matches: list[tuple[float, float, float, float, float]], threshold: float
) -> list[tuple[float, float, float, float, float]]:
    """Discard matches inconsistent with a single two-view geometry."""
    import cv2

    if len(matches) < 8:
        return []

    left = np.array([[m[0], m[1]] for m in matches], dtype=np.float32)
    right = np.array([[m[2], m[3]] for m in matches], dtype=np.float32)

    _, inliers = cv2.findFundamentalMat(
        left, right, cv2.FM_RANSAC, threshold, 0.99, maxIters=5000
    )
    if inliers is None:
        return []

    keep = inliers.ravel().astype(bool)
    filtered = [m for m, ok in zip(matches, keep) if ok]
    # Too few consistent matches means the pair did not really match. The
    # unfiltered list is the very set that just failed the test, so passing
    # it on would put false tie points into the adjustment.
    return filtered if len(filtered) >= 8 else []


# -- full-resolution refinement ------------------------------------------------


def _project(h: np.ndarray, cols: np.ndarray, rows: np.ndarray):
    w = h[2, 0] * cols + h[2, 1] * rows + h[2, 2]
    return ((h[0, 0] * cols + h[0, 1] * rows + h[0, 2]) / w,
            (h[1, 0] * cols + h[1, 1] * rows + h[1, 2]) / w)


class _Window:
    """A block of one band of a photo, read once, sampled bilinearly."""

    def __init__(self, dataset, band: int, c0: int, r0: int, c1: int, r1: int):
        from rasterio.windows import Window

        c0, r0 = max(0, c0), max(0, r0)
        c1, r1 = min(dataset.width, c1), min(dataset.height, r1)
        self.c0, self.r0 = c0, r0
        self.ok = c1 - c0 >= 4 and r1 - r0 >= 4
        if self.ok:
            self.data = dataset.read(band, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(np.float32)
            gy, gx = np.gradient(self.data)
            self.gx, self.gy = gx.astype(np.float32), gy.astype(np.float32)

    def sample(self, layer: np.ndarray, cols: np.ndarray, rows: np.ndarray):
        from scipy.ndimage import map_coordinates

        local_r = rows - self.r0
        local_c = cols - self.c0
        inside = ((local_r >= 0) & (local_r <= layer.shape[0] - 1)
                  & (local_c >= 0) & (local_c <= layer.shape[1] - 1))
        values = map_coordinates(layer, np.vstack([local_r.ravel(), local_c.ravel()]),
                                 order=1, mode="nearest").reshape(cols.shape)
        return values, inside


def least_squares_match(
    template: np.ndarray,
    right: "_Window",
    centre_right: tuple[float, float],
    jacobian: np.ndarray,
    max_iterations: int = 20,
) -> Optional[tuple[float, float, float, float]]:
    """Gruen least-squares matching of a template into the other photo.

    ``template`` is a square patch of the left photo centred on the point.
    The right photo is sampled through the affine
    ``x_r = centre_right + J (dx, dy) + (a0 + a1 dx + a2 dy, b0 + b1 dx + b2 dy)``,
    with (dx, dy) the template offsets, and the six geometric and two
    radiometric (offset, gain) parameters solved by Gauss-Newton.

    Returns (col, row, sigma_px, correlation) on the right photo, or None if
    the solution does not converge or wanders from its start.
    """
    half = template.shape[0] // 2
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    dx, dy = np.meshgrid(offsets, offsets)
    dx, dy = dx.ravel(), dy.ravel()
    f = template.astype(np.float64).ravel()

    a = np.zeros(3)
    b = np.zeros(3)
    r0, r1 = 0.0, 1.0
    base_c = centre_right[0] + jacobian[0, 0] * dx + jacobian[0, 1] * dy
    base_r = centre_right[1] + jacobian[1, 0] * dx + jacobian[1, 1] * dy

    # Radiometry first, so the geometric steps start from matched tones.
    g, inside = right.sample(right.data, base_c, base_r)
    if not inside.all():
        return None
    gm, gs = g.mean(), g.std()
    if gs < 1e-6:
        return None
    r1 = f.std() / gs
    r0 = f.mean() - r1 * gm

    design = np.empty((f.size, 8))
    converged = False
    for _ in range(max_iterations):
        cols = base_c + a[0] + a[1] * dx + a[2] * dy
        rows = base_r + b[0] + b[1] * dx + b[2] * dy
        g, inside = right.sample(right.data, cols, rows)
        if not inside.all():
            return None
        gx, _ = right.sample(right.gx, cols, rows)
        gy, _ = right.sample(right.gy, cols, rows)
        residual = f - (r0 + r1 * g)
        design[:, 0] = r1 * gx
        design[:, 1] = r1 * gx * dx
        design[:, 2] = r1 * gx * dy
        design[:, 3] = r1 * gy
        design[:, 4] = r1 * gy * dx
        design[:, 5] = r1 * gy * dy
        design[:, 6] = 1.0
        design[:, 7] = g
        normal = design.T @ design
        try:
            step = np.linalg.solve(normal, design.T @ residual)
        except np.linalg.LinAlgError:
            return None
        a += step[0:3]
        b += step[3:6]
        r0 += step[6]
        r1 += step[7]
        if abs(a[0]) > 4.0 or abs(b[0]) > 4.0 or not (0.2 < r1 < 5.0):
            return None
        if abs(step[0]) < 0.002 and abs(step[3]) < 0.002:
            converged = True
            break
    if not converged:
        return None

    cols = base_c + a[0] + a[1] * dx + a[2] * dy
    rows = base_r + b[0] + b[1] * dx + b[2] * dy
    g, _ = right.sample(right.data, cols, rows)
    residual = f - (r0 + r1 * g)
    dof = f.size - 8
    s0_sq = float(residual @ residual) / max(dof, 1)
    try:
        covariance = s0_sq * np.linalg.inv(normal)
    except np.linalg.LinAlgError:
        return None
    sigma = float(np.sqrt(max(covariance[0, 0], 0.0) + max(covariance[3, 3], 0.0)))
    correlation = float(np.corrcoef(f, g)[0, 1]) if g.std() > 0 else 0.0
    return centre_right[0] + a[0], centre_right[1] + b[0], sigma, correlation


def _refine_pair(left_ds, right_ds, matches_full, options, band: int, scale: float):
    """Remeasure reduced-image matches at full resolution.

    ``matches_full`` are (left_col, left_row, right_col, right_row, score) in
    full-resolution pixels. The left point is kept on its integer pixel and
    the right point measured to subpixel precision.
    """
    import cv2

    if len(matches_full) < 8:
        return []
    src = np.float64([[m[0], m[1]] for m in matches_full])
    dst = np.float64([[m[2], m[3]] for m in matches_full])
    homography, _ = cv2.findHomography(src, dst, cv2.RANSAC, max(4.0, 1.5 * scale))
    if homography is None:
        return []

    half = int(np.clip(round(3 * scale), 15, 40))
    search = int(np.ceil(2.5 * scale)) + 4
    refined = []
    for lc, lr, rc, rr, score in matches_full:
        c, r = int(round(lc)), int(round(lr))
        # The local affine of the homography at this point.
        pc, pr = _project(homography, np.array([c, c + 1.0, c]), np.array([r, r, r + 1.0]))
        jac = np.array([[pc[1] - pc[0], pc[2] - pc[0]], [pr[1] - pr[0], pr[2] - pr[0]]])
        centre = (float(pc[0]), float(pr[0]))
        found = refine_point(left_ds, right_ds, band, c, r, centre, jac, half, search,
                             options.min_score - 0.1, options.refine, options.max_sigma_px)
        if found is not None:
            col, row, correlation, sigma = found
            refined.append((float(c), float(r), col, row, correlation, sigma))
    return refined


def refine_point(left_ds, right_ds, band: int, c: int, r: int, centre, jac,
                 half: int, search: int, min_score: float, method: str = "lsm",
                 max_sigma_px: float = 0.5):
    """Measure the left photo's pixel (c, r) on the right photo.

    ``centre`` is the predicted right position and ``jac`` the local 2 x 2
    mapping from left to right pixels there. The right photo is resampled
    into the left's geometry; correlation over ``search`` pixels removes the
    prediction's error, then least-squares matching refines it. Returns
    (col, row, correlation, sigma_px or None), or None if the point does not
    match convincingly.
    """
    import cv2

    jac = np.asarray(jac, dtype=float)
    if not np.isfinite(jac).all() or not np.isfinite(centre).all():
        return None
    if (c - half - search < 0 or r - half - search < 0
            or c + half + search >= left_ds.width or r + half + search >= left_ds.height):
        return None
    left = _Window(left_ds, band, c - half, r - half, c + half + 1, r + half + 1)
    if not left.ok or left.data.shape != (2 * half + 1, 2 * half + 1):
        return None
    template = left.data
    if float(template.std()) < 4.0:
        return None

    reach = int(np.ceil((half + search) * max(1.0, np.abs(jac).sum(axis=1).max()))) + 4
    right = _Window(right_ds, band, int(centre[0]) - reach, int(centre[1]) - reach,
                    int(centre[0]) + reach + 1, int(centre[1]) + reach + 1)
    if not right.ok:
        return None

    span = half + search
    offsets = np.arange(-span, span + 1, dtype=np.float64)
    ox, oy = np.meshgrid(offsets, offsets)
    warped, inside = right.sample(
        right.data, centre[0] + jac[0, 0] * ox + jac[0, 1] * oy,
        centre[1] + jac[1, 0] * ox + jac[1, 1] * oy)
    if not inside.all():
        return None
    response = cv2.matchTemplate(warped.astype(np.float32), template, cv2.TM_CCOEFF_NORMED)
    _, best, _, location = cv2.minMaxLoc(response)
    mx, my = location
    if best < min_score or not (0 < mx < response.shape[1] - 1 and 0 < my < response.shape[0] - 1):
        return None
    sx, sy = _parabola(response, (my, mx))
    shift = np.array([mx + sx - search, my + sy - search])
    start = (centre[0] + jac[0] @ shift, centre[1] + jac[1] @ shift)

    if method != "lsm":
        return float(start[0]), float(start[1]), float(best), None
    fit = least_squares_match(template, right, start, jac)
    if fit is None:
        return None
    col, row, sigma, correlation = fit
    if sigma > max_sigma_px or correlation < min_score:
        return None
    return float(col), float(row), float(correlation), float(sigma)


def _thin(points: list[TiePoint], target: int, separation: float) -> list[TiePoint]:
    """Keep the strongest points while forcing spatial spread.

    Points seen on more photos rank first: they tie more of the block
    together. Then quality. A hundred points clustered in one corner
    constrain the block far less than twenty spread across the overlap.
    """
    ordered = sorted(points, key=lambda p: (p.rays, p.score), reverse=True)
    kept: list[TiePoint] = []
    per_partner: dict[str, int] = {}
    for candidate in ordered:
        partners = [obs[0] for obs in candidate.observations[1:]]
        if partners and all(per_partner.get(p, 0) >= target for p in partners):
            continue
        lc, lr = candidate.observations[0][1:3]
        if all((lc - k.observations[0][1]) ** 2 + (lr - k.observations[0][2]) ** 2
               >= separation ** 2 for k in kept):
            kept.append(candidate)
            for p in partners:
                per_partner[p] = per_partner.get(p, 0) + 1
    return kept


def collect_multi(
    master: dict,
    partners: list[dict],
    options: Optional[TiePointOptions] = None,
    progress: Optional[Callable[[float, str], None]] = None,
) -> list[TiePoint]:
    """Tie points from one photo into every photo that overlaps it.

    ``master`` and each partner carry ``id`` and ``path``; each partner may
    carry ``predictor``, mapping a master full-resolution pixel to the
    partner's. Interest points are found once on the master, so a feature
    matched into two partners is one three-ray point.
    """
    import rasterio

    options = options or TiePointOptions()
    say = progress or (lambda fraction, message: None)

    say(0.02, "Loading working images")
    left, left_scale = _load_working_image(master["path"], options.working_max_px)
    if options.method == "fbm":
        left_features = _detect_akaze(left)
    else:
        corners = _detect_interest_points(left, options.target_count, options.min_separation_px)

    # Master pixel (integer, full resolution) -> {partner id: (col, row, score, sigma)}
    found: dict[tuple[int, int], dict[str, tuple]] = {}
    with rasterio.open(master["path"]) as left_ds:
        band = 1 if left_ds.count < 3 else 2
        for index, partner in enumerate(partners):
            def part(fraction, message, index=index):
                say((index + fraction) / max(len(partners), 1), message)

            part(0.05, "Matching reduced images")
            right, right_scale = _load_working_image(partner["path"], options.working_max_px)

            if options.method == "fbm":
                matches = _match_fbm(left_features, _detect_akaze(right))
            else:
                predictor = partner.get("predictor")
                scaled = None
                if predictor is not None:
                    def scaled(col, row, predictor=predictor):
                        full_c, full_r = _to_full(col, row, left_scale)
                        pc, pr = predictor(float(full_c), float(full_r))
                        rc, rr = _to_reduced(pc, pr, right_scale)
                        return float(rc), float(rr)
                matches = _match_ncc(left, right, corners, options, scaled)
            matches = _geometric_filter(matches, options.ransac_threshold)
            if not matches:
                continue

            full = []
            for lc, lr, rc, rr, score in matches:
                flc, flr = _to_full(lc, lr, left_scale)
                frc, frr = _to_full(rc, rr, right_scale)
                full.append((float(flc), float(flr), float(frc), float(frr), score))

            if options.refine in ("lsm", "ncc"):
                part(0.4, f"Refining {len(full)} matches at full resolution")
                with rasterio.open(partner["path"]) as right_ds:
                    refined = _refine_pair(left_ds, right_ds, full, options, band,
                                           float(np.mean(left_scale)))
            else:
                refined = [(round(m[0]), round(m[1]), m[2], m[3], m[4], None) for m in full]

            for lc, lr, rc, rr, score, sigma in refined:
                key = (int(lc), int(lr))
                found.setdefault(key, {})[partner["id"]] = (rc, rr, score, sigma)

    points = [
        TiePoint(
            observations=[(master["id"], float(c), float(r), None)]
            + [(pid, m[0], m[1], m[3]) for pid, m in hits.items()],
            score=float(np.mean([m[2] for m in hits.values()])),
        )
        for (c, r), hits in found.items()
    ]
    # Spread in the master image is measured in full-resolution pixels.
    separation = options.min_separation_px * float(np.mean(left_scale))
    kept = _thin(points, options.target_count, separation)
    say(1.0, f"Accepted {len(kept)} tie points")
    return kept


def collect_tie_points(
    left_path: str,
    right_path: str,
    left_id: str,
    right_id: str,
    options: Optional[TiePointOptions] = None,
    predictor: Optional[Callable[[float, float], tuple[float, float]]] = None,
    progress: Optional[Callable[[float, str], None]] = None,
) -> list[TiePointMatch]:
    """Find tie points between one image pair (see :func:`collect_multi`)."""
    points = collect_multi(
        {"id": left_id, "path": left_path},
        [{"id": right_id, "path": right_path, "predictor": predictor}],
        options, progress,
    )
    out = []
    for point in points:
        (_, lc, lr, _), (_, rc, rr, sigma) = point.observations[:2]
        out.append(TiePointMatch(left_id, right_id, lc, lr, rc, rr, point.score, sigma))
    return out
