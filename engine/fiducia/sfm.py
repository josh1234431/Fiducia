"""Structure from motion without positions: cameras and points from the photos alone.

The drone workflow starts every photo at its GPS position. Photos taken by
hand, from the ground, of a rock face or a building, or from a drone without
GPS, have no such start: where each camera stood has to come from the photos
themselves. This is incremental structure from motion (as in Snavely et al.,
2006, and Schonberger and Frahm, 2016):

1. Features are matched between photos (every pair for a small set, a
   shortlist of likely neighbours for a large one) and checked against the
   two-view geometry of a calibrated camera.
2. Matches are joined into tracks: one point on the object, followed across
   every photo that sees it.
3. The pair with the most matches and a real baseline -- not a pure rotation
   or a flat wall seen straight on -- starts the model: its relative pose
   from the essential matrix, and its shared tracks triangulated.
4. Photos are added one at a time, the one that sees the most points already
   placed first: its pose from those points (perspective-n-point, robust to
   bad matches), then the tracks it newly sees are triangulated.
5. After each addition, and once at the end, a bundle adjustment refines
   every pose, every point and the lens (focal length and radial
   distortion), with a robust loss, and drops observations that still
   disagree.

The result is in a frame of its own: the first camera at the origin, the
distance to the second camera as the unit of length. Scale, position and
orientation come from ground control or a scale bar afterwards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from .drone import DroneMatchOptions, _features, _match, _to_full, _Union, _as_completed, _worker_count

__all__ = ["SfmOptions", "Intrinsics", "match_photos", "reconstruct", "similarity"]


@dataclass
class SfmOptions:
    matching: DroneMatchOptions = field(default_factory=lambda: DroneMatchOptions(depth_band=None))
    exhaustive_up_to: int = 60       # photos; above this, each is matched with a shortlist
    shortlist: int = 15              # likely neighbours per photo in a large set
    max_reprojection_px: float = 4.0 # observations further off than this are dropped, in the end
    first_reprojection: float = 0.04     # ... and at first, before the lens is known, as a
                                         # share of the photo's diagonal (a wide lens bends
                                         # its corners by that much)
    min_angle_deg: float = 1.5       # a point seen from directions closer than this has no depth
    min_pnp_inliers: int = 30        # a photo needs this many placed points to be added
    refine_lens: bool = True         # refine focal length and radial distortion
    # How far the focal length may stray from the photos' own record, as a
    # standard deviation. Photos looking straight down on flat ground cannot
    # tell a longer lens from a higher camera, so without this the focal
    # length drifts; strong geometry (oblique views, relief) still overrules it.
    # None: no prior, for a focal length that is only a guess.
    focal_prior: Optional[float] = 0.05
    # At the end the model is adjusted twice, the focal length free and held
    # at the photos' record, and the free one is kept only if it fits at least
    # this much better. Freed on weak evidence it drifts with the distortion
    # terms instead (by a fifth, on a real drone flight, for 3% better fit).
    focal_evidence: float = 0.05
    focal_known: bool = True         # False when the starting focal length is only a guess


@dataclass
class Intrinsics:
    """A pinhole camera with radial distortion, in full-resolution pixels."""
    focal: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0

    def project(self, cam: np.ndarray) -> np.ndarray:
        """Camera-frame points (x right, y down, z forward) to pixels."""
        x = cam[:, 0] / cam[:, 2]
        y = cam[:, 1] / cam[:, 2]
        r2 = x * x + y * y
        d = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
        return np.column_stack([self.focal * x * d + self.cx, self.focal * y * d + self.cy])

    def normalise(self, px: np.ndarray) -> np.ndarray:
        """Pixels to undistorted image-plane coordinates (z = 1)."""
        xd = (px[:, 0] - self.cx) / self.focal
        yd = (px[:, 1] - self.cy) / self.focal
        x, y = xd.copy(), yd.copy()
        for _ in range(8):
            r2 = x * x + y * y
            d = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
            x, y = xd / d, yd / d
        return np.column_stack([x, y])


# -- matching ------------------------------------------------------------------------------


def _shortlist(features: list, count: int, sample: int = 600) -> list[tuple[int, int]]:
    """Likely neighbours for each photo, from a quick match of its strongest features."""
    import cv2

    n = len(features)
    strongest = []
    for f in features:
        if f["descriptors"] is None:
            strongest.append(None)
            continue
        order = np.argsort(-f["strength"])[:sample]
        strongest.append(f["descriptors"][order].astype(np.float32))
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    scores = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if strongest[i] is None or strongest[j] is None:
                continue
            scores[i, j] = scores[j, i] = len(matcher.match(strongest[i], strongest[j]))
    pairs = set()
    for i in range(n):
        for j in np.argsort(-scores[i])[:count]:
            if j != i and scores[i, j] > 0:
                pairs.add((min(i, int(j)), max(i, int(j))))
    return sorted(pairs)


def match_photos(
    photos: Sequence[dict],
    focal_px: float,
    options: Optional[SfmOptions] = None,
    progress: Optional[Callable[[float, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> tuple[list, dict, dict]:
    """Tracks across photos with no positions.

    ``photos`` carry ``path``; ``focal_px`` is the focal length in full-size
    pixels (from the photos' metadata, or a guess). Returns the tracks (each
    a list of (photo, column, row) in full-size pixels), each matched pair's
    inlier pixels for choosing where to start, and a summary.
    """
    from concurrent.futures import ThreadPoolExecutor

    options = options or SfmOptions()
    say = progress or (lambda fraction, message: None)
    cancelled = should_cancel or (lambda: False)
    m = options.matching
    n = len(photos)

    features: list = [None] * n
    with ThreadPoolExecutor(max_workers=_worker_count()) as pool:
        futures = {pool.submit(_features, photo["path"], m): i for i, photo in enumerate(photos)}
        for done, future in enumerate(_as_completed(futures, cancelled)):
            features[futures[future]] = future.result()
            say(0.3 * (done + 1) / max(n, 1), f"Found features in {done + 1} of {n} photos")

    if n <= options.exhaustive_up_to:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        say(0.32, "Shortlisting likely neighbours")
        pairs = _shortlist(features, options.shortlist)

    def match(pair):
        i, j = pair
        return _match(features[i], features[j], focal_px * float(np.mean(features[i]["scale"])), m)

    results: list = [None] * len(pairs)
    with ThreadPoolExecutor(max_workers=_worker_count()) as pool:
        futures = {pool.submit(match, pair): k for k, pair in enumerate(pairs)}
        for done, future in enumerate(_as_completed(futures, cancelled)):
            results[futures[future]] = future.result()
            say(0.35 + 0.55 * (done + 1) / max(len(pairs), 1), f"Matched {done + 1} of {len(pairs)} pairs")

    def full(i, f):
        x, y = features[i]["points"][f]
        col, row = _to_full(x, y, features[i]["scale"])
        return float(col), float(row)

    union = _Union()
    pair_matches = {}
    for (i, j), inliers in zip(pairs, results):
        if not len(inliers):
            continue
        pair_matches[(i, j)] = (np.array([full(i, a) for a, _ in inliers]),
                                np.array([full(j, b) for _, b in inliers]))
        for a, b in inliers:
            union.join((i, int(a)), (j, int(b)))

    say(0.92, "Joining matches into tracks")
    groups: dict = {}
    for node in list(union.parent):
        groups.setdefault(union.find(node), []).append(node)
    tracks = []
    for nodes in groups.values():
        images = [node[0] for node in nodes]
        # Two features of one photo in one track means a false match somewhere.
        if len(nodes) < 2 or len(set(images)) != len(images):
            continue
        tracks.append([(i, *full(i, f)) for i, f in sorted(nodes)])

    summary = {"photos": n, "pairsTried": len(pairs), "pairsMatched": len(pair_matches),
               "tracks": len(tracks)}
    say(1.0, f"{len(tracks):,} tracks from {len(pair_matches)} matched pairs")
    return tracks, pair_matches, summary


# -- geometry --------------------------------------------------------------------------------


def _rotation(rvec: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))[0]


def _rvec(rotation: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.Rodrigues(rotation)[0].ravel()


def _triangulate(rays: list) -> Optional[np.ndarray]:
    """A point from several (R, t, normalised xy) observations, by linear least squares."""
    rows = []
    for R, t, (x, y) in rays:
        P = np.hstack([R, t.reshape(3, 1)])
        rows.append(x * P[2] - P[0])
        rows.append(y * P[2] - P[1])
    _, _, vt = np.linalg.svd(np.array(rows))
    X = vt[-1]
    if abs(X[3]) < 1e-12:
        return None
    return X[:3] / X[3]


def _max_angle(point: np.ndarray, centres: list) -> float:
    """The widest angle at the point between any two of the rays to it, degrees."""
    rays = [(c - point) / max(np.linalg.norm(c - point), 1e-12) for c in centres]
    best = 0.0
    for a in range(len(rays)):
        for b in range(a + 1, len(rays)):
            best = max(best, math.degrees(math.acos(float(np.clip(rays[a] @ rays[b], -1, 1)))))
    return best


def similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """The scale, rotation and shift best taking source points onto target ones (Umeyama, 1991)."""
    ms, mt = source.mean(axis=0), target.mean(axis=0)
    a, b = source - ms, target - mt
    u, d, vt = np.linalg.svd(b.T @ a / len(source))
    flip = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        flip[2, 2] = -1
    R = u @ flip @ vt
    scale = float(np.trace(np.diag(d) @ flip) / (a ** 2).sum(axis=1).mean())
    return scale, R, mt - scale * R @ ms


# -- reconstruction --------------------------------------------------------------------------


class _Model:
    """The growing reconstruction: poses, points and the observations tying them."""

    def __init__(self, tracks, intrinsics: Intrinsics, options: SfmOptions):
        self.tracks = tracks
        self.k = intrinsics
        self.options = options
        self.poses: dict = {}            # photo -> (rvec, t), world to camera
        self.points: dict = {}           # track -> xyz
        self.dropped: set = set()        # (track, photo) observations judged wrong, for now
        # How far off an observation may be. Generous while the lens is still
        # unknown (a wide lens bends the corners by hundreds of pixels), then
        # tightened as each adjustment learns it.
        self.threshold = options.first_reprojection * math.hypot(intrinsics.cx * 2, intrinsics.cy * 2)
        self.focal_prior = intrinsics.focal       # the focal length the photos record
        self.by_photo: dict = {}
        for t, track in enumerate(tracks):
            for i, col, row in track:
                self.by_photo.setdefault(i, []).append(t)

    def centre(self, i):
        rvec, t = self.poses[i]
        return -_rotation(rvec).T @ t

    def observations(self, t):
        return [(i, c, r) for i, c, r in self.tracks[t] if (t, i) not in self.dropped]

    def try_point(self, t) -> bool:
        seen = [(i, c, r) for i, c, r in self.observations(t) if i in self.poses]
        if len(seen) < 2:
            return False
        norm = self.k.normalise(np.array([[c, r] for _, c, r in seen]))
        rays = []
        for (i, _, _), xy in zip(seen, norm):
            rvec, tvec = self.poses[i]
            rays.append((_rotation(rvec), tvec, xy))
        X = _triangulate(rays)
        if X is None or not np.all(np.isfinite(X)):
            return False
        # In front of every camera, close to every observation, and seen from
        # directions far enough apart to have a depth at all.
        for (i, c, r), (R, tvec, _) in zip(seen, rays):
            cam = R @ X + tvec
            if cam[2] <= 0:
                return False
            px = self.k.project(cam.reshape(1, 3))[0]
            if math.hypot(px[0] - c, px[1] - r) > self.threshold:
                return False
        if _max_angle(X, [self.centre(i) for i, _, _ in seen]) < self.options.min_angle_deg:
            return False
        self.points[t] = X
        return True

    def place(self, i) -> bool:
        import cv2

        known = [(t, c, r) for t in self.by_photo.get(i, []) if t in self.points
                 for j, c, r in self.observations(t) if j == i]
        if len(known) < self.options.min_pnp_inliers:
            return False
        world = np.array([self.points[t] for t, _, _ in known])
        norm = self.k.normalise(np.array([[c, r] for _, c, r in known]))
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            world, norm, np.eye(3), None, iterationsCount=2000,
            reprojectionError=self.threshold / self.k.focal,
            confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inliers is None or len(inliers) < self.options.min_pnp_inliers:
            return False
        keep = inliers.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(world[keep], norm[keep], np.eye(3), None, rvec, tvec)
        self.poses[i] = (rvec.ravel(), tvec.ravel())
        kept = set(keep.tolist())
        for k, (t, _, _) in enumerate(known):
            if k not in kept:
                self.dropped.add((t, i))
        return True

    def adjust(self, fixed: int, scale_pair: tuple, refine_lens: bool, iterations: int = 30) -> None:
        """Bundle adjustment of every pose, every point and the lens.

        Levenberg-Marquardt with exact derivatives, the points eliminated by
        the Schur complement so that only the cameras and the lens are solved
        directly (Triggs et al., 2000), and a Huber weight on each
        observation so that a few bad matches cannot pull the solution. Two
        photos cannot tell focal length from distance, so until three are
        placed the focal length is held and only the distortion is learned.
        The first camera stays where it is and the distance to the second
        holds the scale, which the photos alone cannot fix.
        """
        from scipy import sparse
        from scipy.linalg import cho_factor, cho_solve

        a, b = scale_pair
        photos = [i for i in sorted(self.poses) if i != fixed]
        cam_index = {i: n for n, i in enumerate(photos)}
        point_ids = sorted(self.points)
        if not point_ids:
            return
        pt_index = {t: n for n, t in enumerate(point_ids)}
        obs = [(t, i, c, r) for t in point_ids for i, c, r in self.observations(t) if i in self.poses]
        hold_focal = len(self.poses) < 3
        n_lens = 3 if refine_lens else 0
        n_cam, n_pt = len(photos), len(point_ids)
        m = n_lens + 6 * n_cam

        every = [fixed] + photos
        slot = {i: n for n, i in enumerate(every)}
        R = np.array([_rotation(self.poses[i][0]) for i in every])
        T = np.array([np.asarray(self.poses[i][1], float) for i in every])
        P = np.array([self.points[t] for t in point_ids])
        f0 = self.k.focal
        lens = np.array([1.0, self.k.k1, self.k.k2])
        o_t = np.array([pt_index[t] for t, _, _, _ in obs])
        o_s = np.array([slot[i] for _, i, _, _ in obs])
        o_c = np.array([cam_index.get(i, -1) for _, i, _, _ in obs])
        o_px = np.array([[c, r] for _, _, c, r in obs])
        cx, cy = self.k.cx, self.k.cy
        huber = 2.0
        target = np.linalg.norm(-R[slot[a]].T @ T[slot[a]] + R[slot[b]].T @ T[slot[b]])
        scale_weight = 1e3

        def project(R, T, P, lens):
            focal = f0 * lens[0]
            Xc = (R[o_s] @ P[o_t][:, :, None])[:, :, 0] + T[o_s]
            z = Xc[:, 2]
            x, y = Xc[:, 0] / z, Xc[:, 1] / z
            r2 = x * x + y * y
            d = 1 + lens[1] * r2 + lens[2] * r2 * r2
            uv = np.column_stack([focal * x * d + cx, focal * y * d + cy])
            return Xc, x, y, r2, d, focal, uv - o_px

        def cost_of(e):
            n = np.hypot(e[:, 0], e[:, 1])
            return float(np.where(n <= huber, 0.5 * n * n, huber * (n - 0.5 * huber)).sum())

        def skew(v):
            z = np.zeros(len(v))
            return np.stack([np.stack([z, -v[:, 2], v[:, 1]], 1),
                             np.stack([v[:, 2], z, -v[:, 0]], 1),
                             np.stack([-v[:, 1], v[:, 0], z], 1)], 1)

        def exp_so3(w):
            theta = np.linalg.norm(w)
            if theta < 1e-12:
                return np.eye(3)
            k = w / theta
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * K @ K

        def scale_term(R, T):
            """The scale prior's residual and its derivative for camera b."""
            ca = -R[slot[a]].T @ T[slot[a]]
            Rb, tb = R[slot[b]], T[slot[b]]
            cb = -Rb.T @ tb
            gap = cb - ca
            length = max(np.linalg.norm(gap), 1e-12)
            unit = gap / length
            # C = -R^T t; a left rotation update dR = [w]x R moves C by -R^T [t]x w.
            d_rot = unit @ (-Rb.T @ np.array([[0, -tb[2], tb[1]], [tb[2], 0, -tb[0]], [-tb[1], tb[0], 0]]))
            d_t = unit @ (-Rb.T)
            return scale_weight * (length - target), scale_weight * np.concatenate([d_rot, d_t])

        def prior_cost(lens):
            if not (n_lens and not hold_focal and self.options.focal_prior):
                return 0.0
            prior = self.focal_prior / f0
            return 0.5 * ((lens[0] - prior) / (self.options.focal_prior * prior)) ** 2

        # Where each observation's derivatives land in the reduced system: the
        # same on every iteration, so worked out once.
        cols = np.concatenate([np.tile(np.arange(3), (len(obs), 1)),
                               (n_lens + 6 * np.maximum(o_c, 0))[:, None] + np.arange(6)], axis=1)
        valid = np.ones_like(cols, dtype=bool)
        valid[:, :3] = n_lens > 0
        valid[o_c < 0, 3:] = False
        pair = valid[:, :, None] & valid[:, None, :]
        u_index = (cols[:, :, None] * m + cols[:, None, :])[pair]
        g_index = cols[valid]
        w_rows = np.broadcast_to(cols[:, :, None], (len(obs), 9, 3))
        w_cols = np.broadcast_to((3 * o_t)[:, None, None] + np.arange(3)[None, None, :], (len(obs), 9, 3))
        w_keep = np.broadcast_to(valid[:, :, None], (len(obs), 9, 3))
        w_rows, w_cols = w_rows[w_keep], w_cols[w_keep]
        v_index = (9 * o_t)[:, None] + np.arange(9)

        Xc, x, y, r2, d, focal, e = project(R, T, P, lens)
        cost = cost_of(e)
        lam = 1e-3
        for _ in range(iterations):
            # Derivatives of each observation's pixel against the image-plane
            # point, then against camera coordinates.
            dd = 2 * (lens[1] + 2 * lens[2] * r2)
            J_uvxy = np.empty((len(obs), 2, 2))
            J_uvxy[:, 0, 0] = focal * (d + x * x * dd)
            J_uvxy[:, 0, 1] = focal * x * y * dd
            J_uvxy[:, 1, 0] = focal * y * x * dd
            J_uvxy[:, 1, 1] = focal * (d + y * y * dd)
            z = Xc[:, 2]
            J_xyX = np.zeros((len(obs), 2, 3))
            J_xyX[:, 0, 0] = 1 / z
            J_xyX[:, 0, 2] = -x / z
            J_xyX[:, 1, 1] = 1 / z
            J_xyX[:, 1, 2] = -y / z
            J_X = J_uvxy @ J_xyX                                            # d uv / d camera coords
            J_pt = J_X @ R[o_s]                                             # ... / d point
            J_rot = J_X @ -skew(Xc - T[o_s])                                # ... / d rotation (left)
            J_cam = np.concatenate([J_rot, J_X], axis=2)                    # 2 x 6
            J_lens = np.zeros((len(obs), 2, 3))
            if n_lens:
                J_lens[:, 0, 0] = f0 * x * d
                J_lens[:, 1, 0] = f0 * y * d
                J_lens[:, 0, 1] = focal * x * r2
                J_lens[:, 1, 1] = focal * y * r2
                J_lens[:, 0, 2] = focal * x * r2 * r2
                J_lens[:, 1, 2] = focal * y * r2 * r2

            norm = np.hypot(e[:, 0], e[:, 1])
            w = np.where(norm <= huber, 1.0, huber / np.maximum(norm, 1e-12))

            # The reduced side: lens then cameras. A_o is 2 x (3 + 6).
            A = np.concatenate([J_lens, J_cam], axis=2)
            A[o_c < 0, :, 3:] = 0.0
            Aw = np.swapaxes(A * w[:, None, None], 1, 2)                    # 9 x 2, weighted
            AtA = Aw @ A                                                    # 9 x 9 per observation
            Atb = (Aw @ e[:, :, None])[:, :, 0]
            U = np.bincount(u_index, AtA[pair], minlength=m * m).reshape(m, m)
            g = np.bincount(g_index, Atb[valid], minlength=m)

            Pw = np.swapaxes(J_pt * w[:, None, None], 1, 2)                 # 3 x 2, weighted
            V = np.bincount(v_index.ravel(), (Pw @ J_pt).reshape(-1), minlength=9 * n_pt).reshape(n_pt, 3, 3)
            gp = np.zeros((n_pt, 3))
            for k in range(3):
                gp[:, k] = np.bincount(o_t, (Pw @ e[:, :, None])[:, k, 0], minlength=n_pt)
            Y = Aw @ J_pt                                                   # 9 x 3 per observation

            # The scale prior on camera b.
            if b != fixed:
                residual, grad = scale_term(R, T)
                base = n_lens + 6 * cam_index[b]
                U[base:base + 6, base:base + 6] += np.outer(grad, grad)
                g[base:base + 6] += grad * residual
            if n_lens and hold_focal:
                U[0, 0] += 1e12
            elif n_lens and self.options.focal_prior:
                # lens[0] is the focal length as a ratio to f0; its prior is the
                # metadata focal length, also as a ratio to f0.
                prior = self.focal_prior / f0
                weight = 1.0 / (self.options.focal_prior * prior) ** 2
                U[0, 0] += weight
                g[0] += weight * (lens[0] - prior)

            # Points eliminated: S = U - W V^-1 W^T, from sparse products.
            W = sparse.csr_matrix((Y[w_keep], (w_rows, w_cols)), shape=(m, 3 * n_pt))

            while True:
                Vd = V + lam * np.einsum("nii->ni", V)[:, :, None] * np.eye(3) + 1e-9 * np.eye(3)
                Vinv = np.linalg.inv(Vd)
                Z = Y @ Vinv[o_t]                                     # each observation's 9 x 3 of W V^-1
                WVi = sparse.csr_matrix((Z[w_keep], (w_rows, w_cols)), shape=(m, 3 * n_pt))
                S = U + lam * np.diag(np.diag(U)) - (WVi @ W.T).toarray()
                rhs = -(g - WVi @ gp.ravel())
                try:
                    dr = cho_solve(cho_factor(S + 1e-9 * np.eye(m)), rhs)
                except Exception:
                    dr = np.linalg.lstsq(S, rhs, rcond=None)[0]
                dp = -(Vinv @ (gp + (W.T @ dr).reshape(n_pt, 3))[:, :, None])[:, :, 0]
                lens_new = lens.copy()
                if n_lens:
                    lens_new = lens + dr[:3]
                R_new, T_new = R.copy(), T.copy()
                for i in photos:
                    base = n_lens + 6 * cam_index[i]
                    R_new[slot[i]] = exp_so3(dr[base:base + 3]) @ R[slot[i]]
                    T_new[slot[i]] = T[slot[i]] + dr[base + 3:base + 6]
                P_new = P + dp
                trial = project(R_new, T_new, P_new, lens_new)
                new_cost = cost_of(trial[6]) + prior_cost(lens_new)
                if b != fixed:
                    new_cost += 0.5 * scale_term(R_new, T_new)[0] ** 2
                old_cost = cost + prior_cost(lens) + (0.5 * scale_term(R, T)[0] ** 2 if b != fixed else 0.0)
                if np.isfinite(new_cost) and new_cost < old_cost:
                    R, T, P, lens = R_new, T_new, P_new, lens_new
                    Xc, x, y, r2, d, focal, e = trial
                    improvement = (old_cost - new_cost) / max(old_cost, 1e-12)
                    cost = cost_of(e)
                    lam = max(lam / 3, 1e-9)
                    break
                lam *= 5
                if lam > 1e8:
                    improvement = 0.0
                    break
            if improvement < 1e-9:
                break

        self.k.focal = f0 * float(lens[0])
        if n_lens:
            self.k.k1, self.k.k2 = float(lens[1]), float(lens[2])
        for i in photos:
            self.poses[i] = (_rvec(R[slot[i]]), T[slot[i]].copy())
        for t in point_ids:
            self.points[t] = P[pt_index[t]].copy()

        # Every observation is judged again under the new lens and poses, the
        # dropped ones included: a point thrown out while the lens was still
        # wrong comes back once it is right.
        error = np.hypot(e[:, 0], e[:, 1])
        if refine_lens and not hold_focal and error.size:
            spread = 1.4826 * float(np.median(error)) * 1.18   # a robust sigma of the 2-D error
            self.threshold = max(self.options.max_reprojection_px, min(self.threshold, 5.0 * spread))
        self.rejudge(point_ids)

    def _all_observations(self, point_ids):
        """Every observation of these points by a placed photo, dropped or not, as arrays."""
        rows = [(n, i, c, r) for n, t in enumerate(point_ids) for i, c, r in self.tracks[t] if i in self.poses]
        if not rows:
            return None
        arr = np.array(rows)
        return arr[:, 0].astype(int), arr[:, 1].astype(int), arr[:, 2:]

    def _residuals(self, point_ids, o_p, o_i):
        """Pixel errors, and whether each point is in front of its camera."""
        P = np.array([self.points[t] for t in point_ids])
        photos = sorted(self.poses)
        slot = {i: n for n, i in enumerate(photos)}
        R = np.array([_rotation(self.poses[i][0]) for i in photos])
        T = np.array([np.asarray(self.poses[i][1], float) for i in photos])
        s_i = np.array([slot[i] for i in o_i])
        cam = np.einsum("nij,nj->ni", R[s_i], P[o_p]) + T[s_i]
        front = cam[:, 2] > 0
        cam[~front, 2] = 1.0
        return self.k.project(cam), front

    def rejudge(self, point_ids) -> None:
        """Every observation judged again, the dropped ones included, and points
        left with fewer than two, or seen from too narrow an angle, removed."""
        self.dropped = set()
        found = self._all_observations(point_ids)
        if found is None:
            return
        o_p, o_i, o_px = found
        px, front = self._residuals(point_ids, o_p, o_i)
        bad = ~front | (np.hypot(*(px - o_px).T) > self.threshold)
        for n, i in zip(o_p[bad], o_i[bad]):
            self.dropped.add((point_ids[n], int(i)))
        good = ~bad
        counts = np.bincount(o_p[good], minlength=len(point_ids))
        # The widest angle between rays at each point, from the extremes of its
        # ray directions: exact for two rays, and close enough for the
        # 1.5 degree test with more.
        P = np.array([self.points[t] for t in point_ids])
        centres = {i: self.centre(i) for i in self.poses}
        C = np.array([centres[i] for i in o_i[good]])
        rays = C - P[o_p[good]]
        rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-12)
        mean = np.zeros((len(point_ids), 3))
        np.add.at(mean, o_p[good], rays)
        mean /= np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12)
        spread = np.zeros(len(point_ids))
        cosine = np.clip((rays * mean[o_p[good]]).sum(axis=1), -1, 1)
        np.maximum.at(spread, o_p[good], np.degrees(np.arccos(cosine)))
        for n, t in enumerate(point_ids):
            if counts[n] < 2 or 2 * spread[n] < self.options.min_angle_deg:
                del self.points[t]

    def reprojection(self) -> tuple[float, int]:
        point_ids = sorted(self.points)
        found = self._all_observations(point_ids)
        if found is None:
            return 0.0, 0
        o_p, o_i, o_px = found
        keep = np.array([(point_ids[n], int(i)) not in self.dropped for n, i in zip(o_p, o_i)])
        px, _ = self._residuals(point_ids, o_p[keep], o_i[keep])
        errors = np.hypot(*(px - o_px[keep]).T)
        return (float(np.sqrt(np.mean(errors ** 2))) if errors.size else 0.0), int(errors.size)


def _starting_pair(pair_matches: dict, k: Intrinsics, options: SfmOptions):
    """The pair to start from: many matches, a real baseline, depth to triangulate."""
    import cv2

    best = None
    for (i, j), (pa, pb) in sorted(pair_matches.items(), key=lambda kv: -len(kv[1][0]))[:30]:
        if len(pa) < 50:
            continue
        # A pair explained as well by a homography is a pure rotation or a flat
        # surface seen square on: its relative pose is unreliable.
        H, h_mask = cv2.findHomography(pa, pb, cv2.RANSAC, 4.0)
        h_ratio = float(h_mask.mean()) if h_mask is not None else 0.0
        na, nb = k.normalise(pa), k.normalise(pb)
        E, mask = cv2.findEssentialMat(na, nb, np.eye(3), method=cv2.RANSAC, prob=0.999,
                                       threshold=1.5 / k.focal)
        if E is None or E.shape != (3, 3):
            continue
        _, R, t, pose_mask = cv2.recoverPose(E, na, nb, np.eye(3), mask=mask)
        inliers = pose_mask.ravel() > 0
        if inliers.sum() < 50:
            continue
        rays = [(np.eye(3), np.zeros(3)), (R, t.ravel())]
        centre_b = -R.T @ t.ravel()
        angles = []
        for xa, xb in zip(na[inliers][:300], nb[inliers][:300]):
            X = _triangulate([(rays[0][0], rays[0][1], xa), (rays[1][0], rays[1][1], xb)])
            if X is not None:
                angles.append(_max_angle(X, [np.zeros(3), centre_b]))
        median_angle = float(np.median(angles)) if angles else 0.0
        if median_angle < 2.0 or h_ratio > 0.85:
            continue
        score = inliers.sum() * min(median_angle, 10.0) * (1.0 - h_ratio)
        if best is None or score > best[0]:
            best = (score, i, j, R, t.ravel())
    return best


def reconstruct(
    sizes: Sequence[tuple],
    tracks: list,
    pair_matches: dict,
    focal_px: float,
    options: Optional[SfmOptions] = None,
    progress: Optional[Callable[[float, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> dict:
    """Cameras and points from tracks, in a frame of the model's own.

    ``sizes`` are each photo's (columns, rows); all photos share one camera.
    Returns the placed photos' poses (world to camera, OpenCV axes) and
    centres, the lens, the points with the tracks they came from, and a
    summary.
    """
    options = options or SfmOptions()
    say = progress or (lambda fraction, message: None)
    cancelled = should_cancel or (lambda: False)
    width, height = sizes[0]
    k = Intrinsics(focal=float(focal_px), cx=width / 2.0, cy=height / 2.0)
    model = _Model(tracks, k, options)
    n = len(sizes)

    start = _starting_pair(pair_matches, k, options)
    if start is None:
        raise ValueError("No pair of photos overlaps enough, from far enough apart, to start from.")
    _, a, b, R, t = start
    model.poses[a] = (np.zeros(3), np.zeros(3))
    model.poses[b] = (_rvec(R), t / max(np.linalg.norm(t), 1e-12))
    for track in range(len(tracks)):
        model.try_point(track)
    model.adjust(a, (a, b), refine_lens=options.refine_lens)
    say(0.1, f"Started from photos {a + 1} and {b + 1}")

    failed: set = set()
    last_adjusted = len(model.poses)
    while len(model.poses) < n:
        if cancelled():
            raise InterruptedError("Cancelled")
        waiting = [i for i in range(n) if i not in model.poses and i not in failed]
        if not waiting:
            break
        # The photo that sees the most points already placed goes next.
        seen = {i: sum(1 for t in model.by_photo.get(i, []) if t in model.points) for i in waiting}
        i = max(waiting, key=lambda j: seen[j])
        if not model.place(i):
            failed.add(i)
            continue
        failed.clear()          # more points now; earlier failures may succeed
        for track in model.by_photo.get(i, []):
            if track not in model.points:
                model.try_point(track)
        # Adjust while the model is small and fragile, then each time it has
        # grown by a fifth: the final adjustment refines everything anyway.
        if len(model.poses) <= 5 or len(model.poses) >= 1.2 * last_adjusted:
            model.adjust(a, (a, b), refine_lens=options.refine_lens, iterations=12)
            last_adjusted = len(model.poses)
        say(0.1 + 0.8 * len(model.poses) / n, f"Placed {len(model.poses)} of {n} photos")

    say(0.92, "Final adjustment")
    for _ in range(3):
        for track in range(len(tracks)):
            if track not in model.points:
                model.try_point(track)
        model.adjust(a, (a, b), refine_lens=options.refine_lens, iterations=60)
    model.threshold = options.max_reprojection_px
    model.adjust(a, (a, b), refine_lens=options.refine_lens, iterations=60)
    rms, used = model.reprojection()
    focal_source = "photos"

    if options.refine_lens and options.focal_known:
        import copy

        held = copy.deepcopy(model)
        held.k.focal = held.focal_prior
        held.options = copy.copy(options)
        held.options.focal_prior = 1e-7
        held.adjust(a, (a, b), refine_lens=True, iterations=60)
        held_rms, held_used = held.reprojection()
        if rms > held_rms * (1.0 - options.focal_evidence):
            model, rms, used, k = held, held_rms, held_used, held.k
            focal_source = "metadata"

    track_ids = sorted(model.points)
    lengths = [len([1 for i, _, _ in model.observations(t) if i in model.poses]) for t in track_ids]
    say(1.0, f"{len(model.poses)} of {n} photos placed, {len(track_ids):,} points")
    return {
        "poses": {i: {"rvec": model.poses[i][0].tolist(), "t": model.poses[i][1].tolist(),
                      "centre": model.centre(i).tolist()} for i in sorted(model.poses)},
        "unplaced": [i for i in range(n) if i not in model.poses],
        "lens": {"focal": k.focal, "cx": k.cx, "cy": k.cy, "k1": k.k1, "k2": k.k2},
        "points": np.array([model.points[t] for t in track_ids]).reshape(-1, 3),
        "pointTracks": track_ids,
        "observations": [[(i, c, r) for i, c, r in model.observations(t) if i in model.poses] for t in track_ids],
        "summary": {
            "photos": n,
            "placed": len(model.poses),
            "points": len(track_ids),
            "observations": used,
            "meanTrackLength": float(np.mean(lengths)) if lengths else 0.0,
            "reprojectionRmsPx": rms,
            "focalSource": focal_source,
            "startingPair": [a, b],
        },
    }
