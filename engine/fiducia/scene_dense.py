"""A dense coloured point cloud from photos placed by structure from motion.

The drone surface model (``drone.dense_surface``) is a height for every cell
of a map: right for ground seen from above, wrong for a statue, a rock face
or a building seen from every side, which have surfaces a height grid cannot
hold. Here each photo gets a depth for every pixel instead, and the depths
become points in three dimensions (as in Furukawa and Ponce's multi-view
stereo and Schonberger et al.'s depth-map fusion, 2016, with semi-global
matching for the depths):

1. Each photo is paired with the neighbours that see the same points from a
   useful angle: far enough apart to measure depth, close enough to look
   alike, and side by side rather than one behind the other.
2. Each pair is resampled so that matching pixels lie on the same row
   (epipolar rectification) and matched pixel by pixel by semi-global
   matching, within the range of depths its shared tie points span.
3. A photo's depths from its partners are kept where they agree.
4. Every depth is checked against the other photos that see the same place:
   a point is kept only where another photo's own depths put a surface at
   the same distance. Wind-blown leaves, reflections, sky and the matcher's
   occasional wrong guess rarely survive this.
5. The points are thinned to one per small cube, averaged in position and
   colour, so overlapping photos do not pile up duplicates.

Everything is in the reconstruction's own frame; ``scene`` scales and turns it.
"""

from __future__ import annotations

import math
import threading
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from .drone import _as_completed, _worker_count

__all__ = ["DETAIL_PX", "DenseOptions", "choose_partners", "dense_cloud"]

# Photos are matched with their longer side reduced to this many pixels.
DETAIL_PX = {"low": 1000, "medium": 1600, "high": 2400}


@dataclass
class DenseOptions:
    detail: str = "medium"
    partners: int = 2                # matched neighbours per photo
    min_shared: int = 20             # tie points two photos must share to be paired
    min_angle_deg: float = 4.0       # below this the depth is too uncertain
    max_angle_deg: float = 40.0      # above this the photos look too different to match
    best_angle_deg: float = 15.0
    min_sideways: float = 0.5        # sine of the angle between the baseline and each view
    max_disparities: int = 320       # a wider search is made at a reduced size...
    min_reduction: float = 0.3       # ... down to this, below which the pair is left out
    block: int = 5                   # matching window, pixels
    tolerance: float = 0.012         # depths agree within this share of the distance
    stride: int = 2                  # every n-th pixel each way becomes a point
    confirmations: int = 1           # other photos that must agree with a point
    check_neighbours: int = 6        # photos asked to confirm each photo's points


def _rotation(rvec) -> np.ndarray:
    import cv2

    return cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))[0]


class _Photo:
    """One placed photo at its working size: camera, pose and colours."""

    def __init__(self, index: int, info: dict, pose: dict, lens: dict, max_px: int):
        self.index = index
        self.path = info["path"]
        width, height = int(info["width"]), int(info["height"])
        self.s = min(1.0, max_px / max(width, height))
        self.size = (max(1, round(width * self.s)), max(1, round(height * self.s)))
        f = float(lens["focal"]) * self.s
        cx = (float(lens["cx"]) + 0.5) * self.s - 0.5
        cy = (float(lens["cy"]) + 0.5) * self.s - 0.5
        self.K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        self.D = np.array([float(lens.get("k1", 0.0)), float(lens.get("k2", 0.0)), 0.0, 0.0])
        self.R = _rotation(pose["rvec"])
        self.t = np.asarray(pose["t"], float)
        self.centre = -self.R.T @ self.t
        self.axis = self.R[2]

    def project(self, world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points to working pixels, and their depth along the view."""
        cam = world @ self.R.T + self.t
        z = cam[:, 2]
        safe = np.where(z > 1e-9, z, 1e-9)
        x, y = cam[:, 0] / safe, cam[:, 1] / safe
        r2 = x * x + y * y
        d = 1 + self.D[0] * r2 + self.D[1] * r2 * r2
        f, cx, cy = self.K[0, 0], self.K[0, 2], self.K[1, 2]
        return np.column_stack([f * x * d + cx, f * y * d + cy]), z

    def rays(self, cols: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """Undistorted image-plane coordinates (z = 1) of working pixels."""
        import cv2

        pts = np.column_stack([cols, rows]).astype(np.float64).reshape(-1, 1, 2)
        return cv2.undistortPoints(pts, self.K, self.D).reshape(-1, 2)


class _Images:
    """Working-size colour photos, the most recently used kept in memory."""

    def __init__(self, photos: dict, capacity: int = 12):
        self.photos = photos
        self.capacity = capacity
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.Lock()

    def __getitem__(self, i: int) -> np.ndarray:
        with self.lock:
            if i in self.cache:
                self.cache.move_to_end(i)
                return self.cache[i]
        from PIL import Image

        photo = self.photos[i]
        with Image.open(photo.path) as image:
            image.draft("RGB", photo.size)
            image = image.convert("RGB").resize(photo.size, Image.BILINEAR)
            data = np.ascontiguousarray(np.asarray(image))
        with self.lock:
            self.cache[i] = data
            while len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        return data


def choose_partners(photos: dict, points: np.ndarray, seen: dict, options: DenseOptions) -> dict:
    """For each photo, the neighbours to match it with, best first."""
    partners = {}
    ids = sorted(photos)
    for i in ids:
        scored = []
        for j in ids:
            if j == i:
                continue
            shared = seen[i] & seen[j]
            if len(shared) < options.min_shared:
                continue
            a, b = photos[i], photos[j]
            if a.axis @ b.axis < math.cos(math.radians(60)):
                continue
            base = b.centre - a.centre
            length = np.linalg.norm(base)
            if length < 1e-12:
                continue
            unit = base / length
            sideways = min(np.linalg.norm(np.cross(unit, a.axis)), np.linalg.norm(np.cross(unit, b.axis)))
            if sideways < options.min_sideways:
                continue
            sample = points[list(shared)[:200]]
            ra = a.centre - sample
            rb = b.centre - sample
            cosine = (ra * rb).sum(axis=1) / np.maximum(np.linalg.norm(ra, axis=1) * np.linalg.norm(rb, axis=1), 1e-12)
            angle = float(np.degrees(np.median(np.arccos(np.clip(cosine, -1, 1)))))
            if not options.min_angle_deg <= angle <= options.max_angle_deg:
                continue
            score = len(shared) * math.exp(-((angle - options.best_angle_deg) / options.best_angle_deg) ** 2)
            scored.append((score, j))
        scored.sort(reverse=True)
        # A few spares, for pairs that cannot be matched.
        partners[i] = [j for _, j in scored[:options.partners + 2]]
    return partners


def _pair_depth(a: _Photo, b: _Photo, images: _Images, points: np.ndarray, shared: list,
                options: DenseOptions) -> tuple[Optional[np.ndarray], str]:
    """Depths along photo a's view for each of its working pixels, from matching it with b."""
    import cv2

    R = b.R @ a.R.T
    T = b.t - R @ a.t
    size = a.size
    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(a.K, a.D, b.K, b.D, size, R, T.reshape(3, 1), flags=0, alpha=0)
    vertical = abs(P2[1, 3]) > abs(P2[0, 3])
    axis = 1 if vertical else 0

    # The search: the disparities of the tie points both photos share, widened.
    world = points[shared]
    cam_a = world @ a.R.T + a.t
    rect = cam_a @ R1.T
    front = rect[:, 2] > 0
    if front.sum() < 8:
        return None, "too few shared points in front of both"
    h = np.column_stack([rect[front], np.ones(int(front.sum()))])
    left = h @ P1.T
    right = h @ P2.T
    disparity = left[:, axis] / left[:, 2] - right[:, axis] / right[:, 2]
    lo, hi = np.percentile(disparity, [2, 98])
    span = hi - lo
    lo, hi = lo - 0.25 * span - 4, hi + 0.25 * span + 4
    # A pair seen nearly along its baseline rectifies into a stretched view
    # with disparities beyond any image: nothing to match there.
    if not (np.isfinite(lo) and np.isfinite(hi)) or max(abs(lo), abs(hi)) > 4 * max(size):
        return None, "the photos look along the line between them"
    # Ground running up to the photographer's feet spans many disparities:
    # such a pair is matched smaller, where the same depths span fewer pixels.
    reduction = min(1.0, options.max_disparities / max(hi - lo, 1.0))
    if reduction < options.min_reduction:
        return None, f"depth range too wide to search ({int(hi - lo)} pixels)"
    min_disp = int(math.floor(lo * reduction))
    count = max(32, int(math.ceil((hi * reduction - min_disp) / 16.0)) * 16)

    map_ax, map_ay = cv2.initUndistortRectifyMap(a.K, a.D, R1, P1, size, cv2.CV_32FC1)
    map_bx, map_by = cv2.initUndistortRectifyMap(b.K, b.D, R2, P2, size, cv2.CV_32FC1)
    image_a, image_b = images[a.index], images[b.index]
    rect_a = cv2.remap(image_a, map_ax, map_ay, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    rect_b = cv2.remap(image_b, map_bx, map_by, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    inside = cv2.remap(np.ones(image_a.shape[:2], np.uint8), map_ax, map_ay, cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if vertical:
        # Matching runs along rows: a pair one above the other is turned on its side.
        rect_a, rect_b, inside = (np.ascontiguousarray(np.swapaxes(m, 0, 1)) for m in (rect_a, rect_b, inside))
    full_shape = rect_a.shape[:2]
    if reduction < 1.0:
        small = (max(1, round(full_shape[1] * reduction)), max(1, round(full_shape[0] * reduction)))
        rect_a = cv2.resize(rect_a, small, interpolation=cv2.INTER_AREA)
        rect_b = cv2.resize(rect_b, small, interpolation=cv2.INTER_AREA)
        inside = cv2.resize(inside, small, interpolation=cv2.INTER_NEAREST)
    block = options.block
    matcher = cv2.StereoSGBM_create(
        minDisparity=min_disp, numDisparities=count, blockSize=block,
        P1=8 * 3 * block * block, P2=32 * 3 * block * block, disp12MaxDiff=1,
        uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    found = matcher.compute(rect_a, rect_b).astype(np.float32) / 16.0
    found[(found < min_disp) | (inside == 0)] = -1e6
    if reduction < 1.0:
        # Back to full size, in full-size pixels.
        found = cv2.resize(found, (full_shape[1], full_shape[0]), interpolation=cv2.INTER_NEAREST)
        found = np.where(found > -1e5, found / reduction, -1e6).astype(np.float32)
        min_disp = min_disp / reduction

    # Back to photo a's own pixels: where each lands in the rectified view.
    width, height = size
    cols, rows = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    grid = np.column_stack([cols.ravel(), rows.ravel()]).reshape(-1, 1, 2)
    where = cv2.undistortPoints(grid, a.K, a.D, R=R1, P=P1).reshape(height, width, 2).astype(np.float32)
    sx, sy = (where[..., 1], where[..., 0]) if vertical else (where[..., 0], where[..., 1])
    d = cv2.remap(found, sx, sy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=-1e6)

    f = P1[0, 0]
    base = P2[axis, 3] / P2[axis, axis]
    offset = P1[axis, 2] - P2[axis, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = -base * f / (d - offset)
    X = (where[..., 0] - P1[0, 2]) * Z / f
    Y = (where[..., 1] - P1[1, 2]) * Z / f
    depth = R1[0, 2] * X + R1[1, 2] * Y + R1[2, 2] * Z
    valid = (d > min_disp - 0.5) & np.isfinite(depth) & (Z > 0) & (depth > 0)
    return np.where(valid, depth, np.nan).astype(np.float32), ""


def dense_cloud(
    infos: Sequence[dict],
    poses: dict,
    lenses: dict,
    points: np.ndarray,
    observations: Sequence[list],
    work_dir: str,
    options: Optional[DenseOptions] = None,
    progress: Optional[Callable[[float, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> dict:
    """Dense points and colours, in the reconstruction's frame.

    ``infos`` are every photo's ``path``, ``width`` and ``height``; ``poses``
    and ``lenses`` (focal length, principal point, k1, k2 in full-size
    pixels) are by photo index, for the placed photos; ``points`` and
    ``observations`` are the sparse reconstruction's.
    """
    from concurrent.futures import ThreadPoolExecutor

    options = options or DenseOptions()
    say = progress or (lambda fraction, message: None)
    cancelled = should_cancel or (lambda: False)
    max_px = DETAIL_PX.get(options.detail, DETAIL_PX["medium"])
    photos = {i: _Photo(i, infos[i], poses[i], lenses[i], max_px) for i in poses}
    seen: dict = {i: set() for i in photos}
    for n, obs in enumerate(observations):
        for i, _, _ in obs:
            if i in seen:
                seen[i].add(n)
    points = np.asarray(points, float)

    partners = choose_partners(photos, points, seen, options)
    references = [i for i in sorted(photos) if partners[i]]
    if not references:
        raise ValueError("No two photos see the same place from a useful angle. Dense matching "
                         "needs photos taken a step or two apart, all round the subject.")
    images = _Images(photos)
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    # The distances at which each photo sees its tie points: depths far
    # outside them are sky, or a wrong match.
    bounds = {}
    for i in photos:
        ids = list(seen[i])
        if not ids:
            continue
        _, z = photos[i].project(points[ids])
        z = z[z > 0]
        if z.size:
            bounds[i] = (0.5 * float(np.percentile(z, 2)), 2.0 * float(np.percentile(z, 98)))

    failures: list = []
    lock = threading.Lock()

    def reference(i):
        """Photo i's depths from its partners, kept where they agree."""
        maps = []
        for j in partners[i]:
            if len(maps) >= options.partners:
                break
            if cancelled():
                raise InterruptedError("Cancelled")
            shared = list(seen[i] & seen[j])
            try:
                depth, reason = _pair_depth(photos[i], photos[j], images, points, shared, options)
            except Exception as exc:  # noqa: BLE001 -- one pair that cannot be matched leaves the rest
                depth, reason = None, str(exc).splitlines()[-1][:160]
            if depth is None:
                with lock:
                    failures.append(f"photos {i + 1} and {j + 1}: {reason}")
                continue
            maps.append(depth)
        if not maps:
            return None
        stack = np.stack(maps)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            middle = np.nanmedian(stack, axis=0)
            agree = np.abs(stack - middle) <= options.tolerance * middle
            count = agree.sum(axis=0)
            fused = np.where(agree, stack, 0.0).sum(axis=0) / np.maximum(count, 1)
        fused[count == 0] = np.nan
        if i in bounds:
            near, far = bounds[i]
            fused[(fused < near) | (fused > far)] = np.nan
        path = Path(work_dir) / f"depth_{i}.npy"
        np.save(path, fused.astype(np.float32))
        return str(path), fused[::2, ::2].copy()

    depth_paths: dict = {}
    small: dict = {}
    workers = max(1, min(3, _worker_count()))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(reference, i): i for i in references}
        for done, future in enumerate(_as_completed(futures, cancelled)):
            result = future.result()
            if result is not None:
                depth_paths[futures[future]], small[futures[future]] = result
            say(0.72 * (done + 1) / len(references), f"Depths for {done + 1} of {len(references)} photos")
    if not depth_paths:
        raise ValueError("No pair of photos could be matched densely. " + "; ".join(failures[:3]))

    # One point per cube about the size of a few matched pixels on the subject.
    footprints = []
    for i, coarse in small.items():
        finite = coarse[np.isfinite(coarse)]
        if finite.size:
            footprints.append(float(np.median(finite)) / photos[i].K[0, 0])
    voxel = options.stride * float(np.median(footprints))
    origin = points.mean(axis=0) if len(points) else np.zeros(3)

    # Which photos are asked to confirm each photo's points: those sharing the most.
    checkers = {}
    for i in depth_paths:
        others = sorted((len(seen[i] & seen[k]), k) for k in depth_paths if k != i)
        checkers[i] = [k for count, k in reversed(others) if count >= options.min_shared][:options.check_neighbours]

    keys_all, sum_p, sum_c, counts = None, None, None, None
    pending: list = []

    def reduce(batch):
        nonlocal keys_all, sum_p, sum_c, counts
        keys = np.concatenate([b[0] for b in batch] + ([keys_all] if keys_all is not None else []))
        p = np.concatenate([b[1] for b in batch] + ([sum_p] if sum_p is not None else []))
        c = np.concatenate([b[2] for b in batch] + ([sum_c] if sum_c is not None else []))
        n = np.concatenate([np.ones(len(b[0])) for b in batch] + ([counts] if counts is not None else []))
        unique, inverse = np.unique(keys, return_inverse=True)
        keys_all = unique
        sum_p = np.column_stack([np.bincount(inverse, p[:, k], minlength=len(unique)) for k in range(3)])
        sum_c = np.column_stack([np.bincount(inverse, c[:, k], minlength=len(unique)) for k in range(3)])
        counts = np.bincount(inverse, n, minlength=len(unique))

    emitted = 0
    confirmed_paths: dict = {}
    for done, i in enumerate(sorted(depth_paths)):
        if cancelled():
            raise InterruptedError("Cancelled")
        photo = photos[i]
        depth = np.load(depth_paths[i])
        Path(depth_paths[i]).unlink(missing_ok=True)
        rows, cols = np.mgrid[0:depth.shape[0]:options.stride, 0:depth.shape[1]:options.stride]
        # The depths that pass, on the same grid, kept for building a mesh.
        kept = np.full(rows.shape, np.nan, np.float32)
        z = depth[rows, cols]
        keep = np.isfinite(z)
        grid_rows, grid_cols = np.nonzero(keep)
        rows, cols, z = rows[keep], cols[keep], z[keep]
        if not z.size:
            continue
        xy = photo.rays(cols, rows)
        cam = np.column_stack([xy * z[:, None], z])
        world = (cam - photo.t) @ photo.R
        confirmed = np.zeros(len(z), int)
        for k in checkers[i]:
            other = photos[k]
            px, zk = other.project(world)
            u = np.rint(px[:, 0] / 2).astype(int)
            v = np.rint(px[:, 1] / 2).astype(int)
            coarse = small[k]
            inside = (zk > 0) & (u >= 0) & (v >= 0) & (u < coarse.shape[1]) & (v < coarse.shape[0])
            theirs = np.full(len(z), np.nan, np.float32)
            theirs[inside] = coarse[v[inside], u[inside]]
            with np.errstate(invalid="ignore"):
                confirmed += (np.abs(theirs - zk) <= options.tolerance * zk)
        good = confirmed >= options.confirmations
        kept[grid_rows[good], grid_cols[good]] = z[good]
        confirmed_paths[i] = str(Path(work_dir) / f"confirmed_{i}.npy")
        np.save(confirmed_paths[i], kept)
        if good.any():
            world = world[good]
            colours = images[i][rows[good], cols[good]].astype(np.float64)
            q = np.floor((world - origin) / voxel).astype(np.int64) + (1 << 20)
            q = np.clip(q, 0, (1 << 21) - 1)
            key = (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]
            pending.append((key, world, colours))
            emitted += int(good.sum())
        if sum(len(b[0]) for b in pending) > 4_000_000:
            reduce(pending)
            pending = []
        say(0.72 + 0.26 * (done + 1) / len(depth_paths), f"Checked the depths of {done + 1} of {len(depth_paths)} photos")
    if pending:
        reduce(pending)
    if keys_all is None:
        raise ValueError("No depth was confirmed by a second photo. The photos may overlap too "
                         "little, or the surface may have too little texture to match.")

    dense = sum_p / counts[:, None]
    colours = np.clip(np.rint(sum_c / counts[:, None]), 0, 255).astype(np.uint8)
    say(1.0, f"{len(dense):,} points")
    return {
        "points": dense,
        "colours": colours,
        "summary": {
            "photos": len(photos),
            "references": len(depth_paths),
            "pairs": sum(min(len(partners[i]), options.partners) for i in references),
            "pairsFailed": failures[:12],
            "pointsMatched": emitted,
            "points": int(len(dense)),
            "voxel": voxel,
            "detail": options.detail,
            "workingPx": max_px,
        },
        # Each photo's checked depths, every stride-th working pixel: what a mesh is built from.
        "depthMaps": confirmed_paths,
        "depthStride": options.stride,
    }
