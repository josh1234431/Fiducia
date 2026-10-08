"""Drone photos: from GPS-tagged frames to a block the bundle can solve.

A drone flight comes with no calibration certificate and no fiducials, but
every photo carries its own approximate position (GNSS), its heading and the
gimbal's tilt, and enough of the camera (focal length, image size, often a
35 mm equivalent) to start from. That is all structure from motion needs when
the photos look down and the positions are known:

* the camera is taken from the photos' metadata and refined afterwards by
  self-calibration in the bundle;
* each photo starts at its GNSS position with the attitude its gimbal
  reports, and the positions are carried into the adjustment as observations
  with an honest precision, which also georeferences the block;
* photos are matched only with their GNSS neighbours, so a flight of hundreds
  of photos costs a few matches per photo rather than every pair;
* matches are filtered by the two-view epipolar geometry and joined into
  tracks across all the photos that see a feature, so one ground point is one
  multi-ray tie point rather than many pairs.

Everything after this -- weighting, robust estimation, data snooping,
precision and accuracy -- is the ordinary block adjustment.
"""

from __future__ import annotations

import math
from pathlib import Path
import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from .tiepoints import TiePoint, _load_working_image, _to_full

__all__ = ["PhotoMeta", "read_metadata", "pixel_pitch_mm", "camera_from_metadata",
           "suggest_crs", "attitude_from_gimbal", "neighbour_pairs", "DroneMatchOptions",
           "collect_tracks", "heading_and_height", "surface_from_points", "stereo_pairs",
           "fuse_dems", "dense_surface"]

FULL_FRAME_DIAGONAL_MM = math.hypot(36.0, 24.0)   # 43.27 mm, the 35 mm equivalent's reference


# -- metadata ------------------------------------------------------------------


@dataclass
class PhotoMeta:
    path: str
    width: int = 0
    height: int = 0
    make: str = ""
    model: str = ""
    focal_mm: Optional[float] = None
    focal_35mm: Optional[float] = None
    # Pixels per unit on the focal plane, when the camera records it.
    focal_plane_x_res: Optional[float] = None
    focal_plane_unit_mm: Optional[float] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    relative_altitude: Optional[float] = None
    gimbal_yaw: Optional[float] = None
    gimbal_pitch: Optional[float] = None
    gimbal_roll: Optional[float] = None
    flight_yaw: Optional[float] = None
    taken: str = ""
    # DJI's factory calibration, in pixels, when the photo carries it.
    calibrated_focal_px: Optional[float] = None
    calibrated_centre_px: Optional[tuple] = None
    notes: list = field(default_factory=list)

    @property
    def has_position(self) -> bool:
        return self.latitude is not None and self.longitude is not None and self.altitude is not None

    @property
    def camera_key(self) -> tuple:
        return (self.make, self.model, self.width, self.height,
                round(self.focal_mm or 0.0, 2))

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _rational(value) -> Optional[float]:
    try:
        if isinstance(value, tuple) and len(value) == 2:
            return float(value[0]) / float(value[1]) if value[1] else None
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _dms(value, ref) -> Optional[float]:
    try:
        degrees = sum(_rational(part) / div for part, div in zip(value, (1.0, 60.0, 3600.0)))
    except TypeError:
        return None
    if ref in ("S", "W", b"S", b"W"):
        degrees = -degrees
    return degrees


_XMP_NUMBER = re.compile(r'(?:drone-dji|Camera|crs):(\w+)\s*=\s*"([+-]?[0-9.eE+-]+)"')
_XMP_ELEMENT = re.compile(r'<(?:drone-dji|Camera):(\w+)>([+-]?[0-9.eE+-]+)</')


def _xmp_values(path: str) -> dict:
    """Numbers from the photo's XMP packet (DJI and similar write flight data there)."""
    with open(path, "rb") as handle:
        head = handle.read(512 * 1024)
    start = head.find(b"<x:xmpmeta")
    end = head.find(b"</x:xmpmeta>", start)
    if start < 0 or end < 0:
        return {}
    text = head[start:end].decode("utf-8", "replace")
    values = {}
    for name, number in _XMP_NUMBER.findall(text) + _XMP_ELEMENT.findall(text):
        try:
            values[name] = float(number)
        except ValueError:
            pass
    return values


def read_metadata(path: str) -> PhotoMeta:
    """What a photo says about itself: camera, position and attitude."""
    from PIL import ExifTags, Image

    meta = PhotoMeta(path=path)
    with Image.open(path) as image:
        meta.width, meta.height = image.size
        exif = image.getexif()
    tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
    meta.make = str(tags.get("Make", "")).strip("\x00 ").strip()
    meta.model = str(tags.get("Model", "")).strip("\x00 ").strip()

    detail = {ExifTags.TAGS.get(k, k): v for k, v in exif.get_ifd(0x8769).items()}
    meta.focal_mm = _rational(detail.get("FocalLength"))
    meta.focal_35mm = _rational(detail.get("FocalLengthIn35mmFilm")) or None
    meta.focal_plane_x_res = _rational(detail.get("FocalPlaneXResolution"))
    unit = detail.get("FocalPlaneResolutionUnit")
    meta.focal_plane_unit_mm = {2: 25.4, 3: 10.0, 4: 1.0}.get(int(unit)) if unit else None
    meta.taken = str(detail.get("DateTimeOriginal") or tags.get("DateTime") or "")

    gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in exif.get_ifd(0x8825).items()}
    if "GPSLatitude" in gps and "GPSLongitude" in gps:
        meta.latitude = _dms(gps["GPSLatitude"], gps.get("GPSLatitudeRef"))
        meta.longitude = _dms(gps["GPSLongitude"], gps.get("GPSLongitudeRef"))
    if "GPSAltitude" in gps:
        altitude = _rational(gps["GPSAltitude"])
        if altitude is not None and gps.get("GPSAltitudeRef") in (1, b"\x01"):
            altitude = -altitude
        meta.altitude = altitude

    xmp = _xmp_values(path)
    meta.relative_altitude = xmp.get("RelativeAltitude")
    if "AbsoluteAltitude" in xmp and meta.altitude is None:
        meta.altitude = xmp["AbsoluteAltitude"]
    meta.gimbal_yaw = xmp.get("GimbalYawDegree")
    meta.gimbal_pitch = xmp.get("GimbalPitchDegree")
    meta.gimbal_roll = xmp.get("GimbalRollDegree")
    meta.flight_yaw = xmp.get("FlightYawDegree")
    if "CalibratedFocalLength" in xmp:
        meta.calibrated_focal_px = xmp["CalibratedFocalLength"]
    if "CalibratedOpticalCenterX" in xmp and "CalibratedOpticalCenterY" in xmp:
        meta.calibrated_centre_px = (xmp["CalibratedOpticalCenterX"], xmp["CalibratedOpticalCenterY"])
    return meta


def pixel_pitch_mm(meta: PhotoMeta) -> tuple[Optional[float], str]:
    """The size of one pixel on the sensor, and where the figure came from."""
    if meta.focal_plane_x_res and meta.focal_plane_unit_mm:
        pitch = meta.focal_plane_unit_mm / meta.focal_plane_x_res
        if 0.0005 < pitch < 0.05:
            return pitch, "focal-plane resolution in the photo"
    if meta.focal_mm and meta.focal_35mm and meta.width and meta.height:
        diagonal = FULL_FRAME_DIAGONAL_MM * meta.focal_mm / meta.focal_35mm
        long_side, short_side = max(meta.width, meta.height), min(meta.width, meta.height)
        if short_side / long_side < 0.7:
            # A widescreen photo is a crop of a 4:3 sensor (drones, phones), and
            # the 35 mm equivalent describes the whole sensor: measured against
            # the crop's own diagonal the pixels come out ~9% too big.
            return (diagonal / math.hypot(long_side, 0.75 * long_side),
                    "35 mm equivalent focal length (16:9 crop of a 4:3 sensor)")
        return diagonal / math.hypot(meta.width, meta.height), "35 mm equivalent focal length"
    if meta.calibrated_focal_px and meta.focal_mm:
        return meta.focal_mm / meta.calibrated_focal_px, "factory calibration in the photo"
    return None, ""


def camera_from_metadata(metas: Sequence[PhotoMeta]) -> dict:
    """A digital frame camera for the project, from the photos themselves.

    It is a starting point: the bundle refines the focal length, principal
    point and distortion by self-calibration.
    """
    groups: dict[tuple, list[PhotoMeta]] = {}
    for meta in metas:
        groups.setdefault(meta.camera_key, []).append(meta)
    key, members = max(groups.items(), key=lambda kv: len(kv[1]))
    sample = members[0]
    pitch, source = pixel_pitch_mm(sample)
    if pitch is None:
        raise ValueError(
            f"The {sample.make} {sample.model} photos don't say how big the sensor is "
            "(no focal-plane resolution or 35 mm equivalent). Enter the pixel size in "
            "the Camera step.")
    if not sample.focal_mm:
        raise ValueError("The photos don't record a focal length. Enter it in the Camera step.")
    focal = sample.focal_mm
    note = f"Pixel size from the {source}"
    if sample.calibrated_focal_px:
        focal = sample.calibrated_focal_px * pitch
        note += "; focal length from the factory calibration in the photos"
    # The principal point starts at the image centre; self-calibration estimates
    # its offset (x0, y0) along with the focal length and distortion.
    name = " ".join(part for part in (sample.make, sample.model) if part) or "Drone camera"
    return {
        "camera": {
            "kind": "digital",
            "name": name,
            "focalMm": round(float(focal), 4),
            "pixelPitchMm": round(float(pitch), 6),
            "columns": int(sample.width),
            "rows": int(sample.height),
        },
        "photos": len(members),
        "otherCameras": len(groups) - 1,
        "note": note,
    }


def suggest_crs(latitude: float, longitude: float) -> str:
    """The UTM zone a block sits in, on WGS 84."""
    zone = int((longitude + 180.0) // 6.0) % 60 + 1
    return f"EPSG:{(32600 if latitude >= 0 else 32700) + zone}"


def attitude_from_gimbal(yaw_deg: float, pitch_deg: float = -90.0) -> tuple[float, float, float]:
    """Omega, phi, kappa (radians) of a camera with the given gimbal heading and tilt.

    Gimbal yaw is the compass direction the top of the image faces, clockwise
    from north; pitch is -90 straight down. Roll is left out: gimbals hold it
    near zero, and the adjustment recovers what remains.
    """
    yaw = math.radians(yaw_deg if yaw_deg is not None else 0.0)
    pitch = math.radians(pitch_deg if pitch_deg is not None else -90.0)
    view = np.array([math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch), math.sin(pitch)])
    right = np.array([math.cos(yaw), -math.sin(yaw), 0.0])
    back = -view
    up = np.cross(back, right)
    # Rows: the camera's x (right), y (up) and z (backwards) axes in ground terms,
    # i.e. the ground-to-camera rotation Rz(kappa) Ry(phi) Rx(omega).
    m = np.vstack([right, up / np.linalg.norm(up), back])
    phi = math.asin(max(-1.0, min(1.0, m[2, 0])))
    omega = math.atan2(-m[2, 1], m[2, 2])
    kappa = math.atan2(-m[1, 0], m[0, 0])
    return omega, phi, kappa


# -- which photos to match -------------------------------------------------------


def neighbour_pairs(positions: np.ndarray, footprint_m: float, neighbours: int = 10) -> list[tuple[int, int]]:
    """Pairs of photos close enough to overlap, nearest first.

    Each photo is paired with up to ``neighbours`` others whose centres are
    within one and a half footprints; a flight of n photos gives about n x k
    pairs rather than n squared.
    """
    from scipy.spatial import cKDTree

    positions = np.asarray(positions, float)[:, :2]
    if len(positions) < 2:
        return []
    tree = cKDTree(positions)
    radius = 1.5 * footprint_m
    pairs = set()
    count = min(neighbours + 1, len(positions))
    distances, indices = tree.query(positions, k=count)
    for i in range(len(positions)):
        for d, j in zip(np.atleast_1d(distances[i]), np.atleast_1d(indices[i])):
            j = int(j)
            if j != i and d <= radius:
                pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)


# -- features, matches and tracks --------------------------------------------------


@dataclass
class DroneMatchOptions:
    working_max_px: int = 2400      # photos are matched at this size
    features: int = 8000            # strongest SIFT features kept per photo
    ratio: float = 0.8              # Lowe's ratio test
    ransac_px: float = 1.5          # epipolar inlier threshold, working pixels
    min_inliers: int = 30           # weaker pairs are left out
    target_per_image: int = 500     # tie points kept per photo after thinning
    grid: tuple = (8, 6)            # cells used to spread them across the photo
    neighbours: int = 10
    # Depths kept, as fractions of the pair's median: trees can reach half-way
    # up to a low drone, but nothing sits above it or far below the ground.
    depth_band: tuple = (0.45, 2.2)


def _features(path: str, options: DroneMatchOptions):
    """SIFT features spread over the whole photo.

    Kept by strength alone, they all come from the busiest texture: on a drone
    photo that is tree canopy, which moves in the wind and repeats itself, while
    the road and grass beside it, which match well, get none. So every cell of
    a grid keeps its own strongest share.
    """
    import cv2

    image, scale = _load_working_image(path, options.working_max_px)
    sift = cv2.SIFT_create()
    keypoints = sift.detect(image, None)
    height, width = image.shape[:2]
    cols, rows = options.grid
    share = max(1, options.features // (cols * rows))
    cells: dict = {}
    for kp in keypoints:
        x, y = kp.pt
        cell = (min(cols - 1, int(x / width * cols)), min(rows - 1, int(y / height * rows)))
        cells.setdefault(cell, []).append(kp)
    kept = []
    for members in cells.values():
        members.sort(key=lambda kp: -kp.response)
        kept.extend(members[:share])
    keypoints, descriptors = sift.compute(image, kept)
    # SIFT descriptors are whole numbers 0-255: kept as bytes they take a
    # quarter of the memory, which matters with hundreds of photos open.
    if descriptors is not None:
        descriptors = np.clip(np.rint(descriptors), 0, 255).astype(np.uint8)
    points = np.array([kp.pt for kp in keypoints], dtype=np.float64).reshape(-1, 2)
    strength = np.array([kp.response for kp in keypoints], dtype=np.float64)
    return {"points": points, "descriptors": descriptors, "strength": strength,
            "scale": scale, "size": image.shape[::-1]}


def _match(a: dict, b: dict, focal_px: float, options: DroneMatchOptions):
    """Indices of features in a and b that are the same ground, geometry-checked."""
    import cv2

    if a["descriptors"] is None or b["descriptors"] is None:
        return np.zeros((0, 2), int)
    if len(a["descriptors"]) < 8 or len(b["descriptors"]) < 8:
        return np.zeros((0, 2), int)
    matcher = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 64})
    da, db = a["descriptors"].astype(np.float32), b["descriptors"].astype(np.float32)
    forward = matcher.knnMatch(da, db, k=2)
    backward = matcher.knnMatch(db, da, k=1)
    best_back = {m[0].queryIdx: m[0].trainIdx for m in backward if m}
    pairs = []
    for candidates in forward:
        if len(candidates) < 2:
            continue
        first, second = candidates
        if first.distance < options.ratio * second.distance and best_back.get(first.trainIdx) == first.queryIdx:
            pairs.append((first.queryIdx, first.trainIdx))
    if len(pairs) < options.min_inliers:
        return np.zeros((0, 2), int)
    pairs = np.array(pairs, int)
    pa, pb = a["points"][pairs[:, 0]], b["points"][pairs[:, 1]]
    # The essential matrix, with the camera's own focal length, is stricter than
    # a fundamental matrix: it knows the photos come from the same calibrated lens.
    width, height = a["size"]
    k = np.array([[focal_px, 0, width / 2.0], [0, focal_px, height / 2.0], [0, 0, 1]])
    essential, mask = cv2.findEssentialMat(pa, pb, k, method=cv2.RANSAC, prob=0.999,
                                           threshold=options.ransac_px)
    if essential is None or mask is None:
        return np.zeros((0, 2), int)
    keep = mask.ravel().astype(bool)
    if keep.sum() < options.min_inliers:
        return np.zeros((0, 2), int)
    # The epipolar test only asks that a match lie on its line, not where along
    # it. Along a strip those lines run with the flight, so repeated texture
    # (mown grass, canopy) passes with the wrong partner. Triangulated with
    # the pair's relative pose, such a match lands behind a camera or far off
    # the depth of the rest of the scene.
    pa_in, pb_in = pa[keep], pb[keep]
    _, _, _, front, world = cv2.recoverPose(essential, pa_in, pb_in, k,
                                            distanceThresh=1e6, mask=None)
    depth = world[2] / np.where(np.abs(world[3]) < 1e-12, 1e-12, world[3])
    good = (front.ravel() > 0) & np.isfinite(depth) & (depth > 0)
    if good.sum() >= options.min_inliers:
        median = float(np.median(depth[good]))
        good &= (depth > median * options.depth_band[0]) & (depth < median * options.depth_band[1])
    inliers = pairs[keep][good]
    return inliers if len(inliers) >= options.min_inliers else np.zeros((0, 2), int)


def _worker_count() -> int:
    """Threads for matching: the cores, less one for the interface."""
    import os

    return max(1, min(8, (os.cpu_count() or 2) - 1))


def _as_completed(futures, cancelled):
    """Futures as they finish, stopping the rest if the job is cancelled."""
    from concurrent.futures import as_completed

    for future in as_completed(futures):
        if cancelled():
            for other in futures:
                other.cancel()
            raise InterruptedError("Cancelled")
        yield future


class _Union:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        parent = self.parent
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def join(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def collect_tracks(
    photos: Sequence[dict],
    focal_mm: float,
    pitch_mm: float,
    footprint_m: float,
    options: Optional[DroneMatchOptions] = None,
    progress: Optional[Callable[[float, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> tuple[list[TiePoint], dict]:
    """Tie points across a drone block.

    ``photos`` carry ``id``, ``path`` and ``position`` (ground x, y). Returns
    the tie points (full-resolution pixel coordinates, each a feature followed
    across every photo that sees it) and a summary.
    """
    options = options or DroneMatchOptions()
    say = progress or (lambda fraction, message: None)
    cancelled = should_cancel or (lambda: False)
    n = len(photos)

    # Photos and pairs are independent, and OpenCV lets go of Python while
    # it works, so threads keep every core busy.
    from concurrent.futures import ThreadPoolExecutor

    workers = _worker_count()
    features: list = [None] * n
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_features, photo["path"], options): i for i, photo in enumerate(photos)}
        for done, future in enumerate(_as_completed(futures, cancelled)):
            features[futures[future]] = future.result()
            say(0.35 * (done + 1) / max(n, 1), f"Found features in {done + 1} of {n} photos")

    positions = np.array([photo["position"] for photo in photos], float)
    pairs = neighbour_pairs(positions, footprint_m, options.neighbours)

    def match(pair):
        i, j = pair
        scale = float(np.mean(features[i]["scale"]))
        return _match(features[i], features[j], focal_mm / (pitch_mm * scale), options)

    union = _Union()
    matched_pairs = 0
    results: list = [None] * len(pairs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(match, pair): k for k, pair in enumerate(pairs)}
        for done, future in enumerate(_as_completed(futures, cancelled)):
            results[futures[future]] = future.result()
            say(0.35 + 0.5 * (done + 1) / max(len(pairs), 1),
                f"Matched {done + 1} of {len(pairs)} pairs")
    # Joined in pair order, so the tracks do not depend on which thread
    # happened to finish first.
    for (i, j), inliers in zip(pairs, results):
        if len(inliers):
            matched_pairs += 1
        for fa, fb in inliers:
            union.join((i, int(fa)), (j, int(fb)))

    say(0.88, "Joining matches into tracks")
    groups: dict = {}
    for node in list(union.parent):
        groups.setdefault(union.find(node), []).append(node)

    tracks = []
    for nodes in groups.values():
        images = [node[0] for node in nodes]
        # Two features in one photo joined into one track means a false match
        # somewhere in the chain; the whole track is unreliable.
        if len(set(images)) != len(images) or len(nodes) < 2:
            continue
        strength = float(np.mean([features[i]["strength"][f] for i, f in nodes]))
        tracks.append((nodes, strength))

    # Spread the kept points over every photo: longest tracks first, a cell of a
    # photo's grid stops accepting once it has its share.
    cols, rows = options.grid
    per_cell = max(1, int(math.ceil(options.target_per_image / (cols * rows))))
    filled: dict = {}
    tracks.sort(key=lambda t: (-len(t[0]), -t[1]))
    kept = []
    for nodes, strength in tracks:
        cells = []
        for i, f in nodes:
            width, height = features[i]["size"]
            x, y = features[i]["points"][f]
            cells.append((i, min(cols - 1, int(x / width * cols)), min(rows - 1, int(y / height * rows))))
        if all(filled.get(cell, 0) >= per_cell for cell in cells):
            continue
        for cell in cells:
            filled[cell] = filled.get(cell, 0) + 1
        observations = []
        for i, f in nodes:
            x, y = features[i]["points"][f]
            col, row = _to_full(x, y, features[i]["scale"])
            observations.append((photos[i]["id"], float(col), float(row), None))
        kept.append(TiePoint(observations=observations, score=strength))

    rays = [len(p.observations) for p in kept]
    summary = {
        "photos": n,
        "pairsTried": len(pairs),
        "pairsMatched": matched_pairs,
        "tracks": len(tracks),
        "tiePoints": len(kept),
        "multiRay": int(sum(1 for r in rays if r >= 3)),
        "meanRays": float(np.mean(rays)) if rays else 0.0,
    }
    say(1.0, f"{len(kept)} tie points from {matched_pairs} matched pairs")
    return kept, summary


# -- heading and height from the tie points ------------------------------------------


def heading_and_height(photos: Sequence[dict], points: Sequence[TiePoint], focal_px: float,
                       min_shared: int = 12) -> tuple[dict, Optional[float]]:
    """Each photo's heading, and the block's flying height, from how features move.

    For when the photos carry no gimbal yaw or height above the ground. Between
    two overlapping nadir photos, the ground slides through the frame opposite
    to the camera's motion, turned by the photo's heading and scaled by
    focal length over flying height. With the ground step between the GPS
    positions D (east + i north) and the image shift U (right + i up):
    -U = (f / H) D exp(i heading), the heading clockwise from north.

    Returns ({photo id: heading in degrees}, flying height in metres or None).
    """
    position = {p["id"]: complex(p["position"][0], p["position"][1]) for p in photos}
    shifts: dict = {}
    for point in points:
        obs = [(image_id, col, row) for image_id, col, row, _ in point.observations if image_id in position]
        for a in range(len(obs)):
            for b in range(a + 1, len(obs)):
                (ia, ca, ra), (ib, cb, rb) = obs[a], obs[b]
                shifts.setdefault((ia, ib), []).append(complex(cb - ca, -(rb - ra)))

    turns: dict = {}
    heights = []
    for (ia, ib), values in shifts.items():
        if len(values) < min_shared:
            continue
        step = position[ib] - position[ia]
        if abs(step) < 1.0:
            continue
        shift = complex(np.median([v.real for v in values]), np.median([v.imag for v in values]))
        if abs(shift) < 1.0:
            continue
        turn = -shift * step.conjugate() / (abs(shift) * abs(step))
        weight = math.sqrt(len(values))
        # From b's side the pair says the same with both signs flipped.
        turns.setdefault(ia, []).append(turn * weight)
        turns.setdefault(ib, []).append(turn * weight)
        heights.append(focal_px * abs(step) / abs(shift))

    heading = {pid: math.degrees(np.angle(sum(t))) % 360.0 for pid, t in turns.items()}
    # A photo that shares too little takes its nearest placed neighbour's heading.
    known = [pid for pid in position if pid in heading]
    for pid in position:
        if pid not in heading and known:
            nearest = min(known, key=lambda k: abs(position[k] - position[pid]))
            heading[pid] = heading[nearest]
    return heading, (float(np.median(heights)) if heights else None)


# -- a ground surface for orthophotos -------------------------------------------------


def surface_from_points(points: np.ndarray, output_path: str, crs: str, bounds: tuple,
                        cell_m: float = 1.0, smooth_cells: float = 3.0) -> dict:
    """A smooth elevation surface from the block's triangulated tie points.

    A drone block rarely comes with a DEM, but its solution places a few
    thousand points on the ground (and on whatever stands on it). Gridded and
    smoothed, they are a fair surface to orthorectify onto: it puts tree crowns
    and buildings roughly where they stand, without the spikes a raw
    triangulation of sparse points would throw into the photos.

    ``bounds`` (xmin, ymin, xmax, ymax) is the area to cover, normally the
    photos' footprints; beyond the points the nearest height carries on.
    Smoothing over about three cells keeps the ramps between a crown and the
    ground beside it gentle: steeper ones smear the photo along the ray where
    the surface is only guessed.
    """
    import rasterio
    from rasterio.transform import from_origin
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter

    xyz = np.asarray(points, dtype=float).reshape(-1, 3)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if len(xyz) < 10:
        raise ValueError("Too few solved tie points to build a surface")
    # Points far off the rest (a mismatch that survived) would raise a spike.
    z = xyz[:, 2]
    median = float(np.median(z))
    spread = max(float(np.median(np.abs(z - median))) * 1.4826, 0.5)
    xyz = xyz[np.abs(z - median) < 6.0 * spread + 5.0]

    xmin, ymin, xmax, ymax = bounds
    cols = max(2, int(math.ceil((xmax - xmin) / cell_m)))
    rows = max(2, int(math.ceil((ymax - ymin) / cell_m)))
    xs = xmin + (np.arange(cols) + 0.5) * cell_m
    ys = ymax - (np.arange(rows) + 0.5) * cell_m
    gx, gy = np.meshgrid(xs, ys)

    # One height per occupied cell, the median of its points, so dense patches
    # of tie points do not outvote sparse ones.
    ci = np.clip(((xyz[:, 0] - xmin) / cell_m).astype(int), 0, cols - 1)
    ri = np.clip(((ymax - xyz[:, 1]) / cell_m).astype(int), 0, rows - 1)
    cells: dict = {}
    for c, r, h in zip(ci, ri, xyz[:, 2]):
        cells.setdefault((r, c), []).append(h)
    keys = np.array(list(cells.keys()))
    seeds = np.column_stack([xs[keys[:, 1]], ys[keys[:, 0]]])
    heights = np.array([float(np.median(v)) for v in cells.values()])

    grid = griddata(seeds, heights, (gx, gy), method="linear")
    outside = ~np.isfinite(grid)
    if outside.any():
        grid[outside] = griddata(seeds, heights, (gx[outside], gy[outside]), method="nearest")
    grid = gaussian_filter(grid, smooth_cells, mode="nearest").astype(np.float32)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", driver="GTiff", width=cols, height=rows, count=1,
                       dtype="float32", crs=crs, transform=from_origin(xmin, ymax, cell_m, cell_m),
                       nodata=-9999.0, compress="DEFLATE", tiled=True) as dst:
        dst.write(grid, 1)
        dst.update_tags(SOURCE="Fiducia: surface from drone tie points")
    return {"path": str(output_path), "points": int(len(xyz)), "cellM": cell_m,
            "width": cols, "height": rows, "minZ": float(np.min(grid)), "maxZ": float(np.max(grid))}


# -- a dense surface from the photos themselves -----------------------------------------


def _nan_median(grid: np.ndarray, size: int, min_valid: Optional[int] = None,
                rows_per_block: int = 256) -> np.ndarray:
    """A median filter over the measured cells only.

    Isolated cells that disagree with their neighbourhood (a match through a
    gap in the canopy, wind in the leaves) go; edges such as kerbs and walls,
    which half the window agrees on, stay. A cell with too few measured
    neighbours is left unmeasured rather than invented.
    """
    import warnings
    from numpy.lib.stride_tricks import sliding_window_view

    half = size // 2
    need = min_valid if min_valid is not None else (size * size) // 3
    padded = np.pad(grid, half, constant_values=np.nan)
    out = np.full(grid.shape, np.nan, dtype=np.float32)
    for r0 in range(0, grid.shape[0], rows_per_block):
        r1 = min(grid.shape[0], r0 + rows_per_block)
        windows = sliding_window_view(padded[r0:r1 + 2 * half], (size, size)).reshape(
            r1 - r0, grid.shape[1], size * size)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            median = np.nanmedian(windows, axis=2)
        enough = np.isfinite(windows).sum(axis=2) >= need
        out[r0:r1] = np.where(enough & np.isfinite(grid[r0:r1]), median, np.nan)
    return out


DENSE_DETAIL_PX = {"low": 1200, "medium": 2000, "high": 3000}


def stereo_pairs(images: Sequence[dict], shared: dict, flying_height_m: float,
                 per_photo: int = 2, min_shared: int = 40) -> list[tuple[str, str]]:
    """Which photos to dense-match, and with whom.

    A pair is worth matching when it overlaps well (many shared tie points)
    and its base is neither too short to measure height nor so long the views
    differ too much: a base of a fifth to two thirds of the flying height.
    Each photo takes its best ``per_photo`` partners, so every part of the
    block is seen by several pairs for the fusion to vote among.
    """
    centre = {i["id"]: np.asarray(i["exterior"][:3], float) for i in images}
    scored: dict[str, list] = {}
    for (a, b), count in shared.items():
        if a not in centre or b not in centre or count < min_shared:
            continue
        ratio = float(np.linalg.norm(centre[a] - centre[b])) / max(flying_height_m, 1e-6)
        if not 0.2 <= ratio <= 0.67:
            continue
        # Best near a ratio of 0.35, and with more overlap.
        score = count * math.exp(-((ratio - 0.35) / 0.2) ** 2)
        scored.setdefault(a, []).append((score, a, b))
        scored.setdefault(b, []).append((score, a, b))
    chosen = set()
    for options in scored.values():
        for _, a, b in sorted(options, reverse=True)[:per_photo]:
            chosen.add((a, b) if a < b else (b, a))
    return sorted(chosen)


def fuse_dems(paths: Sequence[str], output_path: str, crs: str, resolution: float,
              fallback_path: Optional[str] = None, tolerance_m: float = 1.0,
              min_agree: int = 2, tile: int = 512, despeckle: int = 5,
              soften_cells: float = 1.0, min_patch_m2: float = 10.0,
              progress: Optional[Callable[[float, str], None]] = None) -> dict:
    """One surface from many pair DEMs on a shared lattice.

    Each cell takes the median of the pairs that see it, then the mean of those
    within ``tolerance_m`` of it; a cell needs ``min_agree`` pairs to agree.
    Wind in the trees and mismatches differ from pair to pair and are voted
    out; real surfaces repeat. Cells nothing agrees on take the fallback
    surface (the tie-point surface), so the result has no holes to trip an
    orthophoto.
    """
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject
    from rasterio.windows import Window

    sources = []
    for path in paths:
        with rasterio.open(path) as d:
            sources.append((path, d.bounds, d.nodata))
    west = min(b.left for _, b, _ in sources)
    north = max(b.top for _, b, _ in sources)
    east = max(b.right for _, b, _ in sources)
    south = min(b.bottom for _, b, _ in sources)
    west = math.floor(west / resolution) * resolution
    north = math.ceil(north / resolution) * resolution
    width = int(math.ceil((east - west) / resolution))
    height = int(math.ceil((north - south) / resolution))
    transform = from_origin(west, north, resolution, resolution)

    out = np.full((height, width), np.nan, dtype=np.float32)
    support = np.zeros((height, width), dtype=np.uint8)
    seen = np.zeros((height, width), dtype=bool)   # matched by at least one pair
    handles = [rasterio.open(p) for p, _, _ in sources]
    try:
        tiles = [(r, c) for r in range(0, height, tile) for c in range(0, width, tile)]
        for n, (r0, c0) in enumerate(tiles):
            h, w = min(tile, height - r0), min(tile, width - c0)
            tx0, ty0 = west + c0 * resolution, north - r0 * resolution
            stack = []
            for d in handles:
                b = d.bounds
                if b.left >= tx0 + w * resolution or b.right <= tx0 or \
                        b.top <= ty0 - h * resolution or b.bottom >= ty0:
                    continue
                # Same lattice: whole-cell offsets.
                col = int(round((tx0 - b.left) / resolution))
                row = int(round((b.top - ty0) / resolution))
                a = d.read(1, window=Window(col, row, w, h), boundless=True,
                           fill_value=d.nodata if d.nodata is not None else -9999.0).astype(np.float32)
                if d.nodata is not None:
                    a[a == d.nodata] = np.nan
                stack.append(a)
            if not stack:
                continue
            cube = np.stack(stack)
            with np.errstate(all="ignore"):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    median = np.nanmedian(cube, axis=0)
                    near = np.abs(cube - median[None]) <= tolerance_m
                    count = near.sum(axis=0)
                    value = np.where(near, cube, 0.0).sum(axis=0) / np.maximum(count, 1)
            agreed = count >= min_agree
            seen[r0:r0 + h, c0:c0 + w] = np.isfinite(cube).any(axis=0)
            out[r0:r0 + h, c0:c0 + w] = np.where(agreed, value, np.nan)
            support[r0:r0 + h, c0:c0 + w] = np.minimum(count, 255)
            if progress:
                progress((n + 1) / len(tiles), "Fusing pair surfaces")
    finally:
        for d in handles:
            d.close()

    if despeckle and despeckle > 1:
        out = _nan_median(out, despeckle)
    if min_patch_m2:
        # Inside canopy the matches come in scraps, real crowns mixed with
        # ground seen through gaps; stitched to the smooth fallback they fold
        # the orthophoto into rings. Only connected areas of some size (open
        # ground, a car, a lone tree) are kept as measured.
        from scipy.ndimage import label

        labels, count = label(np.isfinite(out))
        if count:
            sizes = np.bincount(labels.ravel())
            small = sizes < min_patch_m2 / (resolution * resolution)
            small[0] = False
            out[small[labels]] = np.nan
    measured = np.isfinite(out)
    filled_from_fallback = 0
    if fallback_path and (~measured).any():
        with rasterio.open(fallback_path) as d:
            back = np.full((height, width), np.nan, dtype=np.float32)
            reproject(d.read(1), back, src_transform=d.transform, src_crs=d.crs,
                      dst_transform=transform, dst_crs=crs, src_nodata=d.nodata,
                      dst_nodata=np.nan, resampling=Resampling.bilinear)
        gap = ~measured & np.isfinite(back)
        out[gap] = back[gap]
        filled_from_fallback = int(gap.sum())

    if soften_cells and np.isfinite(out).any():
        # A cell-sharp step at a car's edge or a crown's, or where measured
        # heights meet the fallback, makes an orthophoto stretch the pixels
        # beside it; softened over a cell or so, it does not.
        from scipy.ndimage import gaussian_filter

        valid = np.isfinite(out)
        weight = gaussian_filter(valid.astype(np.float32), soften_cells)
        blurred = gaussian_filter(np.where(valid, out, 0.0).astype(np.float32), soften_cells)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(valid, blurred / np.maximum(weight, 1e-6), np.nan).astype(np.float32)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    nodata = -9999.0
    with rasterio.open(output_path, "w", driver="GTiff", width=width, height=height, count=1,
                       dtype="float32", crs=crs, transform=transform, nodata=nodata,
                       compress="DEFLATE", tiled=True, blockxsize=256, blockysize=256,
                       BIGTIFF="IF_SAFER") as dst:
        dst.write(np.where(np.isfinite(out), out, nodata).astype(np.float32), 1)
        dst.update_tags(SOURCE="Fiducia: dense surface fused from drone stereo pairs")
    finite = out[np.isfinite(out)]
    return {
        "path": str(output_path), "width": width, "height": height, "resolution": resolution,
        "bounds": [west, north - height * resolution, west + width * resolution, north],
        "pairs": len(paths),
        # Of the ground the pairs cover, not of the rectangle around it.
        "measuredFraction": float(measured[seen].mean()) if seen.any() else 0.0,
        "fallbackCells": filled_from_fallback,
        "elevationRange": [float(finite.min()), float(finite.max())] if finite.size else None,
    }


def dense_surface(images: Sequence[dict], camera: dict, observations: Sequence[dict], block: dict,
                  crs: str, work_dir: str, output_path: str, detail: str = "medium",
                  resolution: Optional[float] = None,
                  progress: Optional[Callable[[float, str], None]] = None,
                  should_cancel: Optional[Callable[[], bool]] = None) -> dict:
    """A dense surface model for a solved drone block.

    Overlapping pairs are resampled to epipolar geometry and matched pixel by
    pixel (semi-global matching), each into its own small DEM on a shared
    lattice, and the DEMs are fused by vote. Searches are bounded by the
    tie-point surface, so a low drone's wide range of disparities stays
    tractable. Where no two pairs agree (moving canopy, water, deep shadow)
    the tie-point surface fills in.
    """
    import itertools

    from . import stereo_dem

    say = progress or (lambda fraction, message: None)
    cancelled = should_cancel or (lambda: False)
    surface = block.get("surface") or {}
    if not surface.get("path"):
        raise ValueError("Solve the block first: the dense search is bounded by its tie-point surface")
    placed = [i for i in images if i.get("exterior") and i.get("online", True)]
    ids = {i["id"] for i in placed}

    by_point: dict = {}
    for o in observations:
        if o["imageId"] in ids:
            by_point.setdefault(o["pointId"], set()).add(o["imageId"])
    shared: dict = {}
    for members in by_point.values():
        for a, b in itertools.combinations(sorted(members), 2):
            shared[(a, b)] = shared.get((a, b), 0) + 1
    pairs = stereo_pairs(placed, shared, float(block.get("flyingHeightM") or 50.0), per_photo=3)
    if not pairs:
        raise ValueError("No pair of photos overlaps well enough to match densely")

    gsd = float(block.get("gsdM") or 0.02)
    max_px = DENSE_DETAIL_PX.get(detail, 2000)
    columns = max(int(camera.get("columns") or 0), int(camera.get("rows") or 0)) or 4000
    # Several matched pixels per cell, so each pair gives a cell more than one
    # vote: about eight ground pixels at the matching resolution.
    resolution = float(resolution or round(max(0.05, 8 * gsd * columns / min(max_px, columns)) * 20) / 20)

    lookup = {i["id"]: i for i in placed}
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    def one_pair(pair):
        """A pair's DEM path, or (None, reason)."""
        a, b = pair
        left, right = lookup[a], lookup[b]
        size = max_px
        for _ in range(3):
            pair = stereo_dem.generate_epipolar_pair(left, right, camera, work_dir, max_px=size,
                                                     fit_frame=True)
            camera_z = (pair.left_eo[2] + pair.right_eo[2]) / 2.0
            options = stereo_dem.DemOptions(
                detail=detail if detail in stereo_dem.DETAIL_LEVELS else "medium",
                min_elevation=float(surface["minZ"]) - 6.0,
                max_elevation=min(float(surface["maxZ"]) + 8.0, camera_z - 6.0),
                output_crs=crs, output_path=str(Path(work_dir) / f"dem_{left['name']}_{right['name']}.tif"),
                pixel_sampling=1, output_resolution=resolution, smoothing="low",
                pad_search=True, fill_voids=False, snap_grid=True, speckle_filter=False)
            try:
                return stereo_dem.extract_dem(pair, options)["outputPath"], None
            except ValueError as exc:
                # A long base over tall trees can need more disparity range
                # than the matcher allows: try again at a coarser scale.
                if "too wide" in str(exc):
                    size = int(size * 0.6)
                    continue
                return None, f"{left['name']}–{right['name']}: {exc}"
        return None, f"{left['name']}–{right['name']}: search range too wide"

    from concurrent.futures import ThreadPoolExecutor

    produced, failed = [], []
    # Fewer threads than for matching features: each pair holds two
    # resampled photos and a matcher's cost volume.
    workers = max(1, min(4, _worker_count()))
    outcomes: list = [None] * len(pairs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one_pair, pair): k for k, pair in enumerate(pairs)}
        for done, future in enumerate(_as_completed(futures, cancelled)):
            outcomes[futures[future]] = future.result()
            say(0.9 * (done + 1) / len(pairs), f"Matched {done + 1} of {len(pairs)} pairs densely")
    for path, reason in outcomes:
        if path:
            produced.append(path)
        else:
            failed.append(reason)
    if not produced:
        raise ValueError("No pair could be matched. " + "; ".join(failed[:3]))

    say(0.92, "Fusing pair surfaces")
    from .geodesy import resolve_crs
    info = fuse_dems(produced, output_path, resolve_crs(crs).to_wkt(), resolution,
                     fallback_path=surface["path"])
    info.update({"pairsTried": len(pairs), "pairsFailed": failed[:12], "detail": detail})
    return info
