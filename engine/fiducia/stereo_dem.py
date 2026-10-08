"""Digital elevation models from stereo pairs.

Two stages, exactly as in the classical workflow:

**Epipolar resampling.** The pair is rewritten so that conjugate points share a
row. Once that holds, the search for a match collapses from two dimensions to
one, which is what makes dense matching tractable at all.

**Dense matching.** Semi-global matching by default -- it propagates a
smoothness assumption along several paths through the image, which is
dramatically better than plain correlation over the low-texture areas that
wreck aerial DEMs, and it is what modern implementations use. Normalised
cross-correlation is kept as an option because many workflows still specify it.

Disparity is then converted to elevation by intersecting the two rays, and the
result geocoded onto the output grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .camera import CameraModel, FiducialFit
from .collinearity import rotation_matrix

__all__ = ["EpipolarPair", "DemOptions", "generate_epipolar_pair", "extract_dem", "DETAIL_LEVELS"]

DETAIL_LEVELS = ("low", "medium", "high", "extra_high")
EXTRACTION_METHODS = ("sgm", "ncc")
TERRAIN_TYPES = ("flat", "rolling", "mountainous")


@dataclass
class EpipolarPair:
    left_id: str
    right_id: str
    left_path: str
    right_path: str
    baseline_m: float
    mean_height_m: float
    base_height_ratio: float
    width: int
    height: int
    scale_mm_per_px: float
    left_eo: list
    right_eo: list
    focal_mm: float
    # Where the epipolar principal point sits, in pixels; None is the centre.
    centre_col: Optional[float] = None
    centre_row: Optional[float] = None

    @property
    def centre(self) -> tuple[float, float]:
        return (self.width / 2.0 if self.centre_col is None else self.centre_col,
                self.height / 2.0 if self.centre_row is None else self.centre_row)

    def to_dict(self) -> dict:
        return {
            "leftId": self.left_id,
            "rightId": self.right_id,
            "leftPath": self.left_path,
            "rightPath": self.right_path,
            "baselineM": self.baseline_m,
            "meanHeightM": self.mean_height_m,
            "baseHeightRatio": self.base_height_ratio,
            "width": self.width,
            "height": self.height,
            "scaleMmPerPx": self.scale_mm_per_px,
            "leftEo": list(self.left_eo),
            "rightEo": list(self.right_eo),
            "focalMm": self.focal_mm,
            "centreCol": self.centre_col,
            "centreRow": self.centre_row,
        }


@dataclass
class DemOptions:
    method: str = "sgm"
    detail: str = "high"
    terrain: str = "rolling"
    smoothing: str = "medium"          # none | low | medium | high
    apply_wallis: bool = False
    min_elevation: Optional[float] = None
    max_elevation: Optional[float] = None
    # Where the ground is expected, from the reference DEM or the control,
    # when the operator gives no limits: centres the disparity search on it.
    expected_min_z: Optional[float] = None
    expected_max_z: Optional[float] = None
    # Ground area the DEM may cover (west, south, east, north), output CRS.
    bounds: Optional[tuple] = None
    background_value: float = 0.0
    pixel_sampling: int = 4
    output_resolution: Optional[float] = None
    use_clip_region: bool = True
    vertical_datum: str = "mean_sea_level"
    output_path: str = ""
    output_crs: str = ""
    # The system the orientations -- and so the triangulated points -- are in.
    # Points are converted into output_crs before gridding. "" means the same.
    control_crs: str = ""
    nodata: float = -9999.0
    # Pad the images rather than discard every column whose search window
    # runs off the left edge. A drone pair's disparities are a large part of
    # the frame, and discarding them loses most of the overlap.
    pad_search: bool = False
    # Fill small voids in this pair's DEM. Off when many pairs are fused:
    # filled cells would outvote real matches from the other pairs.
    fill_voids: bool = True
    # Align the grid to whole multiples of the resolution, so DEMs of
    # different pairs share cells.
    snap_grid: bool = False
    # The matcher's speckle filter drops patches whose disparity jumps by more
    # than two pixels. Right for terrain; on a low drone's view of tree crowns
    # it removes the trees. Off when many pairs are fused, which votes out
    # noise instead.
    speckle_filter: bool = True


def _wallis_filter(image: np.ndarray, window: int = 31,
                   target_mean: float = 127.0, target_std: float = 50.0) -> np.ndarray:
    """Local contrast normalisation.

    Stereo matching fails in shadow and on bright water because the local
    contrast there is near zero. Wallis rebalances every neighbourhood to a
    common mean and spread, which buys back a surprising amount of matchable
    texture at the cost of amplifying noise in genuinely flat areas.
    """
    import cv2

    source = image.astype(np.float32)
    mean = cv2.blur(source, (window, window))
    mean_square = cv2.blur(source * source, (window, window))
    std = np.sqrt(np.maximum(mean_square - mean * mean, 0.0))

    gain = target_std / np.maximum(std, 1.0)
    gain = np.clip(gain, 0.3, 4.0)
    result = (source - mean) * gain + target_mean
    return np.clip(result, 0, 255).astype(np.uint8)


# Matching resolution for each detail level: the long side of the epipolar
# images, in pixels. Finer imagery gives a more precise surface (height
# precision is proportional to the matched pixel's size) at the cost of time.
DETAIL_MAX_PX = {"low": 2000, "medium": 3000, "high": 4000, "extra_high": 8000}


def _load_grey(path: str, max_px: int) -> tuple[np.ndarray, tuple[float, float]]:
    """A reduced single band and its (x, y) reduction factors."""
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as dataset:
        factor = max(1.0, max(dataset.width, dataset.height) / max_px)
        out_w = max(1, int(dataset.width / factor))
        out_h = max(1, int(dataset.height / factor))
        band = 2 if dataset.count >= 3 else 1
        data = dataset.read(
            band, out_shape=(out_h, out_w), resampling=Resampling.average
        ).astype(np.float32)
        actual = (dataset.width / out_w, dataset.height / out_h)

    finite = data[np.isfinite(data)]
    if finite.size:
        lo, hi = np.percentile(finite, [1, 99])
        data = np.clip((data - lo) / max(float(hi - lo), 1e-6), 0, 1) * 255.0
    return data.astype(np.uint8), (float(actual[0]), float(actual[1]))


def generate_epipolar_pair(
    left_image: dict,
    right_image: dict,
    camera_dict: dict,
    output_dir: str,
    max_px: int = 4000,
    progress: Optional[Callable[[float, str], None]] = None,
    ground_height_m: Optional[float] = None,
    fit_frame: bool = False,
) -> EpipolarPair:
    """Resample a stereo pair into epipolar geometry.

    The normalised (epipolar) geometry is built by constructing a common
    rotation whose x axis lies along the baseline between the two perspective
    centres, then reprojecting both photos through it. After this, conjugate
    points differ only in column.

    The rays are straight, as the adjustment's are once refraction and earth
    curvature are removed; where the camera applies those corrections they
    are put back here at ``ground_height_m``, so the epipolar images agree
    with the model the orientations came from.
    """
    import cv2
    import rasterio

    camera = CameraModel.from_dict(camera_dict)
    left_eo = np.asarray(left_image["exterior"], dtype=float)
    right_eo = np.asarray(right_image["exterior"], dtype=float)

    left_fit = FiducialFit.from_dict(left_image["fiducialFit"]) if left_image.get("fiducialFit") else None
    right_fit = FiducialFit.from_dict(right_image["fiducialFit"]) if right_image.get("fiducialFit") else None

    baseline_vector = right_eo[:3] - left_eo[:3]
    baseline = float(np.linalg.norm(baseline_vector))
    if baseline < 1e-6:
        raise ValueError("The two photos share a perspective centre -- no stereo geometry")

    # Build the common epipolar frame: x along the baseline, z roughly
    # preserving the mean viewing direction of the pair.
    x_axis = baseline_vector / baseline
    mean_rotation = rotation_matrix(
        (left_eo[3] + right_eo[3]) / 2,
        (left_eo[4] + right_eo[4]) / 2,
        (left_eo[5] + right_eo[5]) / 2,
    )
    nominal_z = mean_rotation[2]
    y_axis = np.cross(nominal_z, x_axis)
    norm = np.linalg.norm(y_axis)
    if norm < 1e-9:
        y_axis = np.cross(np.array([0.0, 0.0, 1.0]), x_axis)
        norm = np.linalg.norm(y_axis)
    y_axis = y_axis / norm
    z_axis = np.cross(x_axis, y_axis)
    epipolar_rotation = np.vstack([x_axis, y_axis, z_axis])

    if progress:
        progress(0.15, "Resampling left photo")

    left_grey, left_scale = _load_grey(left_image["path"], max_px)
    right_grey, right_scale = _load_grey(right_image["path"], max_px)

    height, width = left_grey.shape
    scale_mm = camera.pixel_scale_mm(left_fit) * float(np.mean(left_scale))
    correct = camera.apply_atmospheric or camera.apply_earth_curvature
    ground_z = float(ground_height_m) if ground_height_m is not None else 0.0
    centre_col, centre_row = width / 2.0, height / 2.0
    if fit_frame:
        # Size the frame to hold both photos as the epipolar rotation turns
        # them. Kept at the photo's own shape, a pair flown along the photo's
        # long side (every drone strip) loses most of each frame.
        corners = []
        source_h, source_w = left_grey.shape
        full = np.array([[0, 0], [source_w * left_scale[0], 0],
                         [source_w * left_scale[0], source_h * left_scale[1]],
                         [0, source_h * left_scale[1]]], dtype=float) - 0.5
        for eo, fit in ((left_eo, left_fit), (right_eo, right_fit)):
            film = camera.pixel_to_film(full, fit)
            rays = rotation_matrix(eo[3], eo[4], eo[5]).T @ np.vstack(
                [film.T, np.full(4, -camera.focal_mm)])
            epi = epipolar_rotation @ rays
            corners.append(np.column_stack([-camera.focal_mm * epi[0] / epi[2],
                                            -camera.focal_mm * epi[1] / epi[2]]) / scale_mm)
        corners = np.vstack(corners)
        x0, y0 = corners.min(axis=0)
        x1, y1 = corners.max(axis=0)
        width, height = int(math.ceil(x1 - x0)) + 2, int(math.ceil(y1 - y0)) + 2
        # Epipolar column = ex / scale + centre_col; row = -ey / scale + centre_row.
        centre_col, centre_row = -x0 + 1.0, y1 + 1.0

    def build_map(row0, row1, eo, fit, decimation):
        """Lookup table taking epipolar pixels (rows row0..row1) to reduced
        source pixels, whole numbers at pixel centres throughout."""
        rows, cols = np.mgrid[row0:row1, 0:width].astype(np.float64)
        # Epipolar focal-plane coordinates for this output grid.
        ex = (cols - centre_col) * scale_mm
        ey = -(rows - centre_row) * scale_mm

        directions = np.stack([ex.ravel(), ey.ravel(), np.full(ex.size, -camera.focal_mm)])
        # Epipolar frame -> world -> this photo's camera frame.
        photo = rotation_matrix(eo[3], eo[4], eo[5]) @ (epipolar_rotation.T @ directions)

        with np.errstate(divide="ignore", invalid="ignore"):
            film = np.column_stack([-camera.focal_mm * photo[0] / photo[2],
                                    -camera.focal_mm * photo[1] / photo[2]])
        film = np.nan_to_num(film, nan=0.0, posinf=0.0, neginf=0.0)
        if correct:
            from .corrections import EARTH_RADIUS_M, from_flat

            film = from_flat(film, camera.focal_mm, float(eo[2]), ground_z,
                             bool(camera.apply_atmospheric), bool(camera.apply_earth_curvature),
                             camera.earth_radius_m or EARTH_RADIUS_M)
        pixels = camera.film_to_pixel(film, fit)

        # A full-resolution pixel-centre coordinate p lies at (p + 0.5) / s - 0.5
        # in a copy reduced by s.
        shape = (row1 - row0, width)
        map_x = ((pixels[:, 0] + 0.5) / decimation[0] - 0.5).reshape(shape).astype(np.float32)
        map_y = ((pixels[:, 1] + 0.5) / decimation[1] - 0.5).reshape(shape).astype(np.float32)
        return map_x, map_y

    # In blocks of rows, so the lookup tables never need more than a few
    # tens of megabytes however fine the detail.
    left_epi = np.zeros((height, width), dtype=np.uint8)
    right_epi = np.zeros((height, width), dtype=np.uint8)
    block_rows = max(64, int(2_000_000 // max(width, 1)))
    for row0 in range(0, height, block_rows):
        row1 = min(height, row0 + block_rows)
        for source, eo, fit, scale, target in (
                (left_grey, left_eo, left_fit, left_scale, left_epi),
                (right_grey, right_eo, right_fit, right_scale, right_epi)):
            map_x, map_y = build_map(row0, row1, eo, fit, scale)
            target[row0:row1] = cv2.remap(source, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)
        if progress:
            progress(0.15 + 0.6 * row1 / height, "Resampling into epipolar geometry")

    if progress:
        progress(0.8, "Writing epipolar images")

    work_dir = Path(output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{left_image['name']}_{right_image['name']}"
    left_path = work_dir / f"el_{stem}.tif"
    right_path = work_dir / f"er_{stem}.tif"

    for array, target in ((left_epi, left_path), (right_epi, right_path)):
        with rasterio.open(
            target, "w", driver="GTiff", width=width, height=height,
            count=1, dtype="uint8", compress="DEFLATE", tiled=True,
            blockxsize=256, blockysize=256,
        ) as sink:
            sink.write(array, 1)

    mean_height = float((left_eo[2] + right_eo[2]) / 2.0)
    if progress:
        progress(1.0, "Epipolar pair ready")

    return EpipolarPair(
        left_id=left_image["id"],
        right_id=right_image["id"],
        left_path=str(left_path),
        right_path=str(right_path),
        baseline_m=baseline,
        mean_height_m=mean_height,
        base_height_ratio=baseline / max(mean_height, 1e-6),
        width=width,
        height=height,
        scale_mm_per_px=scale_mm,
        left_eo=left_eo.tolist(),
        right_eo=right_eo.tolist(),
        focal_mm=camera.focal_mm,
        centre_col=centre_col if fit_frame else None,
        centre_row=centre_row if fit_frame else None,
    )


def _match_disparity(
    left: np.ndarray,
    right: np.ndarray,
    options: DemOptions,
    min_disparity: int,
    num_disparities: int,
) -> np.ndarray:
    """Disparity in pixels, searched over [min_disparity, + num_disparities).

    The window must sit where the ground actually is. In a normal stereo pair
    the whole scene is shifted by baseline * focal / height -- hundreds or
    thousands of pixels -- and a window around zero never sees it.
    """
    import cv2

    # Detail sets the image resolution (DETAIL_MAX_PX); the window is sized
    # to it. Smaller windows resolve more but are noisier, and at the finest
    # level the pixels are already small enough that 5 is the useful floor.
    block = {"low": 9, "medium": 7, "high": 5, "extra_high": 5}.get(options.detail, 5)
    # Terrain type sets the smoothness penalties -- flat terrain tolerates
    # much stronger regularisation than mountainous.
    terrain_weight = {"flat": 2.0, "rolling": 1.0, "mountainous": 0.5}.get(options.terrain, 1.0)

    num_disparities = max(16, int(math.ceil(num_disparities / 16.0)) * 16)

    # The matchers store disparity as 16-bit integers in sixteenths of a
    # pixel, so nothing beyond 2047 px survives -- and a large-format frame
    # matched at fine detail has disparities of thousands of pixels. The
    # right image is shifted by the window start instead, the search runs
    # from zero, and the shift is added back.
    shifted = np.zeros_like(right)
    if min_disparity >= 0:
        shifted[:, min_disparity:] = right[:, :right.shape[1] - min_disparity]
    else:
        shifted[:, :min_disparity] = right[:, -min_disparity:]

    if options.method == "ncc":
        matcher = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=max(5, block | 1))
        matcher.setUniquenessRatio(10)
        matcher.setSpeckleWindowSize(100)
        matcher.setSpeckleRange(32)
        raw = matcher.compute(left, shifted)
        disparity = raw.astype(np.float32) / 16.0
        disparity[raw <= 0] = np.nan
        disparity += min_disparity
    else:
        channels = 1
        matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disparities,
            blockSize=block,
            P1=int(8 * channels * block * block * terrain_weight),
            P2=int(32 * channels * block * block * terrain_weight),
            disp12MaxDiff=2,
            uniquenessRatio=8,
            speckleWindowSize=150 if options.speckle_filter else 0,
            speckleRange=2 if options.speckle_filter else 0,
            preFilterCap=31,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        pad = num_disparities + block if options.pad_search else 0
        if pad:
            raw = matcher.compute(np.pad(left, ((0, 0), (pad, 0))),
                                  np.pad(shifted, ((0, 0), (pad, 0))))[:, pad:]
        else:
            raw = matcher.compute(left, shifted)
        disparity = raw.astype(np.float32) / 16.0
        # Unmatched pixels come back below the window (as -1).
        disparity[raw < 0] = np.nan
        disparity += min_disparity

    # Only left pixels whose whole search window lies on the right image can
    # be trusted: elsewhere some candidates were padding, and the matcher's
    # path costs carry that into its answer. (Without the shift, the matcher
    # discards these columns itself.)
    if options.pad_search and options.method != "ncc":
        # Padded instead: only a match that landed in the padding is lost.
        columns = np.arange(disparity.shape[1])[None, :]
        with np.errstate(invalid="ignore"):
            disparity[(columns - disparity) < block // 2] = np.nan
        # Nor one matched against the empty frame around the right photo.
        rows_idx = np.indices(disparity.shape)[0]
        source = np.nan_to_num(columns - disparity, nan=-1).round().astype(np.int64)
        inside = (source >= 0) & (source < right.shape[1])
        content = np.zeros(disparity.shape, dtype=bool)
        content[inside] = right[rows_idx[inside], source[inside]] > 0
        disparity[~content | (left == 0)] = np.nan
    else:
        disparity[:, :max(0, min_disparity) + num_disparities + block // 2] = np.nan

    smoothing = {"none": 0, "low": 3, "medium": 5, "high": 9}.get(options.smoothing, 5)
    if smoothing:
        filled = np.nan_to_num(disparity, nan=0.0)
        mask = np.isfinite(disparity).astype(np.float32)
        blurred = cv2.medianBlur(filled, smoothing)
        disparity = np.where(mask > 0, blurred, np.nan)

    return disparity


def extract_dem(
    pair: EpipolarPair,
    options: DemOptions,
    progress: Optional[Callable[[float, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> dict:
    """Dense-match an epipolar pair and geocode the result as a DEM."""
    import cv2
    import rasterio
    from rasterio.transform import Affine

    from .geodesy import file_crs

    if progress:
        progress(0.05, "Loading epipolar pair")

    with rasterio.open(pair.left_path) as dataset:
        left = dataset.read(1)
    with rasterio.open(pair.right_path) as dataset:
        right = dataset.read(1)

    if options.apply_wallis:
        if progress:
            progress(0.12, "Applying Wallis filter")
        left = _wallis_filter(left)
        right = _wallis_filter(right)

    if should_cancel and should_cancel():
        raise InterruptedError("Cancelled")

    # Where the ground can be: the operator's limits, else the expected range
    # (reference DEM or control), with a margin for buildings and the error
    # in either; failing both, a generous band below the flying height.
    focal_px = pair.focal_mm / max(pair.scale_mm_per_px, 1e-9)
    flying = pair.mean_height_m
    low = options.min_elevation if options.min_elevation is not None else options.expected_min_z
    high = options.max_elevation if options.max_elevation is not None else options.expected_max_z
    if low is None or high is None:
        low, high = flying - 0.95 * flying, flying - 0.6 * flying
    if options.min_elevation is None or options.max_elevation is None:
        margin = max(40.0, 0.25 * (high - low))
        low, high = low - margin, high + margin
    if high >= flying:
        high = flying - 0.05 * flying

    # Disparity is baseline * focal / distance below the cameras: higher
    # ground is nearer, so has the larger disparity.
    d_low = pair.baseline_m * focal_px / max(flying - low, 1.0)
    d_high = pair.baseline_m * focal_px / max(flying - high, 1.0)
    min_disparity = int(math.floor(d_low)) - 8
    span = int(math.ceil(d_high - d_low)) + 16
    if span > 1024:
        raise ValueError(
            f"The elevation range to search ({low:.0f} to {high:.0f} m) is too wide for "
            "this pair at this detail. Set minimum and maximum elevations, or use a "
            "lower detail level."
        )

    if progress:
        progress(0.25, f"Dense matching ({options.method.upper()}, {options.detail} detail)")

    disparity = _match_disparity(left, right, options, min_disparity, span)

    if should_cancel and should_cancel():
        raise InterruptedError("Cancelled")

    if progress:
        progress(0.6, "Converting disparity to elevation")

    # Depth from disparity, D = B * f / d, is measured along the epipolar
    # frame's viewing axis -- the pair's mean camera axis, not the vertical.
    # Each point is triangulated in that frame and rotated back to the world,
    # so a tilted pair gives level ground rather than a tilted surface.
    height, width = disparity.shape
    step = max(1, int(options.pixel_sampling))
    rows, cols = np.indices((height, width))[:, ::step, ::step]
    sampled = disparity[::step, ::step]
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = (pair.baseline_m * focal_px) / sampled
    keep = np.isfinite(depth) & (depth > 0)
    if not keep.any():
        raise ValueError(
            "Dense matching produced no valid elevations. The most common cause "
            "is a weak base-to-height ratio or an exterior orientation that has "
            f"not converged (this pair: B/H = {pair.base_height_ratio:.3f})."
        )

    if progress:
        progress(0.75, "Geocoding")

    left_eo = np.asarray(pair.left_eo, dtype=float)
    scale = pair.scale_mm_per_px
    centre_col, centre_row = pair.centre
    ex = (cols[keep] - centre_col) * scale
    ey = -(rows[keep] - centre_row) * scale
    dv = depth[keep]

    baseline_vector = np.asarray(pair.right_eo[:3]) - np.asarray(pair.left_eo[:3])
    x_axis = baseline_vector / np.linalg.norm(baseline_vector)
    mean_rotation = rotation_matrix(
        (pair.left_eo[3] + pair.right_eo[3]) / 2,
        (pair.left_eo[4] + pair.right_eo[4]) / 2,
        (pair.left_eo[5] + pair.right_eo[5]) / 2,
    )
    y_axis = np.cross(mean_rotation[2], x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    epipolar_rotation = np.vstack([x_axis, y_axis, z_axis])

    # The point in the epipolar frame is (ex, ey, -f) scaled by D / f.
    offsets = epipolar_rotation.T @ (
        np.stack([ex, ey, np.full(ex.size, -pair.focal_mm)]) * (dv / pair.focal_mm)
    )
    ground_x = left_eo[0] + offsets[0]
    ground_y = left_eo[1] + offsets[1]
    zv = left_eo[2] + offsets[2]

    # Only elevations inside the searched band are real matches; anything at
    # the edge of the window or beyond is the matcher giving up.
    finite = (np.isfinite(ground_x) & np.isfinite(ground_y) & np.isfinite(zv)
              & (zv >= low) & (zv <= high))
    ground_x, ground_y, zv = ground_x[finite], ground_y[finite], zv[finite]
    if ground_x.size < 10:
        raise ValueError(
            "Dense matching produced no valid elevations. The most common cause "
            "is a weak base-to-height ratio or an exterior orientation that has "
            f"not converged (this pair: B/H = {pair.base_height_ratio:.3f})."
        )

    # Triangulated in the control's system; gridded in the output's. The
    # points are converted before binning, so no elevation is resampled.
    from .ortho import _converter

    to_output = _converter(options.control_crs or options.output_crs, options.output_crs)
    if to_output:
        ground_x, ground_y = to_output(ground_x, ground_y)

    # Keep to the ground the pair can see; a stray point far outside would
    # otherwise size the grid to cover it.
    if options.bounds:
        bw, bs, be, bn = options.bounds
        inside = (ground_x >= bw) & (ground_x <= be) & (ground_y >= bs) & (ground_y <= bn)
        ground_x, ground_y, zv = ground_x[inside], ground_y[inside], zv[inside]
        if ground_x.size < 10:
            raise ValueError("Too few matched points fell inside the pair's overlap")

    resolution = options.output_resolution or (
        scale * step * (pair.mean_height_m - 0.5 * (low + high)) / pair.focal_mm)
    resolution = max(float(resolution), 1e-6)

    west, east = float(ground_x.min()), float(ground_x.max())
    south, north = float(ground_y.min()), float(ground_y.max())
    if options.snap_grid:
        west = math.floor(west / resolution) * resolution
        north = math.ceil(north / resolution) * resolution
    out_w = max(1, int(math.ceil((east - west) / resolution)))
    out_h = max(1, int(math.ceil((north - south) / resolution)))
    if out_w * out_h > 400_000_000:
        raise ValueError(
            f"The DEM would be {out_w} x {out_h} cells at {resolution:.2f} m, far larger "
            "than this pair can cover. The orientation or the elevation range is wrong."
        )

    grid_sum = np.zeros(out_w * out_h)
    grid_count = np.zeros(out_w * out_h)
    cc = np.clip(((ground_x - west) / resolution).astype(np.int64), 0, out_w - 1)
    rr = np.clip(((north - ground_y) / resolution).astype(np.int64), 0, out_h - 1)
    flat = rr * out_w + cc
    np.add.at(grid_sum, flat, zv)
    np.add.at(grid_count, flat, 1.0)

    grid = np.where(grid_count > 0, grid_sum / np.maximum(grid_count, 1), np.nan)
    grid = grid.reshape(out_h, out_w)

    from .lidar import _fill_voids

    if options.fill_voids:
        grid = _fill_voids(grid, "linear", 12)
    output = np.where(np.isfinite(grid), grid, options.nodata).astype(np.float32)

    if progress:
        progress(0.9, "Writing DEM")

    Path(options.output_path).parent.mkdir(parents=True, exist_ok=True)
    transform = Affine(resolution, 0.0, west, 0.0, -resolution, north)

    with rasterio.open(
        options.output_path, "w", driver="GTiff", width=out_w, height=out_h,
        count=1, dtype="float32", crs=file_crs(options.output_crs),
        transform=transform, nodata=options.nodata, tiled=True,
        blockxsize=256, blockysize=256, compress="DEFLATE", BIGTIFF="IF_SAFER",
    ) as sink:
        sink.write(output, 1)

    try:
        from . import raster

        raster.build_overviews(options.output_path)
    except Exception:
        pass

    if progress:
        progress(1.0, "DEM complete")

    finite_grid = grid[np.isfinite(grid)]
    return {
        "outputPath": options.output_path,
        "width": out_w,
        "height": out_h,
        "resolution": resolution,
        "bounds": [west, south, east, north],
        "coverage": float(np.isfinite(grid).mean()),
        "elevationRange": [float(finite_grid.min()), float(finite_grid.max())],
        "baseHeightRatio": pair.base_height_ratio,
        "method": options.method,
        "detail": options.detail,
        "pointsUsed": int(ground_x.size),
        "leftId": pair.left_id,
        "rightId": pair.right_id,
    }
