"""Automatic ground control against a reference orthomosaic.

Collecting ground control by hand is the slowest step in the whole chain: find
a feature on the photograph, find the same feature on the reference, click
both, read the elevation, accept, repeat. On a typical block it takes longer
than everything else put together.

Everything needed to do it automatically is already in the project. There is a
rectified reference image, a DEM, and a sensor model good enough to say roughly
where a photo pixel lands on the ground. So:

1. Find well-conditioned interest points on the photograph.
2. For each one, rectify a small patch of the photograph onto the ground using
   the current model and the DEM.
3. Correlate that patch against the reference at the same ground resolution.
4. The offset that maximises correlation is the correction to the ground
   coordinate; the DEM supplies the elevation.

Rectifying before correlating is the part that makes this work. A raw aerial
photo and an orthomosaic differ in scale, rotation and terrain displacement,
and correlating them directly fails. Once both are on the ground at the same
resolution they are directly comparable, and ordinary cross-correlation is
enough.

The output is a set of proposals with scores, not measurements. The operator
reviews them -- which is a completely different job from finding them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from .camera import CameraModel, FiducialFit
from .collinearity import rotation_matrix

__all__ = ["GcpProposal", "AutoControlOptions", "AutoControlResult", "match_ground_control"]


@dataclass
class GcpProposal:
    col: float
    row: float
    x: float
    y: float
    z: float
    score: float
    shift_m: float

    def to_dict(self) -> dict:
        return {
            "col": float(self.col), "row": float(self.row),
            "x": float(self.x), "y": float(self.y), "z": float(self.z),
            "score": float(self.score), "shiftM": float(self.shift_m),
        }


@dataclass
class AutoControlOptions:
    target_count: int = 20
    patch_px: int = 96              # correlation patch, at reference resolution
    search_m: float = 90.0          # how far the model might be wrong, in metres
    min_score: float = 0.55
    min_separation_m: float = 120.0
    edge_margin_px: int = 150       # keep away from the frame border
    candidate_pool: int = 220
    ransac_sigma: float = 3.0       # outlier rejection on the shift field
    # Ground height used wherever no DEM covers a point. Sea level is wrong
    # for almost all land: relief displacement is proportional to height, so
    # at 1700 m (Johannesburg) a sea-level assumption misplaces every point by
    # more than half its distance from nadir divided by the flying height.
    background_elevation: float = 0.0


@dataclass
class AutoControlResult:
    proposals: list[GcpProposal]
    tested: int
    rejected_score: int
    rejected_geometry: int
    mean_shift_m: float
    seed: str
    message: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "proposals": [p.to_dict() for p in self.proposals],
            "tested": self.tested,
            "rejectedScore": self.rejected_score,
            "rejectedGeometry": self.rejected_geometry,
            "meanShiftM": float(self.mean_shift_m),
            "seed": self.seed,
            "message": self.message,
            "warnings": self.warnings,
        }


# -- mapping between the photograph and the ground -------------------------


def _dem_sampler(dem_path: Optional[str], control_crs: Optional[str]):
    """Heights at control-frame points, read in the DEM's own projection.

    Returns None without a DEM. The DEM may be stored in another system than
    the control; reading it at control-frame numbers would take heights from
    the wrong place entirely.
    """
    if not dem_path:
        return None
    from . import raster
    from .ortho import _converter, _dem_crs

    to_dem = _converter(control_crs, _dem_crs(dem_path))

    def sample(xs, ys):
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        if to_dem:
            xs, ys = to_dem(xs, ys)
        return raster.sample_elevation(dem_path, xs, ys)
    return sample


def _rigorous_mapping(camera: CameraModel, fit, eo: np.ndarray, dem_sample,
                      base_elevation: float = 0.0):
    """Photo pixel <-> ground, using collinearity and the DEM.

    Returns ``(pixel_to_ground, ground_to_pixel)``.
    """
    from . import raster

    rot = rotation_matrix(eo[3], eo[4], eo[5])

    def pixel_to_ground(pixels: np.ndarray, iterations: int = 3) -> np.ndarray:
        film = camera.pixel_to_film(pixels, fit)
        directions = np.column_stack(
            [film[:, 0], film[:, 1], np.full(film.shape[0], -camera.focal_mm)]
        ) @ rot

        elevation = np.full(film.shape[0], float(base_elevation))
        ground = None
        for _ in range(iterations):
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (elevation - eo[2]) / directions[:, 2]
            ground = np.column_stack(
                [eo[0] + t * directions[:, 0], eo[1] + t * directions[:, 1], elevation]
            )
            if dem_sample is None:
                break
            sampled = dem_sample(ground[:, 0], ground[:, 1])
            new_elevation = np.where(np.isfinite(sampled), sampled, elevation)
            if np.allclose(new_elevation, elevation, atol=0.05):
                elevation = new_elevation
                break
            elevation = new_elevation

        ground[:, 2] = elevation
        return ground

    def ground_to_pixel(ground: np.ndarray) -> np.ndarray:
        delta = np.atleast_2d(ground) - eo[:3]
        cam = delta @ rot.T
        with np.errstate(divide="ignore", invalid="ignore"):
            film = np.column_stack([
                -camera.focal_mm * cam[:, 0] / cam[:, 2],
                -camera.focal_mm * cam[:, 1] / cam[:, 2],
            ])
        film = np.nan_to_num(film, nan=0.0, posinf=0.0, neginf=0.0)
        return camera.film_to_pixel(film, fit)

    return pixel_to_ground, ground_to_pixel


def _affine_mapping(observations: Sequence[tuple[float, float, float, float]],
                    dem_sample, base_elevation: float = 0.0):
    """Photo pixel <-> ground from a plane affine fitted to manual control.

    Used to bootstrap before an exterior orientation exists. It ignores
    terrain displacement, so it is only a seed -- but it is easily good enough
    to bring the correct feature inside the search window, which is all the
    correlation needs.
    """
    from . import raster

    data = np.asarray(observations, dtype=float)
    if data.shape[0] < 3:
        raise ValueError("At least three measured control points are needed to seed")

    design = np.column_stack([np.ones(len(data)), data[:, 0], data[:, 1]])
    ax, *_ = np.linalg.lstsq(design, data[:, 2], rcond=None)
    ay, *_ = np.linalg.lstsq(design, data[:, 3], rcond=None)

    forward = np.array([[ax[1], ax[2]], [ay[1], ay[2]]])
    if abs(np.linalg.det(forward)) < 1e-12:
        raise ValueError("The seed control points are collinear")
    inverse = np.linalg.inv(forward)
    offset = -inverse @ np.array([ax[0], ay[0]])

    def pixel_to_ground(pixels: np.ndarray, iterations: int = 1) -> np.ndarray:
        pixels = np.atleast_2d(pixels)
        xs = ax[0] + ax[1] * pixels[:, 0] + ax[2] * pixels[:, 1]
        ys = ay[0] + ay[1] * pixels[:, 0] + ay[2] * pixels[:, 1]
        zs = np.full(len(xs), float(base_elevation))
        if dem_sample is not None:
            sampled = dem_sample(xs, ys)
            zs = np.where(np.isfinite(sampled), sampled, 0.0)
        return np.column_stack([xs, ys, zs])

    def ground_to_pixel(ground: np.ndarray) -> np.ndarray:
        ground = np.atleast_2d(ground)
        cols = offset[0] + inverse[0, 0] * ground[:, 0] + inverse[0, 1] * ground[:, 1]
        rows = offset[1] + inverse[1, 0] * ground[:, 0] + inverse[1, 1] * ground[:, 1]
        return np.column_stack([cols, rows])

    return pixel_to_ground, ground_to_pixel


# -- patch rendering -------------------------------------------------------


def _photo_patch_on_ground(
    dataset,
    ground_to_pixel,
    centre: tuple[float, float],
    size_px: int,
    resolution: float,
    dem_sample=None,
    fallback_elevation: float = 0.0,
) -> Optional[np.ndarray]:
    """Rectify a square of the photograph onto the ground, in memory.

    This is the same backward projection the ortho generator performs, over a
    patch small enough that reading it costs nothing -- and like it, each
    ground cell is projected at its own height. (An earlier version projected
    every cell at elevation zero, which on real terrain displaces the patch
    by its height times its distance from nadir over the flying height: on a
    hilly synthetic block, errors of 5-16 m that its flat test scene hid.)
    """
    from rasterio.windows import Window
    from scipy.ndimage import map_coordinates

    half = size_px / 2.0
    offsets = (np.arange(size_px) - half + 0.5) * resolution
    grid_x, grid_y = np.meshgrid(centre[0] + offsets, centre[1] - offsets)

    heights = np.full(grid_x.size, float(fallback_elevation))
    if dem_sample is not None:
        sampled = dem_sample(grid_x.ravel(), grid_y.ravel())
        heights = np.where(np.isfinite(sampled), sampled, heights)
    pixels = ground_to_pixel(np.column_stack([grid_x.ravel(), grid_y.ravel(), heights]))
    cols = pixels[:, 0].reshape(size_px, size_px)
    rows = pixels[:, 1].reshape(size_px, size_px)

    inside = (
        np.isfinite(cols) & np.isfinite(rows)
        & (cols >= 0) & (cols < dataset.width)
        & (rows >= 0) & (rows < dataset.height)
    )
    if inside.mean() < 0.85:
        return None

    margin = 3
    c0 = max(0, int(np.floor(np.min(cols[inside]))) - margin)
    r0 = max(0, int(np.floor(np.min(rows[inside]))) - margin)
    c1 = min(dataset.width, int(np.ceil(np.max(cols[inside]))) + margin + 1)
    r1 = min(dataset.height, int(np.ceil(np.max(rows[inside]))) + margin + 1)
    if c1 - c0 < 4 or r1 - r0 < 4:
        return None

    band = 1 if dataset.count < 3 else 2
    block = dataset.read(band, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(np.float32)

    sampled = map_coordinates(
        block,
        np.vstack([
            np.where(inside, rows - r0, 0).ravel(),
            np.where(inside, cols - c0, 0).ravel(),
        ]),
        order=1, mode="constant", cval=0.0,
    ).reshape(size_px, size_px)

    return np.where(inside, sampled, 0.0)


def _reference_patch(reference, centre: tuple[float, float],
                     size_px: int, resolution: float,
                     control_crs=None) -> Optional[np.ndarray]:
    """Read the reference orthomosaic on the same ground grid.

    The grid is in the control's coordinate system -- that is where the
    photo patch is laid out and where the matched position is reported --
    so the reference is resampled into it from whatever system it is in.
    """
    from rasterio.warp import Resampling, reproject
    from rasterio.transform import Affine

    half = size_px * resolution / 2.0
    transform = Affine(resolution, 0.0, centre[0] - half,
                       0.0, -resolution, centre[1] + half)

    band = 1 if reference.count < 3 else 2
    destination = np.zeros((size_px, size_px), dtype=np.float32)
    try:
        reproject(
            source=__import__("rasterio").band(reference, band),
            destination=destination,
            src_transform=reference.transform,
            src_crs=reference.crs,
            dst_transform=transform,
            dst_crs=control_crs or reference.crs,
            resampling=Resampling.bilinear,
        )
    except Exception:
        return None

    return destination


def _stretch(patch: np.ndarray) -> Optional[np.ndarray]:
    finite = patch[np.isfinite(patch) & (patch != 0)]
    if finite.size < patch.size * 0.5:
        return None
    lo, hi = np.percentile(finite, [2, 98])
    if hi - lo < 1e-6:
        return None
    return np.clip((patch - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


# -- the main routine ------------------------------------------------------


def match_ground_control(
    image: dict,
    camera_dict: dict,
    reference_path: str,
    dem_path: Optional[str],
    seed_observations: Optional[Sequence[tuple[float, float, float, float]]] = None,
    options: Optional[AutoControlOptions] = None,
    progress: Optional[Callable[[float, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    control_crs: Optional[str] = None,
    seed_source: str = "manual control",
) -> AutoControlResult:
    """Propose ground control points for one photograph.

    ``control_crs`` is the system the orientation and the proposed
    coordinates are in; the reference image and DEM may each be in another.
    """
    import cv2
    import rasterio

    options = options or AutoControlOptions()
    warnings: list[str] = []

    camera = CameraModel.from_dict(camera_dict)
    fit = FiducialFit.from_dict(image["fiducialFit"]) if image.get("fiducialFit") else None
    dem_sample = _dem_sampler(dem_path, control_crs)
    control_system = None
    if control_crs:
        from .geodesy import resolve_crs

        control_system = resolve_crs(control_crs)

    if camera.kind == "film" and fit is None:
        raise ValueError(
            f"{image.get('name')} has no interior orientation yet. "
            "Measure its fiducial marks first."
        )

    # -- how do we know where a photo pixel lands on the ground? ----------
    if image.get("exterior"):
        pixel_to_ground, ground_to_pixel = _rigorous_mapping(
            camera, fit, np.asarray(image["exterior"], dtype=float), dem_sample,
            options.background_elevation,
        )
        seed = "model"
    elif seed_observations and len(seed_observations) >= 3:
        pixel_to_ground, ground_to_pixel = _affine_mapping(seed_observations, dem_sample, options.background_elevation)
        seed = seed_source
        warnings.append(
            ("Placed automatically on the reference" if seed_source == "placement"
             else "Seeded from a plane fit to your manual points")
            + ", so terrain displacement is not yet accounted for. Re-run after "
            "computing the model to find more, and more precisely."
        )
    else:
        raise ValueError(
            "Nothing to search from. Either compute the model, or measure three "
            "control points by hand and the rest will be found automatically."
        )

    with rasterio.open(reference_path) as reference:
        # The matching grid is in control-frame metres, so the reference's
        # pixel size is measured there too -- a reference in another
        # projection, or in degrees, has a pixel size in other units.
        resolution = float(abs(reference.transform.a))
        if control_system is not None and reference.crs and reference.crs != control_system:
            from pyproj import Transformer

            to_control = Transformer.from_crs(reference.crs, control_system, always_xy=True)
            cx = reference.transform.c + reference.width / 2 * reference.transform.a
            cy = reference.transform.f + reference.height / 2 * reference.transform.e
            (x0, x1), (y0, y1) = to_control.transform(
                [cx, cx + reference.transform.a], [cy, cy])
            resolution = float(math.hypot(x1 - x0, y1 - y0))
        if resolution <= 0:
            raise ValueError("The reference image has no usable ground resolution")

        with rasterio.open(image["path"]) as photo:
            if progress:
                progress(0.05, "Finding candidate features")

            # -- interest points on the photograph -----------------------
            decimation = max(1.0, max(photo.width, photo.height) / 2200.0)
            band = 1 if photo.count < 3 else 2
            small = photo.read(
                band,
                out_shape=(int(photo.height / decimation), int(photo.width / decimation)),
                resampling=rasterio.enums.Resampling.average,
            ).astype(np.float32)

            view = _stretch(small)
            if view is None:
                raise ValueError("The photograph has no usable contrast")

            clip = image.get("clipRegion")
            mask = np.full(view.shape, 255, dtype=np.uint8)
            margin = max(4, int(options.edge_margin_px / decimation))
            mask[:margin, :] = 0
            mask[-margin:, :] = 0
            mask[:, :margin] = 0
            mask[:, -margin:] = 0
            if clip:
                c0, r0, c1, r1 = [v / decimation for v in clip]
                inside = np.zeros_like(mask)
                inside[int(r0):int(r1), int(c0):int(c1)] = 255
                mask = np.minimum(mask, inside)

            corners = cv2.goodFeaturesToTrack(
                view, maxCorners=options.candidate_pool, qualityLevel=0.02,
                minDistance=max(10, int(options.min_separation_m / resolution / decimation / 2)),
                mask=mask, blockSize=9,
            )
            if corners is None or len(corners) == 0:
                raise ValueError("No distinctive features found on the photograph")

            candidates = corners.reshape(-1, 2) * decimation

            # -- correlate each candidate against the reference ----------
            search_px = int(math.ceil(options.search_m / resolution))
            patch = options.patch_px
            proposals: list[GcpProposal] = []
            rejected_score = 0
            tested = 0

            for index, (col, row) in enumerate(candidates):
                if should_cancel and should_cancel():
                    raise InterruptedError("Cancelled")
                if len(proposals) >= options.target_count * 3:
                    break
                if progress and index % 10 == 0:
                    progress(
                        0.1 + 0.8 * index / len(candidates),
                        f"Correlating {index + 1} of {len(candidates)}",
                    )

                ground = pixel_to_ground(np.array([[col, row]]))[0]
                if not np.isfinite(ground[:2]).all():
                    continue

                tested += 1

                needle = _photo_patch_on_ground(
                    photo, ground_to_pixel, (ground[0], ground[1]), patch, resolution,
                    dem_sample, float(ground[2]) if np.isfinite(ground[2])
                    else options.background_elevation,
                )
                if needle is None:
                    continue
                needle_u8 = _stretch(needle)
                if needle_u8 is None or float(needle_u8.std()) < 12.0:
                    continue   # featureless patch: correlation would be noise

                haystack_size = patch + 2 * search_px
                haystack = _reference_patch(
                    reference, (ground[0], ground[1]), haystack_size, resolution,
                    control_system,
                )
                if haystack is None:
                    continue
                haystack_u8 = _stretch(haystack)
                if haystack_u8 is None or float(haystack_u8.std()) < 8.0:
                    continue

                response = cv2.matchTemplate(haystack_u8, needle_u8, cv2.TM_CCOEFF_NORMED)
                _, score, _, location = cv2.minMaxLoc(response)

                if score < options.min_score:
                    rejected_score += 1
                    continue

                # Subpixel refinement on the correlation surface. Without it a
                # match is quantised to whole reference pixels -- half a pixel
                # of bias, which is nothing on a 0.25 m mosaic and a metre on a
                # 2 m one. A parabola through the peak and its neighbours costs
                # nothing and removes it.
                mx, my = location
                sub_x = sub_y = 0.0
                if 0 < mx < response.shape[1] - 1:
                    a, b, c = response[my, mx - 1], response[my, mx], response[my, mx + 1]
                    denom = a - 2 * b + c
                    if abs(denom) > 1e-9:
                        sub_x = float(np.clip(0.5 * (a - c) / denom, -1.0, 1.0))
                if 0 < my < response.shape[0] - 1:
                    a, b, c = response[my - 1, mx], response[my, mx], response[my + 1, mx]
                    denom = a - 2 * b + c
                    if abs(denom) > 1e-9:
                        sub_y = float(np.clip(0.5 * (a - c) / denom, -1.0, 1.0))

                # Offset from the centre of the search window, in metres.
                centre_offset = (response.shape[1] - 1) / 2.0
                dx = (mx + sub_x - centre_offset) * resolution
                dy = -(my + sub_y - centre_offset) * resolution

                corrected_x = ground[0] + dx
                corrected_y = ground[1] + dy

                elevation = ground[2]
                if dem_sample is not None:
                    sampled = dem_sample([corrected_x], [corrected_y])[0]
                    if np.isfinite(sampled):
                        elevation = float(sampled)

                proposals.append(GcpProposal(
                    col=float(col), row=float(row),
                    x=float(corrected_x), y=float(corrected_y), z=float(elevation),
                    score=float(score), shift_m=float(math.hypot(dx, dy)),
                ))

    if not proposals:
        return AutoControlResult(
            [], tested, rejected_score, 0, 0.0, seed,
            message="No confident matches. The reference may not overlap this "
                    "photograph, or the sensor model may be too far out to search from.",
            warnings=warnings,
        )

    # -- reject matches inconsistent with the rest ------------------------
    #
    # A correct match's correction is close to its neighbours' corrections,
    # because the model is wrong smoothly. A mismatch lands anywhere.
    if progress:
        progress(0.92, "Filtering inconsistent matches")

    shifts = np.array([[p.x, p.y] for p in proposals]) - np.array(
        [pixel_to_ground(np.array([[p.col, p.row]]))[0][:2] for p in proposals]
    )
    median_shift = np.median(shifts, axis=0)
    deviation = np.linalg.norm(shifts - median_shift, axis=1)
    spread = np.median(deviation) * 1.4826

    if spread < 1e-6:
        keep = np.ones(len(proposals), dtype=bool)
    else:
        keep = deviation <= max(options.ransac_sigma * spread, resolution * 2)

    rejected_geometry = int((~keep).sum())
    surviving = [p for p, ok in zip(proposals, keep) if ok]

    # -- thin for spread, keeping the strongest ---------------------------
    surviving.sort(key=lambda p: p.score, reverse=True)
    chosen: list[GcpProposal] = []
    for candidate in surviving:
        if len(chosen) >= options.target_count:
            break
        if all(
            math.hypot(candidate.x - k.x, candidate.y - k.y) >= options.min_separation_m
            for k in chosen
        ):
            chosen.append(candidate)

    mean_shift = float(np.mean([p.shift_m for p in chosen])) if chosen else 0.0

    if progress:
        progress(1.0, f"Proposed {len(chosen)} control points")

    message = (
        f"{len(chosen)} control points proposed from {tested} candidates. "
        f"Mean correction {mean_shift:.1f} m."
    )
    if mean_shift > options.search_m * 0.7:
        warnings.append(
            f"Corrections average {mean_shift:.0f} m, close to the {options.search_m:.0f} m "
            "search limit. Some matches may have been cut off -- widen the search."
        )

    return AutoControlResult(
        proposals=chosen,
        tested=tested,
        rejected_score=rejected_score,
        rejected_geometry=rejected_geometry,
        mean_shift_m=mean_shift,
        seed=seed,
        message=message,
        warnings=warnings,
    )
