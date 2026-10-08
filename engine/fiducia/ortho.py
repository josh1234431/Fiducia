"""Orthorectification by tiled backward projection.

For every output pixel the resampler asks: what ground point is this, how high
is it, and where did that point land on the photo? That is a DEM lookup
followed by the collinearity condition, distortion, and interior orientation
in reverse.

Two design choices carry the performance:

**Tiles, not scanlines.** Work is split into square output tiles. Each tile
maps to a compact region of the source photo, so the worker reads one block and
resamples entirely in memory. A scanline-oriented resampler re-reads
overlapping source data for every row.

**Processes, not threads.** The inner loop is numpy and scipy, and the work is
embarrassingly parallel across tiles, so tiles are farmed to a process pool.
A twelve-core machine orthorectifies roughly ten times faster than the
single-threaded path, and the UI process is never blocked.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import rasterio
from rasterio.transform import Affine
from scipy.ndimage import map_coordinates

from .camera import CameraModel, FiducialFit
from .collinearity import rotation_matrix
from .geodesy import file_crs

__all__ = ["OrthoSpec", "OrthoResult", "compute_footprint", "generate_ortho"]

RESAMPLING_ORDER = {"nearest": 0, "bilinear": 1, "cubic": 3}


@dataclass
class OrthoSpec:
    """Everything needed to orthorectify one photo."""

    image_path: str
    output_path: str
    camera: dict                       # CameraModel.to_dict()
    fiducial_fit: Optional[dict]       # FiducialFit.to_dict(), film only
    exterior: Sequence[float]          # X0, Y0, Z0, omega, phi, kappa
    output_crs: str
    pixel_size_x: float
    pixel_size_y: float
    dem_path: Optional[str] = None
    background_elevation: Optional[float] = None
    elevation_scale: float = 1.0
    elevation_offset: float = 0.0
    resampling: str = "bilinear"
    clip_region: Optional[Sequence[float]] = None   # col_min, row_min, col_max, row_max
    bands: Optional[Sequence[int]] = None
    bounds: Optional[Sequence[float]] = None        # west, south, east, north
    nodata: float = 0.0
    # Keep real image values off the nodata value, so nodata only ever means
    # "outside the photograph". The same result as an explicit MIN=1 clamp
    # on the resampler.
    reserve_nodata: bool = True
    tile_size: int = 512
    compress: str = "DEFLATE"
    output_format: str = "GTiff"                    # see export.FORMATS
    jpeg_quality: int = 90
    author: str = ""                                # written into the file's metadata
    project_name: str = ""
    max_workers: int = 0                            # 0 = auto
    # The coordinate system the exterior orientation was solved in -- that of
    # the control. The collinearity equations only hold there. When it
    # differs from output_crs, every output cell is converted into it before
    # being projected into the photo. None means "same as output_crs".
    control_crs: Optional[str] = None
    # Atmospheric refraction and earth curvature, applied in reverse: the
    # collinearity equations give where a straight ray over flat ground
    # would land, and these move it to where the photo really recorded it.
    refraction: bool = False
    curvature: bool = False
    earth_radius_m: float = 6371000.0
    # Align cell edges to whole multiples of the pixel size.
    snap_to_grid: bool = True
    # Leave out ground the photo cannot see: the road under an overhanging
    # crown, the yard behind a wall. Without it those cells take the crown's
    # or the wall's pixels a second time, and the orthophoto folds. Needs a
    # surface model (trees and buildings in it), not a bare-earth DEM.
    occlusion: bool = False

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, (list, tuple)) else v)
                for k, v in self.__dict__.items()}


@dataclass
class OrthoResult:
    output_path: str
    width: int
    height: int
    bounds: tuple
    pixel_size: tuple
    band_count: int
    tiles_written: int
    valid_fraction: float
    elapsed_seconds: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "outputPath": self.output_path,
            "width": self.width,
            "height": self.height,
            "bounds": list(self.bounds),
            "pixelSize": list(self.pixel_size),
            "bandCount": self.band_count,
            "tilesWritten": self.tiles_written,
            "validFraction": self.valid_fraction,
            "elapsedSeconds": self.elapsed_seconds,
            "warnings": self.warnings,
        }


def _image_corners(spec: OrthoSpec, width: int, height: int, per_edge: int = 8) -> np.ndarray:
    """Points around the frame edge, honouring the clip region if one is set.

    More than the four corners: over uneven ground, and once converted into
    another projection, the frame's edges are not straight lines on the
    ground, and a bounding box from the corners alone can clip the image.
    """
    if spec.clip_region:
        c0, r0, c1, r1 = spec.clip_region
    else:
        c0, r0, c1, r1 = 0.0, 0.0, float(width - 1), float(height - 1)
    steps = np.linspace(0.0, 1.0, per_edge, endpoint=False)
    top = [(c0 + (c1 - c0) * s, r0) for s in steps]
    right = [(c1, r0 + (r1 - r0) * s) for s in steps]
    bottom = [(c1 - (c1 - c0) * s, r1) for s in steps]
    left = [(c0, r1 - (r1 - r0) * s) for s in steps]
    return np.array(top + right + bottom + left, dtype=float)


def _converter(source: Optional[str], target: Optional[str]):
    """A function converting (x, y) arrays between two systems, or None.

    None when either is unknown or they are the same system: the identity is
    skipped entirely, which is both faster and exactly the old behaviour.
    """
    if not source or not target:
        return None
    from .geodesy import make_transformer

    transformer = make_transformer(source, target)
    if transformer is None:
        return None

    def convert(xs, ys):
        out_x, out_y = transformer.transform(np.asarray(xs, float), np.asarray(ys, float))
        return np.asarray(out_x, float), np.asarray(out_y, float)
    return convert


def _dem_crs(dem_path: Optional[str]) -> Optional[str]:
    if not dem_path:
        return None
    try:
        with rasterio.open(dem_path) as dem:
            return dem.crs.to_wkt() if dem.crs else None
    except Exception:  # noqa: BLE001
        return None


def _pixels_to_ground(
    pixels: np.ndarray,
    camera: CameraModel,
    fit: Optional[FiducialFit],
    eo: np.ndarray,
    elevation: np.ndarray | float,
) -> np.ndarray:
    """Intersect image rays with a horizontal plane at the given elevation."""
    film = camera.pixel_to_film(pixels, fit)
    rot = rotation_matrix(eo[3], eo[4], eo[5])

    directions = np.column_stack(
        [film[:, 0], film[:, 1], np.full(film.shape[0], -camera.focal_mm)]
    ) @ rot                       # R^T applied as row-vector multiply

    z = np.asarray(elevation, dtype=float)
    if z.ndim == 0:
        z = np.full(film.shape[0], float(z))

    with np.errstate(divide="ignore", invalid="ignore"):
        t = (z - eo[2]) / directions[:, 2]

    ground = np.column_stack(
        [eo[0] + t * directions[:, 0], eo[1] + t * directions[:, 1], z]
    )
    ground[~np.isfinite(t)] = np.nan
    return ground


def _ground_to_pixels(
    ground: np.ndarray,
    camera: CameraModel,
    fit: Optional[FiducialFit],
    eo: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project ground points into a photo: (pixels, in front of the camera).

    The same collinearity as orthorectification uses, without the refraction
    and curvature terms, which move a point by a small fraction of a pixel
    and do not matter for placing a view.
    """
    rot = rotation_matrix(eo[3], eo[4], eo[5])
    delta = np.asarray(ground, dtype=float) - eo[:3]
    u, v, w = (delta @ rot.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        film = np.column_stack([-camera.focal_mm * u / w, -camera.focal_mm * v / w])
    ahead = np.isfinite(film).all(axis=1) & (w < 0)
    film = np.nan_to_num(film, nan=0.0, posinf=0.0, neginf=0.0)
    return camera.film_to_pixel(film, fit), ahead


def transfer_pixel(
    col: float,
    row: float,
    source: dict,
    target: dict,
    camera: CameraModel,
    height_at=None,
    fallback_height: float = 0.0,
    iterations: int = 4,
) -> dict:
    """Where a pixel of one photo falls on another, through the ground.

    ``source`` and ``target`` carry ``exterior`` and ``fiducialFit``. The
    pixel's ray is intersected with the ground at ``fallback_height``, then,
    when ``height_at(x, y)`` gives the terrain height, re-intersected at the
    height found there until it settles. A second pixel a little to the right
    gives the local scale between the two photos, so a view can show the same
    patch of ground at the same size.
    """
    def fit_of(image):
        return FiducialFit.from_dict(image["fiducialFit"]) if image.get("fiducialFit") else None

    eo_a = np.asarray(source["exterior"], dtype=float)
    eo_b = np.asarray(target["exterior"], dtype=float)
    fit_a, fit_b = fit_of(source), fit_of(target)
    pixels = np.array([[col, row], [col + 100.0, row]], dtype=float)

    heights = np.full(2, float(fallback_height))
    ground = _pixels_to_ground(pixels, camera, fit_a, eo_a, heights)
    from_terrain = False
    if height_at is not None:
        for _ in range(iterations):
            if not np.isfinite(ground).all():
                break
            sampled = np.asarray(height_at(ground[:, 0], ground[:, 1]), dtype=float)
            if not np.isfinite(sampled).all():
                break
            from_terrain = True
            if np.allclose(sampled, heights, atol=0.05):
                break
            heights = sampled
            ground = _pixels_to_ground(pixels, camera, fit_a, eo_a, heights)

    if not np.isfinite(ground).all():
        raise ValueError("The centre of the view does not look at the ground.")

    projected, ahead = _ground_to_pixels(ground, camera, fit_b, eo_b)
    spread = float(np.hypot(*(projected[1] - projected[0])))
    return {
        "ground": [float(v) for v in ground[0]],
        "fromTerrain": from_terrain,
        "col": float(projected[0, 0]),
        "row": float(projected[0, 1]),
        "ahead": bool(ahead[0]),
        # Target pixels per source pixel, around this point.
        "scaleRatio": spread / 100.0 if spread > 0 else 1.0,
    }


def compute_footprint(spec: OrthoSpec, iterations: int = 3) -> tuple:
    """Ground bounding box of the photo, refined against the DEM.

    Starts from a flat plane at the mean DEM elevation, then re-intersects the
    corner rays at the elevation actually found there. Terrain of a few hundred
    metres of relief shifts a corner by tens of metres, which matters when the
    output grid is being sized to it.
    """
    from . import raster

    with rasterio.open(spec.image_path) as dataset:
        width, height = dataset.width, dataset.height

    camera = CameraModel.from_dict(spec.camera)
    fit = FiducialFit.from_dict(spec.fiducial_fit) if spec.fiducial_fit else None
    eo = np.asarray(spec.exterior, dtype=float)
    corners = _image_corners(spec, width, height)

    elevation = float(spec.background_elevation or 0.0)
    if spec.dem_path:
        try:
            with rasterio.open(spec.dem_path) as dem:
                band = dem.read(1, out_shape=(64, 64)).astype(float)
                if dem.nodata is not None:
                    band = band[band != dem.nodata]
                band = band[np.isfinite(band)]
                if band.size:
                    elevation = float(np.mean(band))
        except Exception:
            pass

    ground = _pixels_to_ground(corners, camera, fit, eo, elevation)

    control_crs = spec.control_crs or spec.output_crs
    dem_crs = _dem_crs(spec.dem_path)
    to_dem = _converter(control_crs, dem_crs)
    for attempt in range(iterations - 1):
        if not spec.dem_path or not np.isfinite(ground).all():
            break
        try:
            gx, gy = ground[:, 0], ground[:, 1]
            if to_dem:
                gx, gy = to_dem(gx, gy)
            sampled = raster.sample_elevation(spec.dem_path, gx, gy)
            centre = raster.sample_elevation(spec.dem_path, [np.mean(gx)], [np.mean(gy)])
        except Exception:
            break
        # A DEM that misses the whole footprint means the control projection
        # and the DEM disagree about where the ground is. Carrying on would
        # quietly orthorectify onto a flat plane, so stop and say why.
        if attempt == 0 and not np.isfinite(sampled).any() and not np.isfinite(centre).any():
            raise ValueError(
                "The reference DEM does not cover this image in the control "
                "projection. Check that the projection in the Project step "
                "matches the DEM and the control coordinates."
            )
        sampled = np.where(np.isfinite(sampled), sampled, elevation)
        sampled = sampled * spec.elevation_scale + spec.elevation_offset
        ground = _pixels_to_ground(corners, camera, fit, eo, sampled)

    finite = ground[np.isfinite(ground).all(axis=1)]
    if finite.size == 0:
        raise ValueError(
            "Could not project the image footprint onto the ground -- "
            "the exterior orientation is probably not yet solved"
        )

    xs, ys = finite[:, 0], finite[:, 1]
    to_output = _converter(control_crs, spec.output_crs)
    if to_output:
        xs, ys = to_output(xs, ys)
    return (float(np.min(xs)), float(np.min(ys)), float(np.max(xs)), float(np.max(ys)))


# -- worker ---------------------------------------------------------------
#
# Kept at module level and free of closures so it pickles cleanly under the
# Windows spawn start method.

_WORKER_CACHE: dict = {}
_DEPTH_CACHE: dict = {}   # occlusion depth buffers, by path

# How long a pool may go without finishing a single tile before it is judged
# stuck -- workers that died on start-up, or a machine paging itself to a
# standstill -- and the job continues in one process instead.
STALL_SECONDS = 180

# Stop handing out tiles below this much free memory; abandon the pool for a
# single process below the second; stop the job below the third rather than
# let Windows start killing programs.
LOW_WATER_BYTES = int(1.5 * 1024 ** 3)
PARALLEL_FLOOR_BYTES = int(1.0 * 1024 ** 3)
SERIAL_FLOOR_BYTES = int(0.4 * 1024 ** 3)


class LowMemory(RuntimeError):
    """The job stopped itself to keep the machine usable."""

    for_operator = True


def _worker_init(cache_mb: int) -> None:
    """Runs once in each worker, before it reads anything."""
    from . import memory_budget

    memory_budget.limit_raster_cache(cache_mb)


def _stop_pool(pool: ProcessPoolExecutor) -> None:
    """Shut a pool down now, including workers still mid-tile.

    ``shutdown(wait=False)`` alone lets running workers finish, and a worker
    that is the reason for stopping -- stuck, or holding memory the machine
    needs -- must not be left to finish anything.
    """
    processes = list(getattr(pool, "_processes", {}).values())
    try:
        pool.shutdown(wait=False, cancel_futures=True)
    except Exception:  # noqa: BLE001
        pass
    for process in processes:
        try:
            if process.is_alive():
                process.terminate()
        except Exception:  # noqa: BLE001
            pass
    for process in processes:
        try:
            process.join(timeout=5)
        except Exception:  # noqa: BLE001
            pass


def _release_worker_cache() -> None:
    """Close cached datasets held by the in-process tile renderer.

    In a worker process this is unnecessary -- the process exits and takes its
    handles with it. In the single-worker path the renderer runs in the
    engine's own process, and a dataset left open there means Windows refuses
    to let the operator move, rename or delete their own source image until
    Fiducia quits.
    """
    for state in _WORKER_CACHE.values():
        for handle in state[2:4]:
            if handle is not None and not handle.closed:
                try:
                    handle.close()
                except Exception:
                    pass
    _WORKER_CACHE.clear()
    _DEPTH_CACHE.clear()


def _worker_state(spec_dict: dict):
    # The coordinate systems are part of the key: the same photo rendered into
    # another projection in the same process must not reuse old conversions.
    key = (spec_dict["image_path"], spec_dict.get("dem_path"), spec_dict.get("output_crs"),
           spec_dict.get("control_crs"), spec_dict.get("dem_crs"))
    cached = _WORKER_CACHE.get(key)
    if cached is not None:
        return cached

    camera = CameraModel.from_dict(spec_dict["camera"])
    fit = FiducialFit.from_dict(spec_dict["fiducial_fit"]) if spec_dict.get("fiducial_fit") else None
    source = rasterio.open(spec_dict["image_path"])
    dem = rasterio.open(spec_dict["dem_path"]) if spec_dict.get("dem_path") else None
    output_crs = spec_dict.get("output_crs")
    to_control = _converter(output_crs, spec_dict.get("control_crs") or output_crs)
    # A DEM with no coordinate system is taken to share the control's.
    to_dem = _converter(output_crs, spec_dict.get("dem_crs")
                        or spec_dict.get("control_crs") or output_crs)
    state = (camera, fit, source, dem, to_control, to_dem)
    _WORKER_CACHE[key] = state
    return state


def _sample_dem_block(dem, xs: np.ndarray, ys: np.ndarray, fallback: float):
    """Bilinear DEM sampling over a tile, reading one padded block.

    Returns ``(heights, missing)``: ``missing`` marks cells with no DEM data
    under them, which were given the fallback height instead.
    """
    if dem is None:
        return np.full(xs.shape, fallback, dtype=float), np.ones(xs.shape, dtype=bool)

    inverse = ~dem.transform
    cols, rows = inverse * (xs.ravel(), ys.ravel())
    cols = np.asarray(cols, dtype=float) - 0.5
    rows = np.asarray(rows, dtype=float) - 0.5

    finite = np.isfinite(cols) & np.isfinite(rows)
    if not finite.any():
        return np.full(xs.shape, fallback, dtype=float), np.ones(xs.shape, dtype=bool)

    pad = 2
    col_lo = int(np.floor(np.min(cols[finite]))) - pad
    row_lo = int(np.floor(np.min(rows[finite]))) - pad
    col_hi = int(np.ceil(np.max(cols[finite]))) + pad
    row_hi = int(np.ceil(np.max(rows[finite]))) + pad

    from rasterio.windows import Window

    window = Window(col_lo, row_lo, max(1, col_hi - col_lo), max(1, row_hi - row_lo))
    block = dem.read(1, window=window, boundless=True, fill_value=np.nan).astype(np.float32)
    if dem.nodata is not None:
        block[block == dem.nodata] = np.nan

    local = np.vstack([rows - row_lo, cols - col_lo])
    values = map_coordinates(
        np.nan_to_num(block, nan=fallback), local, order=1, mode="nearest"
    )
    missing = map_coordinates(np.isnan(block).astype(np.float32), local, order=1,
                              mode="constant", cval=1.0) > 0.5
    return values.reshape(xs.shape).astype(float), missing.reshape(xs.shape)


def hidden_sidecar(ortho_path: str) -> str:
    """The mask beside an ortho of the pixels its photo could not see."""
    return str(ortho_path) + ".hidden.tif"


def _depth_buffer(spec: OrthoSpec, bounds: tuple, work_path: str) -> Optional[dict]:
    """The nearest surface along every line of sight of one photo.

    Every cell of the surface model is projected into the photo and the
    nearest depth kept per buffer pixel. A ground cell further from the camera
    than that, by more than the surface's own coarseness explains, is behind
    something. The buffer is coarser than the photo -- about one surface cell
    per buffer pixel -- since it can be no sharper than the surface.
    """
    from rasterio.windows import from_bounds

    if not spec.dem_path:
        return None
    camera = CameraModel.from_dict(spec.camera)
    fit = FiducialFit.from_dict(spec.fiducial_fit) if spec.fiducial_fit else None
    eo = np.asarray(spec.exterior, dtype=float)
    control_crs = spec.control_crs or spec.output_crs
    dem_crs = _dem_crs(spec.dem_path) or control_crs
    to_dem = _converter(spec.output_crs, dem_crs)
    to_control = _converter(dem_crs, control_crs)

    west, south, east, north = bounds
    corners_x, corners_y = np.array([west, east, east, west]), np.array([south, south, north, north])
    if to_dem:
        corners_x, corners_y = to_dem(corners_x, corners_y)
    with rasterio.open(spec.dem_path) as dem:
        window = from_bounds(float(np.min(corners_x)), float(np.min(corners_y)),
                             float(np.max(corners_x)), float(np.max(corners_y)), dem.transform)
        window = window.round_offsets().round_lengths()
        z = dem.read(1, window=window, boundless=True, fill_value=np.nan).astype(np.float64)
        if dem.nodata is not None:
            z[z == dem.nodata] = np.nan
        t = dem.window_transform(window)
        cell = float(abs(dem.res[0]))
    if not np.isfinite(z).any():
        return None
    rows, cols = np.nonzero(np.isfinite(z))
    xs, ys = t * (cols + 0.5, rows + 0.5)
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    zs = z[rows, cols] * spec.elevation_scale + spec.elevation_offset
    if to_control:
        xs, ys = to_control(xs, ys)

    rot = rotation_matrix(eo[3], eo[4], eo[5])
    d = np.column_stack([xs - eo[0], ys - eo[1], zs - eo[2]])
    cam = d @ rot.T
    depth = -cam[:, 2]
    ahead = depth > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        film = np.column_stack([-camera.focal_mm * cam[:, 0] / cam[:, 2],
                                -camera.focal_mm * cam[:, 1] / cam[:, 2]])
    film = np.nan_to_num(film)
    pixels = camera.film_to_pixel(film, fit)

    with rasterio.open(spec.image_path) as source:
        src_w, src_h = source.width, source.height
    focal_px = camera.focal_mm / max(camera.pixel_scale_mm(fit), 1e-12)
    height = float(np.median(depth[ahead])) if ahead.any() else 1.0
    scale = max(1, int(0.8 * cell * focal_px / max(height, 1e-6)))
    shape = (int(math.ceil(src_h / scale)), int(math.ceil(src_w / scale)))
    bc = np.floor(pixels[:, 0] / scale).astype(np.int64)
    br = np.floor(pixels[:, 1] / scale).astype(np.int64)
    keep = ahead & (bc >= 0) & (bc < shape[1]) & (br >= 0) & (br < shape[0])
    buffer = np.full(shape, np.inf, dtype=np.float32)
    np.minimum.at(buffer, (br[keep], bc[keep]), depth[keep].astype(np.float32))
    np.save(work_path, buffer)
    # Steep ground within one buffer pixel differs in depth by up to its
    # slope across a few cells; that is not occlusion.
    return {"path": work_path, "scale": scale, "tolerance": 0.3 + 2.5 * cell}


def _render_tile(args) -> tuple:
    """Resample one output tile. Runs in a worker process."""
    spec_dict, window_info = args
    col_off, row_off, tile_w, tile_h = window_info

    camera, fit, source, dem, to_control, to_dem = _worker_state(spec_dict)
    eo = np.asarray(spec_dict["exterior"], dtype=float)
    transform = Affine(*spec_dict["output_transform"])
    bands = spec_dict["band_list"]
    order = RESAMPLING_ORDER.get(spec_dict["resampling"], 1)
    nodata = spec_dict["nodata"]
    fallback = float(spec_dict.get("background_elevation") or 0.0)

    # Output pixel centres -> map coordinates.
    cols = np.arange(col_off, col_off + tile_w, dtype=float) + 0.5
    rows = np.arange(row_off, row_off + tile_h, dtype=float) + 0.5
    grid_c, grid_r = np.meshgrid(cols, rows)
    xs, ys = transform * (grid_c, grid_r)
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    # The output cell in the DEM's system for its height, and in the
    # control's system for the collinearity equations.
    if to_dem:
        dem_x, dem_y = to_dem(xs.ravel(), ys.ravel())
        dem_x, dem_y = dem_x.reshape(xs.shape), dem_y.reshape(xs.shape)
    else:
        dem_x, dem_y = xs, ys
    zs, no_dem = _sample_dem_block(dem, dem_x, dem_y, fallback)
    zs = zs * spec_dict.get("elevation_scale", 1.0) + spec_dict.get("elevation_offset", 0.0)

    if to_control:
        control_x, control_y = to_control(xs.ravel(), ys.ravel())
        control_x, control_y = control_x.reshape(xs.shape), control_y.reshape(xs.shape)
    else:
        control_x, control_y = xs, ys

    # Collinearity: ground -> camera -> focal plane.
    rot = rotation_matrix(eo[3], eo[4], eo[5])
    dx = control_x - eo[0]
    dy = control_y - eo[1]
    dz = zs - eo[2]

    u = rot[0, 0] * dx + rot[0, 1] * dy + rot[0, 2] * dz
    v = rot[1, 0] * dx + rot[1, 1] * dy + rot[1, 2] * dz
    w = rot[2, 0] * dx + rot[2, 1] * dy + rot[2, 2] * dz

    with np.errstate(divide="ignore", invalid="ignore"):
        film_x = -camera.focal_mm * u / w
        film_y = -camera.focal_mm * v / w

    behind = ~np.isfinite(film_x) | ~np.isfinite(film_y) | (w >= 0)

    film = np.column_stack([film_x.ravel(), film_y.ravel()])
    film = np.nan_to_num(film, nan=0.0, posinf=0.0, neginf=0.0)
    if spec_dict.get("refraction") or spec_dict.get("curvature"):
        from .corrections import from_flat

        film = from_flat(film, camera.focal_mm, float(eo[2]), zs.ravel(),
                         bool(spec_dict.get("refraction")), bool(spec_dict.get("curvature")),
                         float(spec_dict.get("earth_radius_m") or 6371000.0))
    pixels = camera.film_to_pixel(film, fit)
    src_c = pixels[:, 0].reshape(tile_h, tile_w)
    src_r = pixels[:, 1].reshape(tile_h, tile_w)

    clip = spec_dict.get("clip_region")
    if clip:
        c0, r0, c1, r1 = clip
    else:
        c0, r0, c1, r1 = 0.0, 0.0, float(source.width - 1), float(source.height - 1)

    inside = (
        np.isfinite(src_c) & np.isfinite(src_r)
        & (src_c >= c0) & (src_c <= c1) & (src_r >= r0) & (src_r <= r1)
        & ~behind
    )

    occlusion = spec_dict.get("depth_buffer")
    if occlusion and inside.any():
        buffer = _DEPTH_CACHE.get(occlusion["path"])
        if buffer is None:
            buffer = np.load(occlusion["path"])
            _DEPTH_CACHE[occlusion["path"]] = buffer
        k = occlusion["scale"]
        br = np.clip(np.floor(np.where(inside, src_r, 0) / k).astype(np.int64), 0, buffer.shape[0] - 1)
        bc = np.clip(np.floor(np.where(inside, src_c, 0) / k).astype(np.int64), 0, buffer.shape[1] - 1)
        hidden = inside & (-w > buffer[br, bc] + occlusion["tolerance"])
    else:
        hidden = None

    out = np.full((len(bands), tile_h, tile_w), nodata, dtype=np.float32)
    if not inside.any():
        return window_info, out.astype(spec_dict["out_dtype"]), 0, 0, None

    # Read only the source block this tile touches, with margin for the
    # interpolation kernel.
    margin = 4
    block_c0 = max(0, int(np.floor(np.min(src_c[inside]))) - margin)
    block_r0 = max(0, int(np.floor(np.min(src_r[inside]))) - margin)
    block_c1 = min(source.width, int(np.ceil(np.max(src_c[inside]))) + margin + 1)
    block_r1 = min(source.height, int(np.ceil(np.max(src_r[inside]))) + margin + 1)

    if block_c1 <= block_c0 or block_r1 <= block_r0:
        return window_info, out.astype(spec_dict["out_dtype"]), 0, 0, None

    from rasterio.windows import Window

    read_window = Window(block_c0, block_r0, block_c1 - block_c0, block_r1 - block_r0)
    block = source.read(bands, window=read_window).astype(np.float32)

    local = np.vstack(
        [
            np.where(inside, src_r - block_r0, 0).ravel(),
            np.where(inside, src_c - block_c0, 0).ravel(),
        ]
    )

    for i in range(len(bands)):
        # "nearest" extends the block's edge rather than padding it with the
        # nodata value: a cubic kernel reaches past the samples it straddles,
        # and padding would pull genuine pixels near a block edge towards
        # black. Cells outside the photo are masked by ``inside`` regardless.
        sampled = map_coordinates(block[i], local, order=order, mode="nearest")
        layer = sampled.reshape(tile_h, tile_w)
        out[i] = np.where(inside, layer, nodata)

    out = _to_output_values(out, inside, nodata, np.dtype(spec_dict["out_dtype"]),
                            reserve=spec_dict.get("reserve_nodata", True))

    return (window_info, out, int(inside.sum()),
            int((inside & no_dem).sum()) if spec_dict.get("dem_path") else 0,
            hidden.astype(np.uint8) if hidden is not None else None)


def _background_for(value: float, dtype: np.dtype):
    """The background value, brought into the range the output type can hold.

    A value the data type cannot represent is truncated to
    its range rather than wrapping (300 in 8-bit would otherwise become 44).
    """
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return int(min(max(round(float(value)), info.min), info.max))
    return float(value)


def _to_output_values(out: np.ndarray, inside: np.ndarray, nodata, dtype: np.dtype,
                      reserve: bool = True) -> np.ndarray:
    """Convert resampled values to the output type without corrupting them.

    Integer output is rounded and clamped: a plain cast truncates, and a
    cubic kernel's slight undershoot below zero would wrap round to 255 and
    show as a white speck.

    The nodata value must then mean "outside the photograph" and nothing
    else. Dark water and deep shadow often reach 0 in one band, and GIS
    software hides a pixel when any band equals nodata, so those real pixels
    appeared as holes. They are moved one step away from nodata, a change
    far below what the eye or any analysis can see.
    """
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        out = np.clip(np.rint(out), info.min, info.max)
    if reserve and nodata is not None:
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            replacement = nodata + 1 if nodata < info.max else nodata - 1
        else:
            replacement = np.nextafter(dtype.type(nodata), dtype.type(np.inf))
        for band in out:
            band[inside & (band == nodata)] = replacement
    return out.astype(dtype)


def generate_ortho(
    spec: OrthoSpec,
    progress: Optional[Callable[[float, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> OrthoResult:
    """Produce one orthorectified GeoTIFF.

    ``progress`` receives a 0-1 fraction and a status line. ``should_cancel``
    is polled between tiles so a long run can be abandoned promptly.
    """
    import time

    started = time.time()
    warnings: list[str] = []

    with rasterio.open(spec.image_path) as dataset:
        band_list = list(spec.bands) if spec.bands else list(range(1, dataset.count + 1))
        band_list = [b for b in band_list if 1 <= b <= dataset.count] or [1]
        src_dtype = dataset.dtypes[0]

    bounds = tuple(spec.bounds) if spec.bounds else compute_footprint(spec)
    west, south, east, north = bounds
    if spec.snap_to_grid and all(math.isfinite(v) for v in bounds):
        # Cell edges on whole multiples of the pixel size: every ortho of the
        # project then shares one grid, and a mosaic of them takes each pixel
        # as it is instead of resampling it by a fraction of a cell.
        px, py = spec.pixel_size_x, spec.pixel_size_y
        west, east = math.floor(west / px) * px, math.ceil(east / px) * px
        south, north = math.floor(south / py) * py, math.ceil(north / py) * py
        bounds = (west, south, east, north)

    if not all(math.isfinite(v) for v in bounds) or east <= west or north <= south:
        raise ValueError(f"Degenerate output footprint {bounds}")

    width = int(math.ceil((east - west) / spec.pixel_size_x))
    height = int(math.ceil((north - south) / spec.pixel_size_y))

    if width <= 0 or height <= 0:
        raise ValueError("Output grid has zero extent -- check the pixel spacing")

    # A photo can only ever cover a bounded patch of ground. A footprint far
    # larger than the source frame means the exterior orientation is wrong, and
    # without this guard the resampler would grind through millions of tiles
    # producing nothing usable. Refusing quickly, with the reason, is the whole
    # difference between a bad solve costing seconds and costing an afternoon.
    with rasterio.open(spec.image_path) as dataset:
        source_pixels = dataset.width * dataset.height
    output_pixels = width * height

    if output_pixels > max(source_pixels * 400, 400_000_000):
        raise ValueError(
            f"The computed footprint would produce a {width:,} x {height:,} pixel "
            f"image from a {source_pixels:,} pixel source. That is not a plausible "
            "orthophoto -- the exterior orientation, the output projection or the "
            "pixel spacing is wrong. Check the model residuals before regenerating."
        )

    transform = Affine(spec.pixel_size_x, 0.0, west, 0.0, -spec.pixel_size_y, north)
    nodata = _background_for(spec.nodata, np.dtype(src_dtype))

    spec_dict = {
        "image_path": spec.image_path,
        "dem_path": spec.dem_path,
        "camera": spec.camera,
        "fiducial_fit": spec.fiducial_fit,
        "exterior": list(spec.exterior),
        "output_transform": tuple(transform)[:6],
        "band_list": band_list,
        "resampling": spec.resampling,
        "nodata": nodata,
        "reserve_nodata": bool(spec.reserve_nodata),
        "clip_region": list(spec.clip_region) if spec.clip_region else None,
        "background_elevation": spec.background_elevation,
        "elevation_scale": spec.elevation_scale,
        "elevation_offset": spec.elevation_offset,
        "out_dtype": src_dtype,
        "output_crs": spec.output_crs,
        "control_crs": spec.control_crs or spec.output_crs,
        "dem_crs": _dem_crs(spec.dem_path),
        "refraction": bool(spec.refraction),
        "curvature": bool(spec.curvature),
        "earth_radius_m": float(spec.earth_radius_m),
    }
    # Which pixels of this ortho show something other than their ground (it
    # was hidden behind a tree or a building), beside the ortho for the
    # mosaic. Any older one goes: it described a different rendering.
    hidden_path = hidden_sidecar(spec.output_path)
    Path(hidden_path).unlink(missing_ok=True)
    hidden_sink = None
    hidden_count = 0

    def write_hidden(hidden, c, r, w, h):
        nonlocal hidden_count
        if hidden_sink is not None and hidden is not None:
            hidden_sink.write(hidden[None], window=Window(c, r, w, h))
            hidden_count += int(hidden.sum())

    depth_path = None
    if spec.occlusion and spec.dem_path:
        if progress:
            progress(0.02, "Finding what the photo cannot see")
        depth_path = spec.output_path + ".depth.npy"
        try:
            spec_dict["depth_buffer"] = _depth_buffer(spec, bounds, depth_path)
        except Exception as exc:  # noqa: BLE001 -- an ortho without it is still an ortho
            warnings.append(f"Hidden ground could not be worked out ({exc}); none was left out.")

    tile = max(128, int(spec.tile_size))
    windows = [
        (c, r, min(tile, width - c), min(tile, height - r))
        for r in range(0, height, tile)
        for c in range(0, width, tile)
    ]

    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": len(band_list),
        "dtype": src_dtype,
        "crs": file_crs(spec.output_crs),
        "transform": transform,
        "nodata": nodata,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": spec.compress,
        "BIGTIFF": "IF_SAFER",
    }

    # Rendering always writes a GeoTIFF, the one format GDAL writes well tile
    # by tile. Any other format, or lossy compression, is produced from that
    # working file afterwards and the working file removed.
    from . import export

    out_format, out_compress = export.normalise(spec.output_format, spec.compress)
    direct = export.writes_directly(out_format, out_compress)
    working_path = spec.output_path if direct else spec.output_path + ".working.tif"
    profile["compress"] = out_compress if direct else "DEFLATE"

    Path(spec.output_path).parent.mkdir(parents=True, exist_ok=True)

    from . import memory_budget

    # Sized by free memory as well as cores. Each worker holds its own copy
    # of the interpreter, the libraries and a raster cache, so on a machine
    # with many cores and little free memory, one per core is how a big photo
    # takes the whole computer down.
    plan = memory_budget.plan_workers(len(windows), spec.max_workers)
    workers = plan.workers
    valid_pixels = 0
    outside_dem = 0
    written = 0
    cancelled = False

    from rasterio.windows import Window

    def gb(value: int) -> str:
        return f"{value / 1024 ** 3:.1f} GB"

    def render_serially(sink, done_already: set, note: str = "single process") -> None:
        """In-process fallback. Slower, but the lightest way to finish."""
        nonlocal valid_pixels, written, cancelled, outside_dem
        remaining = [w for w in windows if w not in done_already]
        for index, window_info in enumerate(remaining):
            if should_cancel and should_cancel():
                cancelled = True
                return
            if index % 16 == 0:
                free = memory_budget.available_bytes()
                if free < SERIAL_FLOOR_BYTES:
                    raise LowMemory(
                        f"Stopped to keep your computer usable: only {gb(free)} of memory "
                        "is free. Close other large programs (ArcGIS, browser tabs) and "
                        "generate the orthophoto again."
                    )
            _, data, count, off_dem, hidden = _render_tile((spec_dict, window_info))
            c, r, w, h = window_info
            sink.write(data, window=Window(c, r, w, h))
            write_hidden(hidden, c, r, w, h)
            valid_pixels += count
            outside_dem += off_dem
            written += 1
            if progress:
                progress(
                    (len(done_already) + index + 1) / len(windows),
                    f"Tile {len(done_already) + index + 1} of {len(windows)} ({note})",
                )

    if spec_dict.get("depth_buffer"):
        hidden_sink = rasterio.open(hidden_path, "w", driver="GTiff", width=width, height=height,
                                    count=1, dtype="uint8", crs=profile["crs"], transform=transform,
                                    nodata=None, tiled=True, blockxsize=256, blockysize=256,
                                    compress="DEFLATE", NBITS=1)
    try:
        with rasterio.open(working_path, "w", **profile) as sink:
            if workers <= 1 or len(windows) == 1:
                render_serially(
                    sink, set(),
                    "single process" if plan.workers >= 1 and "memory" not in plan.reason
                    else f"single process, {plan.reason}",
                )
            else:
                completed: set = set()
                fallback_reason = None
                memory_budget.single_threaded_children()
                pool = ProcessPoolExecutor(
                    max_workers=workers, initializer=_worker_init,
                    initargs=(memory_budget.WORKER_CACHE_MB,),
                )
                try:
                    # A bounded number of tiles in flight: results waiting to be
                    # written cost memory too, and a queue of every tile at once is
                    # thousands of them.
                    queue = iter(windows)
                    pending: dict = {}
                    exhausted = False
                    while True:
                        while not exhausted and len(pending) < 2 * workers:
                            if memory_budget.available_bytes() < LOW_WATER_BYTES:
                                break
                            window_info = next(queue, None)
                            if window_info is None:
                                exhausted = True
                                break
                            pending[pool.submit(_render_tile, (spec_dict, window_info))] = window_info

                        if not pending:
                            if exhausted:
                                break
                            fallback_reason = (f"free memory fell to "
                                               f"{gb(memory_budget.available_bytes())}")
                            break

                        done, _ = wait(pending, timeout=STALL_SECONDS, return_when=FIRST_COMPLETED)
                        if should_cancel and should_cancel():
                            cancelled = True
                            break
                        if not done:
                            fallback_reason = f"no tile finished in {STALL_SECONDS} s"
                            break

                        for future in done:
                            pending.pop(future)
                            window_info, data, count, off_dem, hidden = future.result()
                            c, r, w, h = window_info
                            sink.write(data, window=Window(c, r, w, h))
                            write_hidden(hidden, c, r, w, h)
                            completed.add(window_info)
                            valid_pixels += count
                            outside_dem += off_dem
                            written += 1

                        if progress:
                            progress(written / len(windows),
                                     f"Tile {written} of {len(windows)} ({plan.describe()})")

                        free = memory_budget.available_bytes()
                        if free < PARALLEL_FLOOR_BYTES:
                            fallback_reason = f"free memory fell to {gb(free)}"
                            break
                except BrokenProcessPool as exc:
                    # A worker pool can fail to start for reasons that have nothing
                    # to do with this job -- a Python environment the child cannot
                    # reconstruct, or a system process limit.
                    fallback_reason = f"a worker process stopped ({exc})"
                finally:
                    # Workers are released before any fallback starts, so the
                    # memory they held is back for the single process.
                    _stop_pool(pool)

                if fallback_reason and not cancelled:
                    warnings.append(
                        f"Parallel rendering stopped ({fallback_reason}); "
                        "finished in a single process."
                    )
                    if progress:
                        progress(len(completed) / len(windows),
                                 f"Continuing in one process: {fallback_reason}")
                    render_serially(sink, completed, "single process")

    except BaseException:
        # A half-written orthophoto looks like a result and is not one.
        _release_worker_cache()
        Path(working_path).unlink(missing_ok=True)
        raise

    # Release the in-process source handles before anything else touches the
    # files, so a cancelled run leaves nothing locked either.
    _release_worker_cache()

    if not cancelled:
        from . import export as _export

        _export.write_provenance(working_path, _export.provenance(
            author=spec.author, project=spec.project_name, source=Path(spec.image_path).name,
        ))

    if cancelled:
        Path(working_path).unlink(missing_ok=True)
        raise InterruptedError("Ortho generation cancelled")

    valid_fraction = valid_pixels / float(width * height) if width * height else 0.0
    if spec.dem_path and valid_pixels and outside_dem:
        share = outside_dem / valid_pixels
        if share > 0.01:
            warnings.append(
                f"{share:.0%} of the orthophoto lies outside the DEM and was projected at "
                f"an assumed height of {float(spec.background_elevation or 0.0):.1f} m. "
                "Positions there are only as good as that assumption: extend the DEM "
                "to cover the whole photo, or clip the photo to the DEM."
            )
    elif not spec.dem_path:
        warnings.append(
            f"No DEM: the whole orthophoto was projected at an assumed height of "
            f"{float(spec.background_elevation or 0.0):.1f} m, so relief displaces it. "
            "Use a DEM for anything but flat ground."
        )
    if valid_fraction < 0.05:
        warnings.append(
            f"Only {valid_fraction:.1%} of the output grid received image data. "
            "Check the exterior orientation and the output projection."
        )

    if direct:
        try:
            from . import raster

            raster.build_overviews(spec.output_path)
        except Exception:
            warnings.append("Output written, but overview pyramid could not be built")
    else:
        label = export.FORMATS[out_format]["label"]
        if progress:
            progress(1.0, f"Writing {label}")
        try:
            warnings.extend(export.convert(
                working_path, spec.output_path, out_format, out_compress, spec.jpeg_quality,
                progress=(lambda message: progress(1.0, message)) if progress else None,
            ))
        except BaseException:
            Path(spec.output_path).unlink(missing_ok=True)
            raise
        finally:
            Path(working_path).unlink(missing_ok=True)

    if depth_path:
        Path(depth_path).unlink(missing_ok=True)
    if hidden_sink is not None:
        hidden_sink.close()
        if not hidden_count:
            Path(hidden_path).unlink(missing_ok=True)
    return OrthoResult(
        output_path=spec.output_path,
        width=width,
        height=height,
        bounds=bounds,
        pixel_size=(spec.pixel_size_x, spec.pixel_size_y),
        band_count=len(band_list),
        tiles_written=written,
        valid_fraction=valid_fraction,
        elapsed_seconds=time.time() - started,
        warnings=warnings,
    )
