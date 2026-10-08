"""Everything that happens to an elevation model after extraction.

Stereo matching hands back a surface, not a product. It has holes over water,
spikes on roofs, noise in shadow, and it covers one stereo pair rather than the
block. The work between that and something a client can use is a real stage of
the job, and it belongs here rather than in a round trip through a GIS package.

What lives here:

    merge           several per-pair DEMs into one surface
    fill_voids      interpolate the holes, but only where data is nearby
    smooth          median or gaussian, with the holes respected
    hillshade       the fastest way to see whether a DEM is any good
    contours        the classic deliverable, as GeoJSON
    surface_to_terrain   a surface model filtered down to bare earth
    volume_between  cut and fill between two surfaces
    profile         elevation along a line

Every function reads and writes ordinary GeoTIFF and leaves the input alone.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

__all__ = [
    "TerrainError",
    "merge",
    "fill_voids",
    "smooth",
    "hillshade",
    "contours",
    "surface_to_terrain",
    "volume_between",
    "profile",
    "statistics",
]

NODATA = -9999.0


class TerrainError(RuntimeError):
    pass


# -- shared helpers --------------------------------------------------------


def _require_metres(crs, what: str) -> None:
    """Refuse a DEM whose cells are not metres.

    Slope, shading, filtering and volume all turn cell size into distance. A
    DEM in latitude/longitude has cells in degrees, whose ground size depends
    on latitude; guessing a conversion would give a plausible-looking number
    that is wrong, so the operator is asked to reproject instead.
    """
    if crs is None:
        return
    if crs.is_geographic:
        raise TerrainError(
            f"{what} needs a DEM in a projected coordinate system in metres, and this "
            "one is in degrees of latitude and longitude. Reproject it to your Lo zone "
            "or UTM zone first."
        )
    try:
        factor = crs.linear_units_factor[1]
    except Exception:  # noqa: BLE001
        factor = 1.0
    if abs(factor - 1.0) > 1e-9:
        raise TerrainError(
            f"{what} needs a DEM in metres, and this one is in "
            f"{crs.linear_units or 'other units'}. Reproject it to a metric system first."
        )


def _bounds_and_cell(dataset, crs) -> tuple[tuple, float]:
    """A dataset's bounds and cell size, expressed in ``crs``."""
    from rasterio.warp import transform_bounds

    if dataset.crs is None or crs is None or dataset.crs == crs:
        return tuple(dataset.bounds), abs(dataset.transform.a)
    bounds = transform_bounds(dataset.crs, crs, *dataset.bounds, densify_pts=21)
    # The cell size in the target system: its bounds' span over the cell count.
    cell = min((bounds[2] - bounds[0]) / dataset.width, (bounds[3] - bounds[1]) / dataset.height)
    return tuple(bounds), abs(cell)


def _read(path: str) -> tuple[np.ndarray, object, object, float]:
    """Read an elevation raster as float with voids as NaN."""
    import rasterio

    with rasterio.open(path) as dataset:
        band = dataset.read(1).astype(np.float64)
        nodata = dataset.nodata
        transform = dataset.transform
        crs = dataset.crs

    if nodata is not None:
        band[band == nodata] = np.nan
    # A DEM full of -9999 that never declared it is common enough to be worth
    # catching; so is one where the void value is a very large negative.
    band[band < -1e5] = np.nan
    band[band > 1e6] = np.nan

    return band, transform, crs, (nodata if nodata is not None else NODATA)


def _write(
    path: str,
    data: np.ndarray,
    transform,
    crs,
    nodata: float = NODATA,
    dtype: str = "float32",
) -> str:
    import rasterio

    out = np.where(np.isfinite(data), data, nodata).astype(dtype)
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        path, "w", driver="GTiff",
        width=out.shape[1], height=out.shape[0], count=1,
        dtype=dtype, crs=crs, transform=transform, nodata=nodata,
        tiled=True, blockxsize=256, blockysize=256,
        compress="DEFLATE", predictor=3 if dtype.startswith("float") else 2,
        BIGTIFF="IF_SAFER",
    ) as sink:
        sink.write(out, 1)

    try:
        from . import raster

        raster.build_overviews(path)
    except Exception:
        pass

    return str(path)


def statistics(path: str) -> dict:
    """Summarise an elevation raster, including how much of it is actually there."""
    band, transform, crs, _ = _read(path)
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        raise TerrainError(f"{Path(path).name} contains no valid elevations")

    return {
        "path": str(Path(path).resolve()),
        "width": int(band.shape[1]),
        "height": int(band.shape[0]),
        "cellSize": float(abs(transform.a)),
        "crs": crs.to_string() if crs else None,
        "coverage": float(np.isfinite(band).mean()),
        "voidCells": int((~np.isfinite(band)).sum()),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "median": float(np.median(finite)),
    }


# -- merge -----------------------------------------------------------------


def merge(
    paths: Sequence[str],
    output_path: str,
    method: str = "feather",
    cell_size: Optional[float] = None,
    progress: Optional[Callable[[float, str], None]] = None,
    align: bool = False,
    trim_cells: int = 0,
) -> dict:
    """Join several elevation models into one surface.

    ``feather`` weights each input by how far inside its own valid area a cell
    sits, so the join happens where both surfaces are well supported rather
    than at an arbitrary rectangle edge. ``median`` is the robust choice when
    the inputs disagree -- three overlapping stereo DEMs will each have their
    own blunders, and the median simply ignores them. ``first`` is a straight
    painter's order for when one input is known to be better.
    """
    import rasterio
    from rasterio.warp import Resampling, reproject
    from rasterio.transform import Affine
    from scipy.ndimage import binary_erosion, distance_transform_edt

    if len(paths) < 2:
        raise TerrainError("Merging needs at least two elevation models")

    with rasterio.open(paths[0]) as first:
        crs = first.crs
    if crs is None:
        raise TerrainError("The first elevation model has no coordinate system")

    # Every input's footprint and cell size in the first input's system: the
    # inputs may be in different projections, and mixing their raw numbers
    # would build the merged grid in no system at all.
    described = []
    for path in paths:
        with rasterio.open(path) as dataset:
            bounds, cell = _bounds_and_cell(dataset, crs)
            described.append({"path": path, "bounds": bounds, "res": cell, "crs": dataset.crs})

    resolution = float(cell_size or min(d["res"] for d in described))
    west = min(d["bounds"][0] for d in described)
    south = min(d["bounds"][1] for d in described)
    east = max(d["bounds"][2] for d in described)
    north = max(d["bounds"][3] for d in described)

    width = max(1, int(math.ceil((east - west) / resolution)))
    height = max(1, int(math.ceil((north - south) / resolution)))
    if width * height > 400_000_000:
        raise TerrainError(
            f"The merged grid would be {width} x {height} cells. "
            "Coarsen the cell size."
        )

    transform = Affine(resolution, 0.0, west, 0.0, -resolution, north)

    accumulator = np.zeros((height, width), dtype=np.float64)
    weights = np.zeros((height, width), dtype=np.float64)
    stack: list[np.ndarray] = []

    for index, entry in enumerate(described):
        if progress:
            progress(0.1 + 0.6 * index / len(described),
                     f"Resampling {Path(entry['path']).name}")

        with rasterio.open(entry["path"]) as source:
            destination = np.full((height, width), np.nan, dtype=np.float32)
            reproject(
                source=rasterio.band(source, 1),
                destination=destination,
                src_transform=source.transform, src_crs=source.crs,
                src_nodata=source.nodata,
                dst_transform=transform, dst_crs=crs, dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
        destination[destination < -1e5] = np.nan
        if trim_cells > 0:
            # The outer rim of a matched surface is its least reliable part:
            # half a correlation window of guesswork against the void.
            inner = binary_erosion(np.isfinite(destination), iterations=trim_cells)
            destination[~inner] = np.nan
        stack.append(destination)

    offsets = _vertical_offsets(stack) if align else [0.0] * len(stack)
    for layer, offset in zip(stack, offsets):
        if offset:
            layer += np.float32(offset)

    if method == "feather" and len(stack) >= 3:
        # Where three or more surfaces agree, a blunder in one is outvoted
        # before blending rather than smeared into the result.
        cube = np.stack(stack)
        count = np.isfinite(cube).sum(axis=0)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            centre = np.nanmedian(cube, axis=0)
        outlier = (count >= 3) & (np.abs(cube - centre) > OUTLIER_M)
        for layer, reject in zip(stack, outlier):
            layer[reject] = np.nan
        del cube

    for layer in stack if method != "median" else []:
        valid = np.isfinite(layer)
        if not valid.any():
            continue

        if method == "first":
            take = valid & (weights == 0)
            accumulator[take] = layer[take]
            weights[take] = 1.0
        else:
            # Depth inside the valid region, so the seam lands where both
            # surfaces are well supported.
            weight = distance_transform_edt(valid)
            weight = np.where(valid, np.maximum(weight, 1e-6), 0.0)
            accumulator += np.where(valid, layer * weight, 0.0)
            weights += weight

    if method == "median":
        if progress:
            progress(0.85, "Taking the median")
        cube = np.stack(stack)
        with np.errstate(invalid="ignore"):
            merged = np.nanmedian(cube, axis=0)
        merged[np.all(~np.isfinite(cube), axis=0)] = np.nan
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            merged = np.where(weights > 0, accumulator / np.maximum(weights, 1e-12), np.nan)

    if progress:
        progress(0.95, "Writing")

    _write(output_path, merged, transform, crs)
    result = statistics(output_path)
    result["inputs"] = [str(p) for p in paths]
    result["method"] = method
    if align:
        result["verticalOffsets"] = [round(float(o), 3) for o in offsets]
    return result


# Larger than the disagreement of two good surfaces, smaller than a blunder.
OUTLIER_M = 5.0


def _vertical_offsets(layers: Sequence[np.ndarray], min_overlap: int = 500) -> list[float]:
    """Shifts that level overlapping surfaces onto one another.

    Each stereo pair carries its own small height error from the model, and
    joining them as-is leaves a step at every seam. The median difference in
    each overlap is a robust measure of that step; a least-squares solve over
    all overlaps spreads the correction so the shifts sum to zero, which
    removes the steps without moving the surface as a whole.
    """
    n = len(layers)
    rows, targets = [], []
    for i in range(n):
        for j in range(i + 1, n):
            both = np.isfinite(layers[i]) & np.isfinite(layers[j])
            if both.sum() < min_overlap:
                continue
            step = float(np.median(layers[i][both] - layers[j][both]))
            row = np.zeros(n)
            row[i], row[j] = 1.0, -1.0
            rows.append(row)
            targets.append(-step)
    if not rows:
        return [0.0] * n
    rows.append(np.ones(n))
    targets.append(0.0)
    solution, *_ = np.linalg.lstsq(np.array(rows), np.array(targets), rcond=None)
    return [float(v) for v in solution]


# -- repair ----------------------------------------------------------------


def fill_voids(
    path: str,
    output_path: str,
    max_radius_cells: int = 40,
    method: str = "linear",
    progress: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """Interpolate across holes, but only those close enough to real data.

    A hole in the middle of a dam is not missing data to be invented -- it is
    water, which stereo cannot see and should not guess at. Only voids within
    ``max_radius_cells`` of measured ground are filled; anything larger stays a
    void and says so.
    """
    from scipy.interpolate import griddata
    from scipy.ndimage import distance_transform_edt

    band, transform, crs, _ = _read(path)
    holes = ~np.isfinite(band)

    if not holes.any():
        _write(output_path, band, transform, crs)
        result = statistics(output_path)
        result["filledCells"] = 0
        result["note"] = "There were no voids to fill."
        return result

    known = np.isfinite(band)
    if known.sum() < 4:
        raise TerrainError("Too little valid data to interpolate from")

    if progress:
        progress(0.2, "Measuring void sizes")

    distance = distance_transform_edt(holes)
    fillable = holes & (distance <= max_radius_cells)

    if not fillable.any():
        _write(output_path, band, transform, crs)
        result = statistics(output_path)
        result["filledCells"] = 0
        result["note"] = (
            f"Every void is further than {max_radius_cells} cells from measured "
            "ground, so none were filled."
        )
        return result

    if progress:
        progress(0.45, f"Interpolating {int(fillable.sum()):,} cells")

    known_rows, known_cols = np.nonzero(known)
    target_rows, target_cols = np.nonzero(fillable)

    scipy_method = {"linear": "linear", "cubic": "cubic", "nearest": "nearest"}.get(
        method, "linear"
    )
    filled = griddata(
        np.column_stack([known_rows, known_cols]),
        band[known],
        np.column_stack([target_rows, target_cols]),
        method=scipy_method,
    )

    missing = ~np.isfinite(filled)
    if missing.any():
        # Outside the convex hull the chosen method returns nothing.
        filled[missing] = griddata(
            np.column_stack([known_rows, known_cols]),
            band[known],
            np.column_stack([target_rows[missing], target_cols[missing]]),
            method="nearest",
        )

    result_band = band.copy()
    result_band[target_rows, target_cols] = filled

    if progress:
        progress(0.9, "Writing")

    _write(output_path, result_band, transform, crs)
    stats = statistics(output_path)
    stats["filledCells"] = int(fillable.sum())
    stats["remainingVoids"] = int((holes & ~fillable).sum())
    stats["note"] = (
        f"Filled {int(fillable.sum()):,} cells. "
        f"{int((holes & ~fillable).sum()):,} left as void -- too far from any "
        "measured ground to interpolate honestly."
    )
    return stats


def smooth(
    path: str,
    output_path: str,
    method: str = "median",
    size: int = 3,
    progress: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """Reduce matching noise without dragging voids into the result.

    ``median`` removes isolated spikes while keeping breaklines sharp, which is
    almost always what an extracted DEM needs. ``gaussian`` is gentler and
    blurs edges; use it on already-clean data.
    """
    from scipy.ndimage import gaussian_filter, median_filter, uniform_filter

    band, transform, crs, _ = _read(path)
    valid = np.isfinite(band)
    if not valid.any():
        raise TerrainError("Nothing to smooth")

    if progress:
        progress(0.3, f"{method} filter, {size} cells")

    size = max(3, int(size) | 1)

    if method == "median":
        # Voids would drag the median down, so they are temporarily replaced
        # with the local mean and masked out again afterwards.
        filled = np.where(valid, band, np.nanmean(band[valid]))
        result = median_filter(filled, size=size, mode="nearest")
    elif method == "gaussian":
        sigma = size / 3.0
        filled = np.where(valid, band, 0.0)
        weight = valid.astype(np.float64)
        # Normalised convolution: blur the data and the mask together, so a
        # cell beside a void is not pulled towards zero.
        blurred = gaussian_filter(filled, sigma=sigma, mode="nearest")
        norm = gaussian_filter(weight, sigma=sigma, mode="nearest")
        with np.errstate(invalid="ignore", divide="ignore"):
            result = np.where(norm > 1e-6, blurred / norm, np.nan)
    else:
        filled = np.where(valid, band, np.nanmean(band[valid]))
        result = uniform_filter(filled, size=size, mode="nearest")

    result = np.where(valid, result, np.nan)

    if progress:
        progress(0.9, "Writing")

    _write(output_path, result, transform, crs)
    stats = statistics(output_path)
    before = band[valid]
    after = result[valid]
    stats["method"] = method
    stats["windowCells"] = size
    stats["meanChange"] = float(np.nanmean(np.abs(after - before)))
    stats["maxChange"] = float(np.nanmax(np.abs(after - before)))
    return stats


# -- shaded relief ---------------------------------------------------------


def hillshade(
    path: str,
    output_path: str,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
    progress: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """Shaded relief, by Horn's method.

    This is the single most useful thing you can do to an extracted DEM,
    because elevation rendered as a colour ramp hides exactly the errors that
    matter. Under a low sun, a matching blunder over water reads instantly as a
    crater and a roofline reads as a roofline.
    """
    band, transform, crs, _ = _read(path)
    _require_metres(crs, "Shaded relief")
    valid = np.isfinite(band)
    if not valid.any():
        raise TerrainError("Nothing to shade")

    if progress:
        progress(0.3, "Computing slope and aspect")

    cell = float(abs(transform.a))
    filled = np.where(valid, band, np.nanmean(band[valid])) * z_factor

    # Horn's (1981) gradient, as GDAL and ArcGIS compute it: a 3 x 3 kernel
    # weighting the four direct neighbours twice the diagonals, which is far
    # less noisy than a two-point difference.
    #   a b c
    #   d e f      dz/dx = ((c + 2f + i) - (a + 2d + g)) / 8 cell
    #   g h i      dz/dy = ((g + 2h + i) - (a + 2b + c)) / 8 cell   (y down)
    # Rows increase SOUTHWARD, so the northward gradient is the negative of
    # the row gradient -- getting that sign wrong mirrors the illumination,
    # which is the classic hillshade bug.
    p = np.pad(filled, 1, mode="edge")
    a, b, c = p[:-2, :-2], p[:-2, 1:-1], p[:-2, 2:]
    d, f = p[1:-1, :-2], p[1:-1, 2:]
    g, h, i = p[2:, :-2], p[2:, 1:-1], p[2:, 2:]
    dz_east = ((c + 2 * f + i) - (a + 2 * d + g)) / (8.0 * cell)
    dz_north = -((g + 2 * h + i) - (a + 2 * b + c)) / (8.0 * cell)

    # Illumination by dot product rather than by slope and aspect angles. The
    # surface normal is (-dz/dx, -dz/dy, 1); the light vector points towards
    # the sun, with azimuth measured clockwise from north.
    normal_length = np.sqrt(dz_east**2 + dz_north**2 + 1.0)
    azimuth_rad = math.radians(azimuth)
    altitude_rad = math.radians(altitude)

    light = (
        math.sin(azimuth_rad) * math.cos(altitude_rad),
        math.cos(azimuth_rad) * math.cos(altitude_rad),
        math.sin(altitude_rad),
    )

    shaded = (
        (-dz_east * light[0]) + (-dz_north * light[1]) + light[2]
    ) / normal_length

    shaded = np.clip(shaded, 0.0, 1.0) * 255.0
    shaded = np.where(valid, shaded, 0).astype(np.uint8)

    if progress:
        progress(0.85, "Writing")

    import rasterio

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path, "w", driver="GTiff",
        width=shaded.shape[1], height=shaded.shape[0], count=1,
        dtype="uint8", crs=crs, transform=transform, nodata=0,
        tiled=True, blockxsize=256, blockysize=256, compress="DEFLATE",
    ) as sink:
        sink.write(shaded, 1)

    try:
        from . import raster

        raster.build_overviews(output_path)
    except Exception:
        pass

    return {
        "outputPath": str(output_path),
        "azimuth": azimuth,
        "altitude": altitude,
        "zFactor": z_factor,
        "width": int(shaded.shape[1]),
        "height": int(shaded.shape[0]),
    }


# -- contours --------------------------------------------------------------


def contours(
    path: str,
    output_path: str,
    interval: float = 5.0,
    base: float = 0.0,
    index_every: int = 5,
    simplify_px: float = 0.7,
    smooth_cells: int = 2,
    progress: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """Trace contour lines and write them as GeoJSON.

    Every ``index_every``-th contour is tagged as an index contour, which is
    the convention that makes a contour sheet readable -- heavier line, labels
    on those only.

    The surface is lightly smoothed before tracing. Contours drawn straight off
    a matched DEM are unusably crenellated; a small blur costs nothing in
    accuracy at normal intervals and makes the result draftable.
    """
    import cv2
    from scipy.ndimage import gaussian_filter

    band, transform, crs, _ = _read(path)
    valid = np.isfinite(band)
    finite = band[valid]
    if finite.size == 0:
        raise TerrainError("Nothing to contour")

    low, high = float(finite.min()), float(finite.max())
    if interval <= 0:
        raise TerrainError("The contour interval must be greater than zero")

    levels = np.arange(
        math.floor((low - base) / interval) * interval + base,
        high + interval,
        interval,
    )
    levels = [float(v) for v in levels if low < v < high]

    if not levels:
        raise TerrainError(
            f"No contours fall inside the elevation range "
            f"{low:.1f} to {high:.1f} m at a {interval} m interval."
        )
    if len(levels) > 2000:
        raise TerrainError(
            f"That interval would produce {len(levels):,} contours over a "
            f"{high - low:.0f} m range. Use a coarser interval."
        )

    surface = np.where(valid, band, np.nan)
    if smooth_cells > 0:
        filled = np.where(valid, band, np.nanmean(finite))
        weight = valid.astype(np.float64)
        blurred = gaussian_filter(filled, sigma=smooth_cells, mode="nearest")
        norm = gaussian_filter(weight, sigma=smooth_cells, mode="nearest")
        with np.errstate(invalid="ignore", divide="ignore"):
            surface = np.where(norm > 1e-6, blurred / norm, np.nan)
        surface = np.where(valid, surface, np.nan)

    features = []
    total_vertices = 0

    for index, level in enumerate(levels):
        if progress and index % 5 == 0:
            progress(0.1 + 0.8 * index / len(levels), f"Tracing {level:g} m")

        mask = np.zeros(surface.shape, dtype=np.uint8)
        mask[np.isfinite(surface) & (surface >= level)] = 255

        found, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for polyline in found:
            if len(polyline) < 8:
                continue
            if simplify_px > 0:
                polyline = cv2.approxPolyDP(polyline, simplify_px, True)
            if len(polyline) < 4:
                continue

            points = polyline.reshape(-1, 2).astype(float)
            xs, ys = transform * (points[:, 0] + 0.5, points[:, 1] + 0.5)
            coordinates = [[float(x), float(y)] for x, y in zip(xs, ys)]
            coordinates.append(coordinates[0])
            total_vertices += len(coordinates)

            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coordinates},
                "properties": {
                    "elevation": round(level, 4),
                    "index": (round((level - base) / interval) % index_every) == 0,
                },
            })

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        "type": "FeatureCollection",
        "projectCrs": crs.to_string() if crs else None,
        "features": features,
    }), encoding="utf-8")

    if progress:
        progress(1.0, f"{len(features):,} contours")

    return {
        "outputPath": str(target),
        "interval": interval,
        "base": base,
        "levels": len(levels),
        "lineCount": len(features),
        "vertexCount": total_vertices,
        "elevationRange": [low, high],
        "indexEvery": index_every,
    }


# -- surface to bare earth -------------------------------------------------


def surface_to_terrain(
    path: str,
    output_path: str,
    max_window_cells: int = 33,
    slope_tolerance: float = 0.4,
    initial_tolerance: float = 0.5,
    max_tolerance: float = 8.0,
    progress: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """Filter a surface model down to bare earth.

    A progressive morphological filter: open the surface with a window that
    grows each pass, and treat any cell standing more than a tolerance above
    the opened surface as something sitting on the ground rather than the
    ground itself. The tolerance grows with the window, because a wide window
    legitimately cuts across real terrain relief and a fixed threshold would
    start shaving hilltops.

    Stereo gives you a digital *surface* model -- the tops of trees and roofs.
    Most clients want the ground under them.
    """
    from scipy.ndimage import grey_opening
    from scipy.interpolate import griddata

    band, transform, crs, _ = _read(path)
    _require_metres(crs, "Bare-earth filtering")
    valid = np.isfinite(band)
    if not valid.any():
        raise TerrainError("Nothing to filter")

    cell = float(abs(transform.a))
    working = np.where(valid, band, np.nanmax(band[valid]))
    ground = valid.copy()

    window = 3
    previous_window = 1
    passes = 0

    while window <= max_window_cells:
        passes += 1
        if progress:
            progress(0.1 + 0.65 * window / max_window_cells,
                     f"Opening at {window} cells ({window * cell:.1f} m)")

        opened = grey_opening(working, size=(window, window), mode="nearest")

        # Tolerance grows with the window, capped so a genuine cliff is not
        # mistaken for a building on a large pass.
        tolerance = min(
            initial_tolerance + slope_tolerance * (window - previous_window) * cell,
            max_tolerance,
        )

        ground &= (working - opened) <= tolerance
        working = opened

        previous_window = window
        window = window * 2 + 1

    ground &= valid

    kept = float(ground.sum()) / max(float(valid.sum()), 1.0)
    if ground.sum() < 16:
        raise TerrainError(
            "The filter removed almost everything. The tolerances are probably "
            "too tight for this terrain -- raise the initial tolerance."
        )

    if progress:
        progress(0.8, "Interpolating the ground surface")

    rows, cols = np.nonzero(ground)
    target_rows, target_cols = np.nonzero(valid & ~ground)

    terrain = band.copy()
    if target_rows.size:
        interpolated = griddata(
            np.column_stack([rows, cols]), band[ground],
            np.column_stack([target_rows, target_cols]), method="linear",
        )
        missing = ~np.isfinite(interpolated)
        if missing.any():
            interpolated[missing] = griddata(
                np.column_stack([rows, cols]), band[ground],
                np.column_stack([target_rows[missing], target_cols[missing]]),
                method="nearest",
            )
        terrain[target_rows, target_cols] = interpolated

    terrain = np.where(valid, terrain, np.nan)

    if progress:
        progress(0.94, "Writing")

    _write(output_path, terrain, transform, crs)
    stats = statistics(output_path)

    difference = band[valid] - terrain[valid]
    stats.update({
        "groundFraction": kept,
        "removedCells": int((valid & ~ground).sum()),
        "passes": passes,
        "meanRemovedHeight": float(np.nanmean(difference[difference > 0]))
        if np.any(difference > 0) else 0.0,
        "maxRemovedHeight": float(np.nanmax(difference)) if difference.size else 0.0,
        "note": (
            f"{kept:.0%} of cells were classified as ground. "
            "Check the result against a hillshade before delivering it -- this "
            "filter cannot tell a flat roof from a terrace."
        ),
    })
    return stats


# -- volume ----------------------------------------------------------------


def volume_between(
    surface_path: str,
    base_path: str,
    bounds: Optional[Sequence[float]] = None,
    progress: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """Cut and fill between two surfaces.

    The everyday quantity survey: how much material sits above a design
    surface, and how much is missing beneath it. Both grids are brought onto
    the finer of the two so no volume is lost to resampling.
    """
    import rasterio
    from rasterio.warp import Resampling, reproject
    from rasterio.transform import Affine

    with rasterio.open(surface_path) as a, rasterio.open(base_path) as b:
        if a.crs is None or b.crs is None:
            raise TerrainError("Both surfaces need a coordinate system")
        _require_metres(a.crs, "Volume")

        # Overlap and cell size in the surface's system; the base may be in
        # another and is resampled into it below.
        a_bounds, a_cell = tuple(a.bounds), abs(a.transform.a)
        b_bounds, b_cell = _bounds_and_cell(b, a.crs)
        resolution = min(a_cell, b_cell)
        west = max(a_bounds[0], b_bounds[0])
        south = max(a_bounds[1], b_bounds[1])
        east = min(a_bounds[2], b_bounds[2])
        north = min(a_bounds[3], b_bounds[3])

        if bounds:
            west, south = max(west, bounds[0]), max(south, bounds[1])
            east, north = min(east, bounds[2]), min(north, bounds[3])

        if east <= west or north <= south:
            raise TerrainError("The two surfaces do not overlap")

        width = max(1, int((east - west) / resolution))
        height = max(1, int((north - south) / resolution))
        transform = Affine(resolution, 0, west, 0, -resolution, north)

        if progress:
            progress(0.3, "Bringing both surfaces onto a common grid")

        grids = []
        for dataset in (a, b):
            destination = np.full((height, width), np.nan, dtype=np.float64)
            reproject(
                source=rasterio.band(dataset, 1), destination=destination,
                src_transform=dataset.transform, src_crs=dataset.crs,
                src_nodata=dataset.nodata,
                dst_transform=transform, dst_crs=a.crs, dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
            destination[destination < -1e5] = np.nan
            grids.append(destination)

    if progress:
        progress(0.75, "Differencing")

    difference = grids[0] - grids[1]
    comparable = np.isfinite(difference)
    if not comparable.any():
        raise TerrainError("The surfaces overlap but share no valid cells")

    area = resolution * resolution
    cut = difference[comparable & (difference > 0)]
    fill = difference[comparable & (difference < 0)]

    return {
        "cellSize": resolution,
        "cellArea": area,
        "comparedCells": int(comparable.sum()),
        "comparedArea": float(comparable.sum() * area),
        "cutVolume": float(cut.sum() * area) if cut.size else 0.0,
        "fillVolume": float(-fill.sum() * area) if fill.size else 0.0,
        "netVolume": float(np.nansum(difference[comparable]) * area),
        "meanDifference": float(np.nanmean(difference[comparable])),
        "maxAbove": float(np.nanmax(difference[comparable])),
        "maxBelow": float(np.nanmin(difference[comparable])),
        "bounds": [west, south, east, north],
    }


# -- profile ---------------------------------------------------------------


def profile(
    path: str,
    line: Sequence[Sequence[float]],
    samples: int = 400,
) -> dict:
    """Sample elevation along a polyline.

    Returns distance along the line against height, which is the section a
    client asks for whenever they want to know what the ground does between
    two points.
    """
    from . import raster

    if len(line) < 2:
        raise TerrainError("A profile needs at least two points")

    import rasterio

    with rasterio.open(path) as dataset:
        _require_metres(dataset.crs, "A profile")

    vertices = np.asarray(line, dtype=float)
    segments = np.diff(vertices, axis=0)
    lengths = np.hypot(segments[:, 0], segments[:, 1])
    total = float(lengths.sum())
    if total <= 0:
        raise TerrainError("The profile line has zero length")

    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    distances = np.linspace(0.0, total, max(8, int(samples)))

    xs = np.interp(distances, cumulative, vertices[:, 0])
    ys = np.interp(distances, cumulative, vertices[:, 1])
    zs = raster.sample_elevation(path, xs, ys)

    valid = np.isfinite(zs)
    if not valid.any():
        raise TerrainError("The profile line falls entirely outside the elevation model")

    gradients = np.full(zs.shape, np.nan)
    step = total / (len(distances) - 1)
    gradients[1:-1] = (zs[2:] - zs[:-2]) / (2 * step)

    return {
        "length": total,
        "sampleCount": int(len(distances)),
        "coverage": float(valid.mean()),
        "minElevation": float(np.nanmin(zs)),
        "maxElevation": float(np.nanmax(zs)),
        "relief": float(np.nanmax(zs) - np.nanmin(zs)),
        "maxGradient": float(np.nanmax(np.abs(gradients))) if np.isfinite(gradients).any() else 0.0,
        "samples": [
            {
                "distance": float(d),
                "x": float(x), "y": float(y),
                "elevation": None if not math.isfinite(z) else float(z),
            }
            for d, x, y, z in zip(distances, xs, ys, zs)
        ],
    }
