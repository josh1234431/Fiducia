"""LiDAR point clouds to elevation rasters.

Named for the pulsed light that produces the data. Ingests LAS/LAZ, filters by
ASPRS classification and return number, and rasterises to a DTM or DSM.

The interpolation chain matches what the ArcGIS "LAS Dataset to Raster" tool
does -- binning with a cell assignment rule, then void filling -- because that
is the result people are checking their work against. What is different is that the filtering is explicit and visible: you
can see how many points survived each stage, which is the difference between
"my DTM has holes" and "class 2 only kept 4% of my returns".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import math

import numpy as np

__all__ = ["LidarSummary", "RasterizeOptions", "inspect_cloud", "rasterize", "ASPRS_CLASSES"]

# The ASPRS standard classification codes that actually turn up in practice.
ASPRS_CLASSES = {
    0: "Never classified",
    1: "Unassigned",
    2: "Ground",
    3: "Low vegetation",
    4: "Medium vegetation",
    5: "High vegetation",
    6: "Building",
    7: "Low point (noise)",
    8: "Model key point",
    9: "Water",
    10: "Rail",
    11: "Road surface",
    12: "Overlap",
    13: "Wire guard",
    14: "Wire conductor",
    15: "Transmission tower",
    17: "Bridge deck",
    18: "High noise",
}

CELL_ASSIGNMENT = ("idw", "minimum", "maximum", "mean", "nearest")
VOID_FILL = ("natural_neighbor", "linear", "nearest", "none")


@dataclass
class LidarSummary:
    path: str
    point_count: int
    bounds: tuple
    crs: Optional[str]
    version: str
    point_format: int
    class_histogram: dict
    return_histogram: dict
    elevation_range: tuple
    average_spacing: float

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "pointCount": self.point_count,
            "bounds": list(self.bounds),
            "crs": self.crs,
            "version": self.version,
            "pointFormat": self.point_format,
            "classHistogram": {
                str(k): {"count": v, "label": ASPRS_CLASSES.get(k, f"Class {k}")}
                for k, v in self.class_histogram.items()
            },
            "returnHistogram": {str(k): v for k, v in self.return_histogram.items()},
            "elevationRange": list(self.elevation_range),
            "averageSpacing": self.average_spacing,
        }


@dataclass
class RasterizeOptions:
    output_path: str = ""
    cell_size: float = 1.0
    classes: Sequence[int] = field(default_factory=lambda: [2])
    returns: str = "last"                  # all | first | last | single
    cell_assignment: str = "idw"
    void_fill: str = "natural_neighbor"
    crs: Optional[str] = None
    nodata: float = -9999.0
    max_void_radius_cells: int = 24
    idw_power: float = 2.0


def inspect_cloud(path: str, sample_limit: int = 4_000_000) -> LidarSummary:
    """Read headers and a decimated sample to summarise a cloud.

    Deliberately avoids loading the whole file. A metropolitan LiDAR tile runs
    to tens of millions of points, and the operator only needs to know what is
    in it before deciding how to filter.
    """
    import laspy

    with laspy.open(path) as reader:
        header = reader.header
        count = int(header.point_count)
        step = max(1, count // sample_limit)

        points = reader.read()
        classification = np.asarray(points.classification)
        returns = np.asarray(points.return_number)
        z = np.asarray(points.z)

        if step > 1:
            classification = classification[::step]
            returns = returns[::step]
            z_sample = z[::step]
        else:
            z_sample = z

        class_values, class_counts = np.unique(classification, return_counts=True)
        return_values, return_counts = np.unique(returns, return_counts=True)

        bounds = (
            float(header.x_min), float(header.y_min),
            float(header.x_max), float(header.y_max),
        )
        area = max((bounds[2] - bounds[0]) * (bounds[3] - bounds[1]), 1e-9)
        spacing = float(np.sqrt(area / max(count, 1)))

        crs = None
        try:
            parsed = header.parse_crs()
            if parsed is not None:
                crs = parsed.to_string()
        except Exception:
            pass

        return LidarSummary(
            path=str(Path(path).resolve()),
            point_count=count,
            bounds=bounds,
            crs=crs,
            version=f"{header.version.major}.{header.version.minor}",
            point_format=int(header.point_format.id),
            class_histogram={
                int(v): int(c * step) for v, c in zip(class_values, class_counts)
            },
            return_histogram={
                int(v): int(c * step) for v, c in zip(return_values, return_counts)
            },
            elevation_range=(float(np.min(z_sample)), float(np.max(z_sample))),
            average_spacing=spacing,
        )


def _filter_points(points, options: RasterizeOptions):
    classification = np.asarray(points.classification)
    keep = np.ones(classification.shape, dtype=bool)

    if options.classes:
        keep &= np.isin(classification, np.asarray(list(options.classes), dtype=classification.dtype))

    return_number = np.asarray(points.return_number)
    number_of_returns = np.asarray(points.number_of_returns)

    if options.returns == "first":
        keep &= return_number == 1
    elif options.returns == "last":
        keep &= return_number == number_of_returns
    elif options.returns == "single":
        keep &= number_of_returns == 1

    return keep


def _bin_points(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    bounds: tuple,
    cell: float,
    assignment: str,
    idw_power: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin points into cells. Returns (grid, filled_mask)."""
    west, south, east, north = bounds
    width = max(1, int(np.ceil((east - west) / cell)))
    height = max(1, int(np.ceil((north - south) / cell)))

    cols = np.clip(((xs - west) / cell).astype(np.int64), 0, width - 1)
    rows = np.clip(((north - ys) / cell).astype(np.int64), 0, height - 1)
    flat = rows * width + cols
    size = width * height

    grid = np.full(size, np.nan, dtype=np.float64)

    if assignment == "minimum":
        # np.minimum.at is the correct scatter-reduce here; a plain fancy-index
        # assignment would keep an arbitrary point per cell, not the lowest.
        accumulator = np.full(size, np.inf)
        np.minimum.at(accumulator, flat, zs)
        grid = np.where(np.isfinite(accumulator), accumulator, np.nan)

    elif assignment == "maximum":
        accumulator = np.full(size, -np.inf)
        np.maximum.at(accumulator, flat, zs)
        grid = np.where(np.isfinite(accumulator), accumulator, np.nan)

    elif assignment == "mean":
        totals = np.zeros(size)
        counts = np.zeros(size)
        np.add.at(totals, flat, zs)
        np.add.at(counts, flat, 1.0)
        grid = np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)

    elif assignment == "nearest":
        # The height of the one point closest to each cell's centre: sort by
        # cell, then by distance, and keep the first of each cell.
        centre_x = west + (cols + 0.5) * cell
        centre_y = north - (rows + 0.5) * cell
        distance = np.hypot(xs - centre_x, ys - centre_y)
        order = np.lexsort((distance, flat))
        first = np.ones(order.size, dtype=bool)
        first[1:] = flat[order][1:] != flat[order][:-1]
        chosen = order[first]
        grid[flat[chosen]] = zs[chosen]

    else:  # idw -- weight by distance from the cell centre
        centre_x = west + (cols + 0.5) * cell
        centre_y = north - (rows + 0.5) * cell
        distance = np.hypot(xs - centre_x, ys - centre_y)
        weights = 1.0 / np.power(np.maximum(distance, cell * 0.01), idw_power)

        weighted = np.zeros(size)
        weight_sum = np.zeros(size)
        np.add.at(weighted, flat, zs * weights)
        np.add.at(weight_sum, flat, weights)
        grid = np.where(weight_sum > 0, weighted / np.maximum(weight_sum, 1e-12), np.nan)

    grid = grid.reshape(height, width)
    return grid, np.isfinite(grid)


def _fill_voids(grid: np.ndarray, method: str, max_radius: int) -> np.ndarray:
    """Interpolate across cells that received no points."""
    if method == "none":
        return grid

    holes = ~np.isfinite(grid)
    if not holes.any() or np.isfinite(grid).sum() < 4:
        return grid

    from scipy.interpolate import griddata
    from scipy.ndimage import distance_transform_edt

    # Only fill voids within reach of real data; a hole in the middle of a
    # lake should stay a hole rather than being invented.
    distance = distance_transform_edt(holes)
    fillable = holes & (distance <= max_radius)
    if not fillable.any():
        return grid

    known_rows, known_cols = np.nonzero(np.isfinite(grid))
    known_values = grid[known_rows, known_cols]
    target_rows, target_cols = np.nonzero(fillable)

    scipy_method = {
        "natural_neighbor": "cubic",   # closest available analogue
        "linear": "linear",
        "nearest": "nearest",
    }.get(method, "linear")

    filled = griddata(
        np.column_stack([known_rows, known_cols]),
        known_values,
        np.column_stack([target_rows, target_cols]),
        method=scipy_method,
    )

    # Cubic and linear both return NaN outside the convex hull; mop those up.
    missing = ~np.isfinite(filled)
    if missing.any():
        fallback = griddata(
            np.column_stack([known_rows, known_cols]),
            known_values,
            np.column_stack([target_rows[missing], target_cols[missing]]),
            method="nearest",
        )
        filled[missing] = fallback

    result = grid.copy()
    result[target_rows, target_cols] = filled
    return result


def rasterize(
    path: str,
    options: RasterizeOptions,
    progress: Optional[Callable[[float, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> dict:
    """Filter a point cloud and write an elevation GeoTIFF."""
    import laspy
    import rasterio
    from rasterio.transform import Affine

    from .geodesy import file_crs

    if progress:
        progress(0.05, "Reading point cloud")

    with laspy.open(path) as reader:
        header = reader.header
        points = reader.read()

    if should_cancel and should_cancel():
        raise InterruptedError("Cancelled")

    total = len(points.x)
    if progress:
        progress(0.3, f"Filtering {total:,} points")

    keep = _filter_points(points, options)
    kept = int(keep.sum())
    if kept == 0:
        raise ValueError(
            "No points survived the class and return filter. "
            f"Requested classes {list(options.classes)}; "
            f"file contains {sorted(set(np.asarray(points.classification).tolist()))}."
        )

    xs = np.asarray(points.x)[keep].astype(np.float64)
    ys = np.asarray(points.y)[keep].astype(np.float64)
    zs = np.asarray(points.z)[keep].astype(np.float64)

    # Cell edges on whole multiples of the cell size, so the surface lines up
    # with every other raster made at that size; and the extreme points fall
    # inside the last cell rather than on its far edge.
    cell = options.cell_size
    bounds = (math.floor(xs.min() / cell) * cell, math.floor(ys.min() / cell) * cell,
              (math.floor(xs.max() / cell) + 1) * cell, (math.floor(ys.max() / cell) + 1) * cell)

    if progress:
        progress(0.55, f"Binning {kept:,} points at {options.cell_size} m")

    grid, filled_mask = _bin_points(
        xs, ys, zs, bounds, options.cell_size, options.cell_assignment, options.idw_power
    )

    if should_cancel and should_cancel():
        raise InterruptedError("Cancelled")

    void_count = int((~filled_mask).sum())
    if progress:
        progress(0.75, f"Filling {void_count:,} empty cells")

    grid = _fill_voids(grid, options.void_fill, options.max_void_radius_cells)
    output = np.where(np.isfinite(grid), grid, options.nodata).astype(np.float32)

    crs_name = options.crs
    if not crs_name:
        try:
            parsed = header.parse_crs()
            crs_name = parsed.to_string() if parsed is not None else None
        except Exception:
            crs_name = None
    if not crs_name:
        raise ValueError(
            "The point cloud has no coordinate system and none was supplied. "
            "LAS files frequently omit it -- set it explicitly before rasterising."
        )

    height, width = output.shape
    transform = Affine(options.cell_size, 0.0, bounds[0], 0.0, -options.cell_size, bounds[3])

    Path(options.output_path).parent.mkdir(parents=True, exist_ok=True)
    if progress:
        progress(0.9, "Writing raster")

    with rasterio.open(
        options.output_path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        crs=file_crs(crs_name),
        transform=transform,
        nodata=options.nodata,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="DEFLATE",
        BIGTIFF="IF_SAFER",
    ) as sink:
        sink.write(output, 1)

    try:
        from . import raster

        raster.build_overviews(options.output_path)
    except Exception:
        pass

    if progress:
        progress(1.0, "Complete")

    valid = grid[np.isfinite(grid)]
    return {
        "outputPath": options.output_path,
        "width": width,
        "height": height,
        "cellSize": options.cell_size,
        "bounds": list(bounds),
        "crs": crs_name,
        "pointsTotal": total,
        "pointsUsed": kept,
        "pointsRejected": total - kept,
        "voidCellsFilled": void_count,
        "coverage": float(filled_mask.mean()),
        "elevationRange": [float(valid.min()), float(valid.max())] if valid.size else [0.0, 0.0],
        "classes": list(options.classes),
        "returns": options.returns,
    }


def compare_to_reference(
    derived_path: str,
    reference_path: str,
    sample_points: Optional[np.ndarray] = None,
    max_samples: int = 20000,
) -> dict:
    """RMSE and R-squared of a derived surface against a reference.

    The usual validation against a reference such as SRTM, done in one call
    instead of a round trip through Extract Values to Points, a dBASE export
    and a spreadsheet.
    """
    from . import raster

    derived = raster.open_raster(derived_path)
    reference = raster.open_raster(reference_path)

    # The two surfaces may be in different systems. Everything is done in the
    # derived surface's, and the reference is read where each point really is.
    from .ortho import _converter

    derived_crs = derived.crs.to_wkt() if derived.crs else None
    reference_crs = reference.crs.to_wkt() if reference.crs else None
    to_reference = _converter(derived_crs, reference_crs)
    ref_bounds = reference.bounds
    if to_reference:
        from rasterio.warp import transform_bounds

        ref_bounds = transform_bounds(reference.crs, derived.crs, *reference.bounds, densify_pts=21)
        ref_bounds = type("B", (), dict(zip(("left", "bottom", "right", "top"), ref_bounds)))

    if sample_points is None:
        west = max(derived.bounds.left, ref_bounds.left)
        east = min(derived.bounds.right, ref_bounds.right)
        south = max(derived.bounds.bottom, ref_bounds.bottom)
        north = min(derived.bounds.top, ref_bounds.top)
        if east <= west or north <= south:
            raise ValueError("The two surfaces do not overlap")

        side = int(np.sqrt(max_samples))
        xs, ys = np.meshgrid(
            np.linspace(west, east, side), np.linspace(south, north, side)
        )
        xs, ys = xs.ravel(), ys.ravel()
    else:
        xs, ys = sample_points[:, 0], sample_points[:, 1]

    a = raster.sample_elevation(derived_path, xs, ys)
    if to_reference:
        rx, ry = to_reference(xs, ys)
        b = raster.sample_elevation(reference_path, rx, ry)
    else:
        b = raster.sample_elevation(reference_path, xs, ys)

    if derived.nodata is not None:
        a = np.where(a == derived.nodata, np.nan, a)
    if reference.nodata is not None:
        b = np.where(b == reference.nodata, np.nan, b)

    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        raise ValueError("Not enough overlapping valid cells to compare")

    a, b = a[valid], b[valid]
    difference = a - b

    slope, intercept = np.polyfit(b, a, 1)
    predicted = slope * b + intercept
    ss_residual = float(np.sum((a - predicted) ** 2))
    ss_total = float(np.sum((a - a.mean()) ** 2))
    r_squared = 1.0 - ss_residual / ss_total if ss_total > 0 else 0.0

    median = float(np.median(difference))
    return {
        "sampleCount": int(valid.sum()),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "meanError": float(difference.mean()),
        "stdError": float(difference.std()),
        "maxAbsError": float(np.max(np.abs(difference))),
        # Robust measures, which a few blunders or a patch of trees cannot
        # inflate: the median, the normalised median absolute deviation (an
        # estimate of the standard deviation), and the 95th percentile of the
        # absolute error -- the ASPRS measure for vegetated terrain (VVA).
        # NVA is 1.96 RMSE, the measure for open ground.
        "medianError": median,
        "nmad": float(1.4826 * np.median(np.abs(difference - median))),
        "absError95": float(np.percentile(np.abs(difference), 95)),
        "nva95": float(1.96 * np.sqrt(np.mean(difference**2))),
        "rSquared": float(r_squared),
        "slope": float(slope),
        "intercept": float(intercept),
        "scatter": [
            {"reference": float(bv), "derived": float(av)}
            for av, bv in zip(a[:2000], b[:2000])
        ],
    }
