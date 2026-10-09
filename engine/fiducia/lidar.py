"""LiDAR point clouds to elevation rasters.

Named for the pulsed light that produces the data. Ingests LAS/LAZ, filters by
ASPRS classification and return number, and rasterises to a DTM or DSM. It also
finds the ground in an unclassified cloud, and measures every point's height
above that ground -- a canopy height model over trees, a normalised DSM over a
town.

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

__all__ = [
    "LidarSummary", "RasterizeOptions", "inspect_cloud", "rasterize", "ASPRS_CLASSES",
    "GroundOptions", "classify_ground", "HeightOptions", "height_above_ground",
]

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


def _grid_shape(bounds: tuple, cell: float) -> tuple[int, int]:
    """(rows, columns) of a grid of the given cell size over the bounds."""
    west, south, east, north = bounds
    return (max(1, int(np.ceil((north - south) / cell))),
            max(1, int(np.ceil((east - west) / cell))))


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
    height, width = _grid_shape(bounds, cell)

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


def _resolve_crs(header, supplied: Optional[str]) -> str:
    """The cloud's coordinate system: the one supplied, else the file's own."""
    if supplied:
        return supplied
    try:
        parsed = header.parse_crs()
        if parsed is not None:
            return parsed.to_string()
    except Exception:
        pass
    raise ValueError(
        "The point cloud has no coordinate system and none was supplied. "
        "LAS files frequently omit it -- set it explicitly before rasterising."
    )


def _write_raster(path: str, grid: np.ndarray, bounds: tuple, cell: float,
                  crs_name: str, nodata: float) -> None:
    """A tiled, compressed float GeoTIFF with overviews; NaN becomes nodata."""
    import rasterio
    from rasterio.transform import Affine

    from .geodesy import file_crs

    output = np.where(np.isfinite(grid), grid, nodata).astype(np.float32)
    height, width = output.shape
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", width=width, height=height, count=1,
        dtype="float32", crs=file_crs(crs_name),
        transform=Affine(cell, 0.0, bounds[0], 0.0, -cell, bounds[3]),
        nodata=nodata, tiled=True, blockxsize=256, blockysize=256,
        compress="DEFLATE", BIGTIFF="IF_SAFER",
    ) as sink:
        sink.write(output, 1)
    try:
        from . import raster

        raster.build_overviews(path)
    except Exception:
        pass


def _snapped_bounds(xs: np.ndarray, ys: np.ndarray, cell: float) -> tuple:
    """Cell edges on whole multiples of the cell size, enclosing every point."""
    return (math.floor(xs.min() / cell) * cell, math.floor(ys.min() / cell) * cell,
            (math.floor(xs.max() / cell) + 1) * cell, (math.floor(ys.max() / cell) + 1) * cell)


def _sample_grid(grid: np.ndarray, bounds: tuple, cell: float,
                 xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinear value of a grid (on cell centres) at each point."""
    from scipy.ndimage import map_coordinates

    cols = (xs - bounds[0]) / cell - 0.5
    rows = (bounds[3] - ys) / cell - 0.5
    return map_coordinates(grid, [rows, cols], order=1, mode="nearest")


def _fill_all(grid: np.ndarray) -> np.ndarray:
    """Fill every empty cell: linearly inside the data, nearest outside it."""
    holes = ~np.isfinite(grid)
    if not holes.any() or (~holes).sum() < 4:
        return grid
    return _fill_voids(grid, "linear", max(grid.shape))


# Points that are known noise never take part in ground or height work.
NOISE_CLASSES = (7, 18)


@dataclass
class GroundOptions:
    """Simple Morphological Filter (Pingel, Clarke and McBride, 2013).

    The defaults are the paper's recommended values, which hold across urban,
    forested and mixed ground.
    """
    output_path: str = ""
    cell_size: float = 1.0          # m, of the working minimum surface
    slope: float = 0.15             # rise over run the ground may have
    window: float = 18.0            # m, the widest object to remove (largest building)
    elevation_threshold: float = 0.5   # m, how far off the ground surface still counts
    elevation_scalar: float = 1.25  # extra tolerance per unit of local slope
    low_noise: bool = True          # label points far below the ground as class 7
    low_noise_threshold: float = 2.0   # m below the neighbourhood's ground
    crs: Optional[str] = None


def _disk(radius: int) -> np.ndarray:
    span = np.arange(-radius, radius + 1)
    return (span[:, None] ** 2 + span[None, :] ** 2) <= radius * radius


def classify_ground(
    path: str,
    options: GroundOptions,
    progress: Optional[Callable[[float, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> dict:
    """Label the ground in a cloud and write the result as a new LAS/LAZ file.

    Only the ground labels change: points found to be ground become class 2,
    points that were class 2 but are not ground become class 1, and every
    other class (vegetation, buildings, water, anything a vendor assigned) is
    kept. Known noise (classes 7 and 18) is left out of the search and kept.
    The source file is never modified.
    """
    import laspy
    from scipy.ndimage import grey_opening, median_filter

    def step(fraction, message):
        if progress:
            progress(fraction, message)
        if should_cancel and should_cancel():
            raise InterruptedError("Cancelled")

    step(0.03, "Reading point cloud")
    las = laspy.read(path)
    _resolve_crs(las.header, options.crs)
    classification = np.asarray(las.classification).astype(np.int64)
    xs = np.asarray(las.x, dtype=np.float64)
    ys = np.asarray(las.y, dtype=np.float64)
    zs = np.asarray(las.z, dtype=np.float64)
    candidate = ~np.isin(classification, NOISE_CLASSES)
    if candidate.sum() < 10:
        raise ValueError("The cloud has too few points to find the ground in.")

    cell = float(options.cell_size)
    bounds = _snapped_bounds(xs[candidate], ys[candidate], cell)

    step(0.15, "Building the minimum surface")
    zmin, _ = _bin_points(xs[candidate], ys[candidate], zs[candidate], bounds, cell, "minimum", 2.0)
    zmin = _fill_all(zmin)

    low = np.zeros(xs.shape, dtype=bool)
    if options.low_noise:
        # A point well below the ground around it is a multipath or sensor
        # artefact; left in, it would drag the minimum surface down with it.
        neighbourhood = median_filter(zmin, size=5, mode="nearest")
        reference = _sample_grid(neighbourhood, bounds, cell, xs, ys)
        low = candidate & (zs < reference - options.low_noise_threshold)
        if low.any():
            keep = candidate & ~low
            zmin, _ = _bin_points(xs[keep], ys[keep], zs[keep], bounds, cell, "minimum", 2.0)
            zmin = _fill_all(zmin)

    step(0.3, "Opening the surface")
    # A progressive opening with ever-larger disks: anything that stands
    # further above the opened surface than the slope allows over that
    # distance is an object, not ground.
    objects = np.zeros(zmin.shape, dtype=bool)
    last = zmin
    widest = max(1, int(math.ceil(options.window / cell)))
    for radius in range(1, widest + 1):
        opened = grey_opening(last, footprint=_disk(radius), mode="nearest")
        objects |= (last - opened) > options.slope * radius * cell
        last = opened
        step(0.3 + 0.4 * radius / widest, f"Opening the surface ({radius}/{widest})")

    step(0.72, "Interpolating the ground")
    provisional = np.where(objects, np.nan, zmin)
    provisional = _fill_all(provisional)
    gradient_rows, gradient_cols = np.gradient(provisional, cell)
    slope = np.hypot(gradient_rows, gradient_cols)

    step(0.85, "Labelling points")
    surface = _sample_grid(provisional, bounds, cell, xs, ys)
    local_slope = _sample_grid(slope, bounds, cell, xs, ys)
    tolerance = options.elevation_threshold + options.elevation_scalar * local_slope
    ground = candidate & ~low & (np.abs(zs - surface) <= tolerance)

    updated = classification.copy()
    updated[(classification == 2) & ~ground] = 1
    updated[ground] = 2
    updated[low] = 7
    las.classification = updated.astype(np.asarray(las.classification).dtype)

    step(0.93, "Writing point cloud")
    Path(options.output_path).parent.mkdir(parents=True, exist_ok=True)
    las.write(options.output_path)
    step(1.0, "Complete")

    total = int(xs.size)
    return {
        "outputPath": options.output_path,
        "pointsTotal": total,
        "groundPoints": int(ground.sum()),
        "groundFraction": float(ground.sum() / max(total, 1)),
        "lowNoisePoints": int(low.sum()),
        "relabelledFromGround": int(((classification == 2) & ~ground).sum()),
        "cellSize": cell,
        "objectCells": float(objects.mean()),
    }


HEIGHT_METHODS = ("highest", "pit_free")


@dataclass
class HeightOptions:
    """Height above ground: a normalised DSM, or a canopy height model over vegetation."""
    output_path: str = ""
    cell_size: float = 0.5
    returns: str = "first"               # all | first | last | single
    classes: Sequence[int] = ()          # empty: every class except noise
    method: str = "highest"              # highest | pit_free
    dtm_path: Optional[str] = None       # ground from this raster instead of class 2
    ground_cell_size: float = 1.0        # m, of the DTM made from class 2
    max_height: Optional[float] = None   # drop points higher than this above ground
    pit_free_thresholds: Sequence[float] = (0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0)
    pit_free_max_edge: Optional[float] = None  # m, longest triangle edge above the first layer;
                                               # None: five times the point spacing, at least 1 m
    void_fill: str = "linear"
    max_void_radius_cells: int = 4
    crs: Optional[str] = None
    nodata: float = -9999.0


def _ground_surface(las, options: HeightOptions, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """The ground height under every point, from a DTM raster or class 2."""
    if options.dtm_path:
        from scipy.ndimage import map_coordinates

        from . import raster

        dataset = raster.open_raster(options.dtm_path)
        dtm = dataset.read(1, masked=True).astype(np.float64).filled(np.nan)
        inverse = ~dataset.transform
        cols, rows = inverse * (xs, ys)
        values = map_coordinates(_fill_all(dtm), [np.asarray(rows) - 0.5, np.asarray(cols) - 0.5],
                                 order=1, mode="nearest")
        return values

    classification = np.asarray(las.classification)
    ground = classification == 2
    if ground.sum() < 3:
        raise ValueError(
            "The cloud has no ground points (class 2). Classify the ground first, "
            "or give a terrain model to measure heights from.")
    gx = np.asarray(las.x, dtype=np.float64)[ground]
    gy = np.asarray(las.y, dtype=np.float64)[ground]
    gz = np.asarray(las.z, dtype=np.float64)[ground]
    cell = options.ground_cell_size
    bounds = _snapped_bounds(np.concatenate([gx, xs]), np.concatenate([gy, ys]), cell)
    dtm, _ = _bin_points(gx, gy, gz, bounds, cell, "mean", 2.0)
    return _sample_grid(_fill_all(dtm), bounds, cell, xs, ys)


def _tin_grid(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, bounds: tuple, cell: float,
              max_edge: Optional[float]) -> np.ndarray:
    """A triangulated surface sampled at cell centres; NaN under long-edged triangles."""
    from scipy.spatial import Delaunay

    west, south, east, north = bounds
    height, width = _grid_shape(bounds, cell)
    grid = np.full((height, width), np.nan)
    if xs.size < 3:
        return grid
    # Triangulate in coordinates local to the grid: at map coordinates in the
    # millions the triangulation loses the precision it needs to locate cells.
    try:
        triangulation = Delaunay(np.column_stack([xs - west, ys - south]))
    except Exception:
        return grid

    cx = (np.arange(width) + 0.5) * cell
    cy = (north - south) - (np.arange(height) + 0.5) * cell
    gx, gy = np.meshgrid(cx, cy)
    targets = np.column_stack([gx.ravel(), gy.ravel()])
    simplex = triangulation.find_simplex(targets)
    inside = simplex >= 0
    if max_edge:
        corners = triangulation.points[triangulation.simplices]
        edges = np.stack([
            np.hypot(*(corners[:, 0] - corners[:, 1]).T),
            np.hypot(*(corners[:, 1] - corners[:, 2]).T),
            np.hypot(*(corners[:, 2] - corners[:, 0]).T),
        ], axis=1).max(axis=1)
        short = edges <= max_edge
        inside &= np.where(simplex >= 0, short[np.maximum(simplex, 0)], False)

    chosen = simplex[inside]
    transform = triangulation.transform[chosen]
    offset = targets[inside] - transform[:, 2]
    first_two = np.einsum("nij,nj->ni", transform[:, :2], offset)
    weights = np.column_stack([first_two, 1.0 - first_two.sum(axis=1)])
    values = (zs[triangulation.simplices[chosen]] * weights).sum(axis=1)
    flat = grid.ravel()
    flat[np.nonzero(inside)[0]] = values
    return flat.reshape(height, width)


def _highest_per_cell(xs, ys, zs, bounds, cell):
    """The single highest point of each cell, as point arrays."""
    west, south, east, north = bounds
    height, width = _grid_shape(bounds, cell)
    cols = np.clip(((xs - west) / cell).astype(np.int64), 0, width - 1)
    rows = np.clip(((north - ys) / cell).astype(np.int64), 0, height - 1)
    flat = rows * width + cols
    order = np.lexsort((-zs, flat))
    first = np.ones(order.size, dtype=bool)
    first[1:] = flat[order][1:] != flat[order][:-1]
    chosen = order[first]
    return xs[chosen], ys[chosen], zs[chosen]


def height_above_ground(
    path: str,
    options: HeightOptions,
    progress: Optional[Callable[[float, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> dict:
    """Rasterise each point's height above the ground.

    Over vegetation this is a canopy height model, over buildings a normalised
    DSM. Two methods:

    - highest: the highest point in each cell, the usual quick model.
    - pit_free (Khosravipour et al., 2014): triangulated surfaces of the
      points above a ladder of heights, each discarding triangles with long
      edges, combined by taking the highest. A first return that slipped deep
      into a crown then cannot punch a pit in it.
    """
    import laspy

    def step(fraction, message):
        if progress:
            progress(fraction, message)
        if should_cancel and should_cancel():
            raise InterruptedError("Cancelled")

    if options.method not in HEIGHT_METHODS:
        raise ValueError(f"Unknown method {options.method!r}; use one of {HEIGHT_METHODS}")

    step(0.03, "Reading point cloud")
    las = laspy.read(path)
    crs_name = _resolve_crs(las.header, options.crs)
    classification = np.asarray(las.classification)

    keep = _filter_points(las, RasterizeOptions(classes=list(options.classes), returns=options.returns))
    # Ground points stay in: they are what makes open ground read zero rather
    # than empty, and both methods keep the highest value, so they never pull
    # a canopy down.
    keep &= ~np.isin(classification, NOISE_CLASSES)
    if keep.sum() < 3:
        raise ValueError("No points survived the class and return filter.")

    xs = np.asarray(las.x, dtype=np.float64)[keep]
    ys = np.asarray(las.y, dtype=np.float64)[keep]
    zs = np.asarray(las.z, dtype=np.float64)[keep]

    step(0.2, "Measuring the ground under each point")
    heights = zs - _ground_surface(las, options, xs, ys)
    usable = np.isfinite(heights)
    dropped_high = 0
    if options.max_height is not None:
        too_high = heights > options.max_height
        dropped_high = int((too_high & usable).sum())
        usable &= ~too_high
    xs, ys = xs[usable], ys[usable]
    heights = np.maximum(heights[usable], 0.0)
    if heights.size < 3:
        raise ValueError("No points are left once their heights are measured.")

    cell = float(options.cell_size)
    # The grid covers the whole cloud, so heights line up with the cloud's
    # other rasters; cells with no points stay empty.
    bounds = _snapped_bounds(np.asarray(las.x)[~np.isin(classification, NOISE_CLASSES)],
                             np.asarray(las.y)[~np.isin(classification, NOISE_CLASSES)], cell)

    max_edge = None
    if options.method == "highest":
        step(0.5, f"Binning {heights.size:,} points at {cell} m")
        grid, filled = _bin_points(xs, ys, heights, bounds, cell, "maximum", 2.0)
        grid = _fill_voids(grid, options.void_fill, options.max_void_radius_cells)
    else:
        # Thin to the highest point in each half cell first: the triangulation
        # only ever needs the top of the canopy, and it keeps a dense cloud fast.
        tx, ty, th = _highest_per_cell(xs, ys, heights, bounds, cell / 2)
        # The edge limit has to bridge the gap a deep return leaves in a crown,
        # and that gap scales with the spacing of the points, so it is set from
        # the data unless given.
        _, occupied = _bin_points(tx, ty, th, bounds, cell, "maximum", 2.0)
        spacing = math.sqrt(occupied.sum() * cell * cell / max(th.size, 1))
        max_edge = options.pit_free_max_edge or max(1.0, 5.0 * spacing)
        layers = sorted({0.0, *[float(t) for t in options.pit_free_thresholds]})
        layers = [t for t in layers if t == 0.0 or t < th.max()]
        grid = np.full(_grid_shape(bounds, cell), np.nan)
        for index, threshold in enumerate(layers):
            step(0.3 + 0.6 * index / len(layers), f"Surface above {threshold:g} m")
            above = th >= threshold
            layer = _tin_grid(tx[above], ty[above], th[above], bounds, cell,
                              None if threshold == 0.0 else max_edge)
            grid = np.fmax(grid, layer)
        # The first layer spans the convex hull; keep only cells near real points.
        _, filled = _bin_points(xs, ys, heights, bounds, cell, "maximum", 2.0)
        from scipy.ndimage import binary_dilation

        grid = np.where(binary_dilation(filled, iterations=options.max_void_radius_cells), grid, np.nan)

    step(0.93, "Writing raster")
    _write_raster(options.output_path, grid, bounds, cell, crs_name, options.nodata)
    step(1.0, "Complete")

    valid = grid[np.isfinite(grid)]
    return {
        "outputPath": options.output_path,
        "width": int(grid.shape[1]),
        "height": int(grid.shape[0]),
        "cellSize": cell,
        "bounds": list(bounds),
        "crs": crs_name,
        "method": options.method,
        "pitFreeMaxEdge": float(max_edge) if options.method == "pit_free" else None,
        "returns": options.returns,
        "classes": list(options.classes),
        "groundSource": options.dtm_path or "class 2",
        "pointsUsed": int(heights.size),
        "pointsAboveMaxHeight": dropped_high,
        "maxHeight": float(valid.max()) if valid.size else 0.0,
        "height95": float(np.percentile(valid, 95)) if valid.size else 0.0,
        "coverage": float(np.isfinite(grid).mean()),
    }


def rasterize(
    path: str,
    options: RasterizeOptions,
    progress: Optional[Callable[[float, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> dict:
    """Filter a point cloud and write an elevation GeoTIFF."""
    import laspy

    if progress:
        progress(0.05, "Reading point cloud")

    with laspy.open(path) as reader:
        header = reader.header
        points = reader.read()

    if should_cancel and should_cancel():
        raise InterruptedError("Cancelled")

    crs_name = _resolve_crs(header, options.crs)
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

    if progress:
        progress(0.9, "Writing raster")
    height, width = grid.shape
    _write_raster(options.output_path, grid, bounds, cell, crs_name, options.nodata)

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
