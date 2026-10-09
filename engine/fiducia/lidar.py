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
    "NoiseOptions", "filter_noise", "overview", "section",
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


def inspect_cloud(path: str, sample_limit: Optional[int] = None) -> LidarSummary:
    """Summarise a cloud: counts by class and return, extent, spacing.

    Every point is counted, in one streamed pass that also sorts the cloud
    into tiles for the tools that follow; a second look is instant.
    """
    from .lidar_tiles import TiledCloud

    cloud = TiledCloud(path)
    m = cloud.manifest
    west, south, east, north = m["bounds"]
    area = max((east - west) * (north - south), 1e-9)
    return LidarSummary(
        path=str(Path(path).resolve()),
        point_count=m["points"],
        bounds=tuple(m["bounds"]),
        crs=m["crs"],
        version=m["lasVersion"],
        point_format=m["pointFormat"],
        class_histogram={int(k): int(v) for k, v in m["classHistogram"].items()},
        return_histogram={int(k): int(v) for k, v in m["returnHistogram"].items()},
        elevation_range=tuple(m["elevationRange"]),
        average_spacing=float(np.sqrt(area / max(m["points"], 1))),
    )


def _keep(a: dict, classes: Sequence[int], returns: str) -> np.ndarray:
    """Which points of a chunk pass a class and return filter."""
    keep = np.ones(a["classification"].shape, dtype=bool)
    if classes:
        keep &= np.isin(a["classification"], np.asarray(list(classes), dtype=np.int64))
    if returns == "first":
        keep &= a["return_number"] == 1
    elif returns == "last":
        keep &= a["return_number"] == a["number_of_returns"]
    elif returns == "single":
        keep &= a["number_of_returns"] == 1
    return keep


def _grid_shape(bounds: tuple, cell: float) -> tuple[int, int]:
    """(rows, columns) of a grid of the given cell size over the bounds."""
    west, south, east, north = bounds
    return (max(1, int(np.ceil((north - south) / cell))),
            max(1, int(np.ceil((east - west) / cell))))


class _Grid:
    """Points binned into cells, a chunk at a time: the grid is the only state,
    so a cloud of any size streams through it in bounded memory."""

    def __init__(self, bounds: tuple, cell: float, assignment: str, idw_power: float = 2.0):
        self.bounds, self.cell, self.assignment, self.idw_power = bounds, cell, assignment, idw_power
        self.shape = _grid_shape(bounds, cell)
        size = self.shape[0] * self.shape[1]
        if assignment == "minimum":
            self.a = np.full(size, np.inf)
        elif assignment == "maximum":
            self.a = np.full(size, -np.inf)
        elif assignment == "nearest":
            self.a = np.full(size, np.inf)       # distance of the nearest so far
            self.b = np.full(size, np.nan)       # its height
        else:                                    # mean, idw: weighted sum and weight
            self.a = np.zeros(size)
            self.b = np.zeros(size)

    def add(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> None:
        if xs.size == 0:
            return
        west, south, east, north = self.bounds
        height, width = self.shape
        cell = self.cell
        cols = np.clip(((xs - west) / cell).astype(np.int64), 0, width - 1)
        rows = np.clip(((north - ys) / cell).astype(np.int64), 0, height - 1)
        flat = rows * width + cols
        if self.assignment == "minimum":
            # np.minimum.at is the correct scatter-reduce here; a plain fancy-index
            # assignment would keep an arbitrary point per cell, not the lowest.
            np.minimum.at(self.a, flat, zs)
        elif self.assignment == "maximum":
            np.maximum.at(self.a, flat, zs)
        elif self.assignment == "mean":
            np.add.at(self.a, flat, zs)
            np.add.at(self.b, flat, 1.0)
        else:
            centre_x = west + (cols + 0.5) * cell
            centre_y = north - (rows + 0.5) * cell
            distance = np.hypot(xs - centre_x, ys - centre_y)
            if self.assignment == "nearest":
                # The height of the one point closest to each cell's centre:
                # sort by cell, then by distance, keep the first of each cell,
                # and keep it only if nearer than any earlier chunk's.
                order = np.lexsort((distance, flat))
                first = np.ones(order.size, dtype=bool)
                first[1:] = flat[order][1:] != flat[order][:-1]
                chosen = order[first]
                nearer = distance[chosen] < self.a[flat[chosen]]
                self.a[flat[chosen][nearer]] = distance[chosen][nearer]
                self.b[flat[chosen][nearer]] = zs[chosen][nearer]
            else:  # idw -- weight by distance from the cell centre
                weights = 1.0 / np.power(np.maximum(distance, cell * 0.01), self.idw_power)
                np.add.at(self.a, flat, zs * weights)
                np.add.at(self.b, flat, weights)

    def result(self) -> np.ndarray:
        if self.assignment in ("minimum", "maximum"):
            grid = np.where(np.isfinite(self.a), self.a, np.nan)
        elif self.assignment == "nearest":
            grid = self.b.copy()
        else:
            grid = np.where(self.b > 0, self.a / np.maximum(self.b, 1e-12), np.nan)
        return grid.reshape(self.shape)


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
    grid = _Grid(bounds, cell, assignment, idw_power)
    grid.add(xs, ys, zs)
    result = grid.result()
    return result, np.isfinite(result)


def _fill_voids(grid: np.ndarray, method: str, max_radius: int) -> np.ndarray:
    """Interpolate across cells that received no points."""
    if method == "none":
        return grid

    holes = ~np.isfinite(grid)
    if not holes.any() or np.isfinite(grid).sum() < 4:
        return grid

    from scipy.interpolate import griddata
    from scipy.ndimage import binary_dilation, distance_transform_edt

    # Only fill voids within reach of real data; a hole in the middle of a
    # lake should stay a hole rather than being invented.
    distance = distance_transform_edt(holes)
    fillable = holes & (distance <= max_radius)
    if not fillable.any():
        return grid

    # A hole is interpolated from the cells around its rim; the rest of the
    # grid plays no part, and leaving it out keeps a large grid affordable.
    rim = np.isfinite(grid) & binary_dilation(fillable, iterations=3)
    known_rows, known_cols = np.nonzero(rim)
    if known_rows.size < 4:
        known_rows, known_cols = np.nonzero(np.isfinite(grid))
    known_values = grid[known_rows, known_cols]
    target_rows, target_cols = np.nonzero(fillable)

    scipy_method = {
        "natural_neighbor": "cubic",   # closest available analogue
        "linear": "linear",
        "nearest": "nearest",
    }.get(method, "linear")

    try:
        filled = griddata(
            np.column_stack([known_rows, known_cols]),
            known_values,
            np.column_stack([target_rows, target_cols]),
            method=scipy_method,
        )
    except Exception:
        # A rim with every cell in one row cannot be triangulated.
        filled = np.full(target_rows.size, np.nan)

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

    The filter works on a minimum surface, so the cloud is streamed: once for
    that surface, once more without the low noise it finds, and once to label
    and write, holding only the grid.
    """
    import laspy
    from scipy.ndimage import grey_opening, median_filter

    from .lidar_tiles import arrays, ensure_memory, header_of, stream

    def step(fraction, message):
        if progress:
            progress(fraction, message)
        if should_cancel and should_cancel():
            raise InterruptedError("Cancelled")

    header = header_of(path)
    _resolve_crs(header, options.crs)
    n = int(header.point_count)
    cell = float(options.cell_size)
    bounds = _snapped_bounds(np.array([header.x_min, header.x_max]),
                             np.array([header.y_min, header.y_max]), cell)
    rows, cols = _grid_shape(bounds, cell)
    ensure_memory(rows * cols * 8 * 8, "Finding the ground at this cell size")

    def minimum_surface(skip_low, message, start, end):
        grid = _Grid(bounds, cell, "minimum")
        candidates = 0
        for first, points in stream(path):
            a = arrays(points)
            use = ~np.isin(a["classification"], NOISE_CLASSES)
            if skip_low is not None:
                use &= ~skip_low(a)
            grid.add(a["x"][use], a["y"][use], a["z"][use])
            candidates += int(use.sum())
            step(start + (end - start) * (first + use.size) / max(n, 1), message)
        return _fill_all(grid.result()), candidates

    zmin, candidates = minimum_surface(None, "Building the minimum surface", 0.02, 0.2)
    if candidates < 10:
        raise ValueError("The cloud has too few points to find the ground in.")

    low_of = None
    if options.low_noise:
        # A point well below the ground around it is a multipath or sensor
        # artefact; left in, it would drag the minimum surface down with it.
        neighbourhood = median_filter(zmin, size=5, mode="nearest")

        def low_of(a):
            reference = _sample_grid(neighbourhood, bounds, cell, a["x"], a["y"])
            return (~np.isin(a["classification"], NOISE_CLASSES)) & (a["z"] < reference - options.low_noise_threshold)

        zmin, _ = minimum_surface(low_of, "Rebuilding it without low noise", 0.2, 0.38)

    step(0.4, "Opening the surface")
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
        step(0.4 + 0.25 * radius / widest, f"Opening the surface ({radius}/{widest})")

    step(0.66, "Interpolating the ground")
    provisional = _fill_all(np.where(objects, np.nan, zmin))
    gradient_rows, gradient_cols = np.gradient(provisional, cell)
    slope = np.hypot(gradient_rows, gradient_cols)

    Path(options.output_path).parent.mkdir(parents=True, exist_ok=True)
    totals = {"ground": 0, "low": 0, "relabelled": 0}
    with laspy.open(path) as reader, laspy.open(options.output_path, mode="w", header=reader.header) as writer:
        first = 0
        for points in reader.chunk_iterator(2_000_000):
            a = arrays(points)
            candidate = ~np.isin(a["classification"], NOISE_CLASSES)
            low = low_of(a) if low_of is not None else np.zeros(candidate.shape, dtype=bool)
            surface = _sample_grid(provisional, bounds, cell, a["x"], a["y"])
            tolerance = options.elevation_threshold + options.elevation_scalar * _sample_grid(
                slope, bounds, cell, a["x"], a["y"])
            ground = candidate & ~low & (np.abs(a["z"] - surface) <= tolerance)
            updated = a["classification"].copy()
            relabelled = (a["classification"] == 2) & ~ground
            updated[relabelled] = 1
            updated[ground] = 2
            updated[low] = 7
            points.classification = updated.astype(np.asarray(points.classification).dtype)
            writer.write_points(points)
            totals["ground"] += int(ground.sum())
            totals["low"] += int(low.sum())
            totals["relabelled"] += int(relabelled.sum())
            first += len(points)
            step(0.7 + 0.28 * first / max(n, 1), "Labelling and writing points")
    step(1.0, "Complete")

    return {
        "outputPath": options.output_path,
        "pointsTotal": n,
        "groundPoints": totals["ground"],
        "groundFraction": float(totals["ground"] / max(n, 1)),
        "lowNoisePoints": totals["low"],
        "relabelledFromGround": totals["relabelled"],
        "cellSize": cell,
        "objectCells": float(objects.mean()),
    }


# -- noise ------------------------------------------------------------------

NOISE_METHODS = ("statistical", "isolated")


@dataclass
class NoiseOptions:
    """Points that do not belong to any surface: birds, multipath, sensor spikes.

    statistical: a point whose mean distance to its nearest neighbours is far
    above the cloud's typical spacing (Rusu et al., 2008).
    isolated: a point with too few neighbours within a radius.
    """
    output_path: str = ""
    method: str = "statistical"
    neighbours: int = 8              # statistical: how many neighbours to measure
    std_ratio: float = 2.5           # statistical: how many standard deviations is too far
    radius: float = 2.0              # isolated: m, the search radius
    min_neighbours: int = 3          # isolated: fewer than this within the radius is noise
    crs: Optional[str] = None


def filter_noise(
    path: str,
    options: NoiseOptions,
    progress: Optional[Callable[[float, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> dict:
    """Label noise in a cloud and write the result as a new LAS/LAZ file.

    Noise is labelled, never deleted: a point below the real points around it
    becomes low noise (class 7), one above them high noise (class 18), and
    every rasterising and height tool then leaves it out. Points already
    labelled noise are kept, and the source file is never modified.

    The cloud is measured tile by tile, each with a margin of its neighbours'
    points, so a point at a tile's edge finds the same neighbours as anywhere.
    """
    from scipy.spatial import cKDTree

    from .lidar_tiles import TiledCloud, ensure_memory, header_of, release, scratch, write_classified

    def step(fraction, message):
        if progress:
            progress(fraction, message)
        if should_cancel and should_cancel():
            raise InterruptedError("Cancelled")

    if options.method not in NOISE_METHODS:
        raise ValueError(f"Unknown method {options.method!r}; use one of {NOISE_METHODS}")
    _resolve_crs(header_of(path), options.crs)
    cloud = TiledCloud(path, lambda f, m: step(0.02 + 0.2 * f, m), should_cancel)
    k = max(2, int(options.neighbours))
    if options.method == "statistical":
        margin = max(1.0, 6.0 * cloud.spacing * math.sqrt(k / 8.0))
    else:
        margin = float(options.radius)
    ensure_memory(int(cloud.largest_tile * (1 + 4 * margin / cloud.tile_size) * 400), "Finding noise")

    measure = scratch(cloud.n, np.float32, fill=np.nan)
    original = scratch(cloud.n, np.uint8)
    tiles = sorted(cloud.tiles)
    try:
        for done, tile in enumerate(tiles):
            step(0.22 + 0.4 * done / len(tiles), f"Measuring neighbourhoods, tile {done + 1} of {len(tiles)}")
            d = cloud.load(tile, margin)
            core = d["core"]
            original[d["index"][core]] = d["classification"][core]
            candidate = ~np.isin(d["classification"], NOISE_CLASSES)
            if candidate.sum() < k + 2:
                continue
            x0, y0, _, _ = cloud.tile_bounds(tile)
            xyz = np.column_stack([d["x"][candidate] - x0, d["y"][candidate] - y0, d["z"][candidate]])
            tree = cKDTree(xyz)
            target = core[candidate]
            if not target.any():
                continue
            if options.method == "statistical":
                distance, _ = tree.query(xyz[target], k=k + 1, workers=-1)
                distance = np.where(np.isfinite(distance), distance, margin * 10)
                value = distance[:, 1:].mean(axis=1)
            else:
                value = tree.query_ball_point(xyz[target], r=options.radius, return_length=True, workers=-1) - 1
            measure[d["index"][candidate][target]] = value

        step(0.64, "Setting the limit")
        if options.method == "statistical":
            total = count = squares = 0.0
            for lo in range(0, cloud.n, 5_000_000):
                part = np.asarray(measure[lo:lo + 5_000_000], dtype=np.float64)
                part = part[np.isfinite(part)]
                total += part.sum()
                squares += (part * part).sum()
                count += part.size
            mean = total / max(count, 1)
            limit = mean + options.std_ratio * math.sqrt(max(squares / max(count, 1) - mean * mean, 0.0))

            def is_noise(values):
                return np.isfinite(values) & (values > limit)
        else:
            def is_noise(values):
                return np.isfinite(values) & (values < options.min_neighbours)

        # Low or high is judged against the lowest surface of the real points
        # around it, not against its neighbours: a flock of birds has only
        # other birds for neighbours.
        floor_cell = 2.0
        bounds = _snapped_bounds(np.array(cloud.bounds[0::2]), np.array(cloud.bounds[1::2]), floor_cell)
        floor = _Grid(bounds, floor_cell, "minimum")
        for done, tile in enumerate(tiles):
            step(0.66 + 0.12 * done / len(tiles), "Finding the floor under the noise")
            d = cloud.load(tile, 0.0)
            real = ~np.isin(d["classification"], NOISE_CLASSES) & ~is_noise(np.asarray(measure[d["index"]]))
            floor.add(d["x"][real], d["y"][real], d["z"][real])
        floor = _fill_all(floor.result())

        counts = {"low": 0, "high": 0}
        for done, tile in enumerate(tiles):
            step(0.78 + 0.1 * done / len(tiles), "Labelling noise")
            d = cloud.load(tile, 0.0)
            noisy = is_noise(np.asarray(measure[d["index"]]))
            if not noisy.any():
                continue
            below = d["z"][noisy] < _sample_grid(floor, bounds, floor_cell, d["x"][noisy], d["y"][noisy])
            original[d["index"][noisy][below]] = 7
            original[d["index"][noisy][~below]] = 18
            counts["low"] += int(below.sum())
            counts["high"] += int((~below).sum())

        step(0.9, "Writing point cloud")
        write_classified(path, options.output_path, original,
                         lambda f: step(0.9 + 0.09 * f, "Writing point cloud"))
    finally:
        release(measure)
        release(original)
    step(1.0, "Complete")
    noise = counts["low"] + counts["high"]
    return {
        "outputPath": options.output_path,
        "method": options.method,
        "pointsTotal": cloud.n,
        "noisePoints": noise,
        "lowNoisePoints": counts["low"],
        "highNoisePoints": counts["high"],
        "noiseFraction": float(noise / max(cloud.n, 1)),
    }


# -- looking at a cloud -----------------------------------------------------

# Colours for drawing classes, shared with the interface.
CLASS_COLOURS = {
    0: "#8a9299", 1: "#b6bec4", 2: "#b5895a", 3: "#9fd67a", 4: "#58b858", 5: "#1f7a4d",
    6: "#d9534f", 7: "#ff3df2", 8: "#f0c75e", 9: "#3f8fd2", 10: "#7a5c3e", 11: "#5d6670",
    12: "#e0b45c", 13: "#ffb347", 14: "#ff8c42", 15: "#c9725b", 17: "#8e7cc3", 18: "#ff3df2",
}

def _ramp(values: np.ndarray) -> np.ndarray:
    """Low to high as deep blue, teal, green, yellow, white."""
    stops = np.array([[24, 40, 92], [23, 145, 127], [111, 208, 112], [240, 200, 80], [250, 250, 245]], float)
    position = np.clip(values, 0, 1) * (len(stops) - 1)
    lower = np.floor(position).astype(int).clip(0, len(stops) - 2)
    t = (position - lower)[..., None]
    return (stops[lower] * (1 - t) + stops[lower + 1] * t).astype(np.uint8)


def _hex(colour: str) -> tuple:
    return tuple(int(colour[i:i + 2], 16) for i in (1, 3, 5))


def overview(path: str, size: int = 1024, colour: str = "height") -> dict:
    """A top-down picture of the cloud to draw section lines on.

    Each pixel shows the highest point in it, by height or by class, shaded
    so that trees and buildings stand out from the ground. The pixels are made
    when the cloud is first indexed, so this reads no points at all.
    """
    import base64
    import io

    from PIL import Image

    from .lidar_tiles import TiledCloud

    cloud = TiledCloud(path)
    grid, class_grid = cloud.overview_grids()
    info = cloud.manifest["overview"]
    cell = info["cell"]
    height, width = grid.shape
    class_grid = class_grid.astype(np.int64)
    # Close single-pixel gaps from the nearest filled pixel.
    from scipy.ndimage import distance_transform_edt

    empty = ~np.isfinite(grid)
    if empty.any() and (~empty).any():
        distance, (near_r, near_c) = distance_transform_edt(empty, return_indices=True)
        close = empty & (distance <= 1.5)
        grid[close] = grid[near_r[close], near_c[close]]
        class_grid[close] = class_grid[near_r[close], near_c[close]]
    filled = np.isfinite(grid)

    if colour == "class":
        palette = np.zeros((256, 3), dtype=np.uint8)
        palette[:] = _hex("#b6bec4")
        for code, value in CLASS_COLOURS.items():
            palette[code] = _hex(value)
        rgb = palette[class_grid.clip(0, 255)]
    else:
        low, high = np.nanpercentile(grid, [2, 98]) if filled.any() else (0.0, 1.0)
        rgb = _ramp((np.nan_to_num(grid, nan=low) - low) / max(high - low, 1e-6))

    # A light hillshade, so relief reads in either colouring.
    smooth = np.where(filled, grid, np.nanmedian(grid) if filled.any() else 0.0)
    dy, dx = np.gradient(smooth, cell)
    shade = np.clip(0.75 + 0.25 * (-dx - dy) / np.sqrt(1 + dx * dx + dy * dy), 0.45, 1.0)
    rgb = (rgb * shade[..., None]).astype(np.uint8)
    alpha = np.where(filled, 255, 0).astype(np.uint8)
    image = Image.fromarray(np.dstack([rgb, alpha]), "RGBA")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return {
        "image": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(),
        "bounds": list(info["bounds"]),
        "width": width,
        "height": height,
        "cellSize": cell,
        "elevationRange": [float(np.nanmin(grid)), float(np.nanmax(grid))] if filled.any() else [0, 0],
    }


def section(path: str, start: Sequence[float], end: Sequence[float], width: float = 2.0,
            max_points: int = 150_000) -> dict:
    """The points in a corridor between two map points, seen from the side.

    Only the tiles the corridor crosses are read. Each point comes back with
    its index in the file, so points labelled in this view can be found again
    for training.
    """
    from .lidar_tiles import TiledCloud

    sx, sy = float(start[0]), float(start[1])
    ex, ey = float(end[0]), float(end[1])
    length = math.hypot(ex - sx, ey - sy)
    if length <= 0:
        raise ValueError("A section needs two different end points.")
    half = width / 2
    cloud = TiledCloud(path)
    d = cloud.load_box(min(sx, ex) - half, min(sy, ey) - half, max(sx, ex) + half, max(sy, ey) + half)
    ux, uy = (ex - sx) / length, (ey - sy) / length
    dx, dy = d["x"] - sx, d["y"] - sy
    along = dx * ux + dy * uy
    across = -dx * uy + dy * ux
    inside = np.nonzero((np.abs(across) <= half) & (along >= 0) & (along <= length))[0]
    total = int(inside.size)
    if inside.size > max_points:
        # An even thinning that a repeated request reproduces exactly.
        inside = inside[np.linspace(0, inside.size - 1, max_points).astype(np.int64)]
    inside = inside[np.argsort(along[inside], kind="stable")]
    return {
        "length": length,
        "width": width,
        "total": total,
        "shown": int(inside.size),
        "index": d["index"][inside].tolist(),
        "along": np.round(along[inside], 3).tolist(),
        "z": np.round(d["z"][inside], 3).tolist(),
        "classification": d["classification"][inside].tolist(),
        "returnNumber": d["return_number"][inside].tolist(),
        "numberOfReturns": d["number_of_returns"][inside].tolist(),
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
    pit_free_thresholds: Optional[Sequence[float]] = None  # m; None: every 2.5 m from the ground up
    pit_free_step: float = 2.5
    pit_free_max_edge: Optional[float] = None  # m, longest triangle edge above the first layer;
                                               # None: five times the point spacing, at least 1 m
    void_fill: str = "linear"
    max_void_radius_cells: int = 4
    crs: Optional[str] = None
    nodata: float = -9999.0


def _ground_sampler(path: str, options: "HeightOptions", bounds: tuple):
    """A function giving the ground height at any points, from a DTM or class 2.

    From a DTM, only the window under each tile's points is read. From the
    cloud, the ground points are streamed into a terrain model first.
    """
    from scipy.ndimage import map_coordinates

    if options.dtm_path:
        from rasterio.windows import Window

        from . import raster

        dataset = raster.open_raster(options.dtm_path)
        inverse = ~dataset.transform

        def sample(xs, ys):
            cols, rows = inverse * (xs, ys)
            cols, rows = np.asarray(cols), np.asarray(rows)
            c0 = int(max(0, np.floor(cols.min()) - 2))
            r0 = int(max(0, np.floor(rows.min()) - 2))
            c1 = int(min(dataset.width, np.ceil(cols.max()) + 3))
            r1 = int(min(dataset.height, np.ceil(rows.max()) + 3))
            if c1 <= c0 or r1 <= r0:
                return np.full(xs.shape, np.nan)
            window = dataset.read(1, window=Window(c0, r0, c1 - c0, r1 - r0), masked=True)
            grid = _fill_all(window.astype(np.float64).filled(np.nan))
            return map_coordinates(grid, [rows - r0 - 0.5, cols - c0 - 0.5], order=1, mode="nearest")

        return sample

    from .lidar_tiles import arrays, ensure_memory, stream

    cell = options.ground_cell_size
    gbounds = _snapped_bounds(np.array(bounds[0::2]), np.array(bounds[1::2]), cell)
    rows, cols = _grid_shape(gbounds, cell)
    ensure_memory(rows * cols * 8 * 6, "The terrain model under the heights")
    grid = _Grid(gbounds, cell, "mean")
    found = 0
    for _, points in stream(path):
        a = arrays(points)
        ground = a["classification"] == 2
        grid.add(a["x"][ground], a["y"][ground], a["z"][ground])
        found += int(ground.sum())
    if found < 3:
        raise ValueError(
            "The cloud has no ground points (class 2). Classify the ground first, "
            "or give a terrain model to measure heights from.")
    dtm = _fill_all(grid.result())
    return lambda xs, ys: _sample_grid(dtm, gbounds, cell, xs, ys)


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


def _height_grid(xs, ys, heights, bounds, cell, options: "HeightOptions", max_edge) -> np.ndarray:
    """Heights rasterised over one window, by the chosen method."""
    if options.method == "highest":
        grid, _ = _bin_points(xs, ys, heights, bounds, cell, "maximum", 2.0)
        return _fill_voids(grid, options.void_fill, options.max_void_radius_cells)
    # Thin to the highest point in each half cell first: the triangulation
    # only ever needs the top of the canopy, and it keeps a dense cloud fast.
    tx, ty, th = _highest_per_cell(xs, ys, heights, bounds, cell / 2)
    # The published ladder (0, 2, 5, 10, 15 m...) leaves pits up to a step
    # deep wherever a return lands inside a crown; a step of 2.5 m closed
    # them in testing for about 15% more time.
    ladder = (options.pit_free_thresholds if options.pit_free_thresholds is not None
              else np.arange(0.0, float(th.max()) if th.size else 0.0, options.pit_free_step))
    layers = sorted({0.0, *[float(t) for t in ladder]})
    layers = [t for t in layers if t == 0.0 or (th.size and t < th.max())]
    grid = np.full(_grid_shape(bounds, cell), np.nan)
    for threshold in layers:
        above = th >= threshold
        layer = _tin_grid(tx[above], ty[above], th[above], bounds, cell,
                          None if threshold == 0.0 else max_edge)
        grid = np.fmax(grid, layer)
    # The first layer spans the convex hull; keep only cells near real points.
    from scipy.ndimage import binary_dilation

    _, filled = _bin_points(xs, ys, heights, bounds, cell, "maximum", 2.0)
    return np.where(binary_dilation(filled, iterations=options.max_void_radius_cells), grid, np.nan)


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

    The raster is made tile by tile, each from its own points and a margin of
    its neighbours', and written window by window, so neither the cloud nor
    the raster has to fit in memory.
    """
    import rasterio
    from rasterio.transform import Affine
    from rasterio.windows import Window

    from .geodesy import file_crs
    from .lidar_tiles import TiledCloud, ensure_memory, header_of

    def step(fraction, message):
        if progress:
            progress(fraction, message)
        if should_cancel and should_cancel():
            raise InterruptedError("Cancelled")

    if options.method not in HEIGHT_METHODS:
        raise ValueError(f"Unknown method {options.method!r}; use one of {HEIGHT_METHODS}")
    crs_name = _resolve_crs(header_of(path), options.crs)
    cloud = TiledCloud(path, lambda f, m: step(0.02 + 0.15 * f, m), should_cancel)
    cell = float(options.cell_size)
    west, south, east, north = cloud.bounds
    # The grid covers the whole cloud, so heights line up with the cloud's
    # other rasters; cells with no points stay empty.
    bounds = _snapped_bounds(np.array([west, east]), np.array([south, north]), cell)
    rows_total, cols_total = _grid_shape(bounds, cell)

    step(0.18, "Measuring the ground")
    ground_at = _ground_sampler(path, options, cloud.bounds)
    tiles = sorted(cloud.tiles)
    pad = options.max_void_radius_cells + 2
    edge_x = (min(t[0] for t in tiles), max(t[0] for t in tiles))
    edge_y = (min(t[1] for t in tiles), max(t[1] for t in tiles))

    def window_of(tile):
        """Output cells whose centres fall in the tile; the outer tiles take the rim."""
        tw, ts, te, tn = cloud.tile_bounds(tile)
        c0 = 0 if tile[0] == edge_x[0] else int(math.ceil((tw - bounds[0]) / cell - 0.5))
        c1 = cols_total - 1 if tile[0] == edge_x[1] else int(math.ceil((te - bounds[0]) / cell - 0.5)) - 1
        r0 = 0 if tile[1] == edge_y[1] else int(math.floor((bounds[3] - tn) / cell - 0.5)) + 1
        r1 = rows_total - 1 if tile[1] == edge_y[0] else int(math.floor((bounds[3] - ts) / cell - 0.5))
        c0, c1 = max(c0, 0), min(c1, cols_total - 1)
        r0, r1 = max(r0, 0), min(r1, rows_total - 1)
        return c0, c1, r0, r1

    def heights_near(tile, margin):
        d = cloud.load(tile, margin)
        keep = _keep(d, options.classes, options.returns) & ~np.isin(d["classification"], NOISE_CLASSES)
        xs, ys = d["x"][keep], d["y"][keep]
        h = d["z"][keep] - ground_at(xs, ys)
        usable = np.isfinite(h)
        high = 0
        if options.max_height is not None:
            too_high = h > options.max_height
            high = int((too_high & usable & d["core"][keep]).sum())
            usable &= ~too_high
        core = d["core"][keep][usable]
        return xs[usable], ys[usable], np.maximum(h[usable], 0.0), core, high

    max_edge = None
    if options.method == "pit_free":
        # The edge limit has to bridge the gap a deep return leaves in a crown,
        # and that gap scales with the spacing of the points, so it is set from
        # the whole cloud's spacing unless given.
        thinned = occupied = 0
        for done, tile in enumerate(tiles):
            step(0.25 + 0.1 * done / len(tiles), "Measuring the point spacing")
            c0, c1, r0, r1 = window_of(tile)
            if c1 < c0 or r1 < r0:
                continue
            wb = (bounds[0] + c0 * cell, bounds[3] - (r1 + 1) * cell,
                  bounds[0] + (c1 + 1) * cell, bounds[3] - r0 * cell)
            xs, ys, h, core, _ = heights_near(tile, 0.0)
            inside = (xs >= wb[0]) & (xs < wb[2]) & (ys >= wb[1]) & (ys < wb[3])
            if not inside.any():
                continue
            tx, ty, th = _highest_per_cell(xs[inside], ys[inside], h[inside], wb, cell / 2)
            _, filled = _bin_points(tx, ty, th, wb, cell, "maximum", 2.0)
            thinned += th.size
            occupied += int(filled.sum())
        spacing = math.sqrt(occupied * cell * cell / max(thinned, 1))
        max_edge = options.pit_free_max_edge or max(1.0, 5.0 * spacing)

    margin = max(pad * cell, 2.0 * (max_edge or 0.0), 3.0 * cell)
    ensure_memory(int(cloud.largest_tile * (1 + 4 * margin / cloud.tile_size) * 500), "The height model")

    Path(options.output_path).parent.mkdir(parents=True, exist_ok=True)
    used = dropped_high = filled_cells = 0
    peak = -np.inf
    sample = []
    with rasterio.open(
        options.output_path, "w", driver="GTiff", width=cols_total, height=rows_total, count=1,
        dtype="float32", crs=file_crs(crs_name),
        transform=Affine(cell, 0.0, bounds[0], 0.0, -cell, bounds[3]),
        nodata=options.nodata, tiled=True, blockxsize=256, blockysize=256,
        compress="DEFLATE", BIGTIFF="IF_SAFER",
    ) as sink:
        # Cells no tile writes (no points there) must read as nodata.
        blank = np.full((256, cols_total), options.nodata, dtype=np.float32)
        for r in range(0, rows_total, 256):
            h_rows = min(256, rows_total - r)
            sink.write(blank[:h_rows], 1, window=Window(0, r, cols_total, h_rows))
        for done, tile in enumerate(tiles):
            step(0.36 + 0.58 * done / len(tiles), f"Heights, tile {done + 1} of {len(tiles)}")
            c0, c1, r0, r1 = window_of(tile)
            if c1 < c0 or r1 < r0:
                continue
            # Work on the window grown by a few cells, so that void filling and
            # triangles see across the tile's edge, then keep the window.
            e0, e1 = max(c0 - pad, 0), min(c1 + pad, cols_total - 1)
            f0, f1 = max(r0 - pad, 0), min(r1 + pad, rows_total - 1)
            eb = (bounds[0] + e0 * cell, bounds[3] - (f1 + 1) * cell,
                  bounds[0] + (e1 + 1) * cell, bounds[3] - f0 * cell)
            xs, ys, h, core, high = heights_near(tile, margin)
            dropped_high += high
            used += int(core.sum())
            inside = (xs >= eb[0]) & (xs < eb[2]) & (ys >= eb[1]) & (ys < eb[3])
            if inside.sum() < 3:
                continue
            grid = _height_grid(xs[inside], ys[inside], h[inside], eb, cell, options, max_edge)
            grid = grid[r0 - f0:r1 - f0 + 1, c0 - e0:c1 - e0 + 1]
            valid = grid[np.isfinite(grid)]
            if valid.size:
                filled_cells += valid.size
                peak = max(peak, float(valid.max()))
                sample.append(valid[:: max(1, valid.size // 20000)])
            sink.write(np.where(np.isfinite(grid), grid, options.nodata).astype(np.float32), 1,
                       window=Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1))
    if used < 3:
        raise ValueError("No points survived the class and return filter.")
    try:
        from . import raster

        raster.build_overviews(options.output_path)
    except Exception:
        pass
    step(1.0, "Complete")

    values = np.concatenate(sample) if sample else np.zeros(0)
    return {
        "outputPath": options.output_path,
        "width": int(cols_total),
        "height": int(rows_total),
        "cellSize": cell,
        "bounds": list(bounds),
        "crs": crs_name,
        "method": options.method,
        "pitFreeMaxEdge": float(max_edge) if max_edge else None,
        "returns": options.returns,
        "classes": list(options.classes),
        "groundSource": options.dtm_path or "class 2",
        "pointsUsed": int(used),
        "pointsAboveMaxHeight": int(dropped_high),
        "maxHeight": float(peak) if np.isfinite(peak) else 0.0,
        "height95": float(np.percentile(values, 95)) if values.size else 0.0,
        "coverage": float(filled_cells / max(rows_total * cols_total, 1)),
    }


def rasterize(
    path: str,
    options: RasterizeOptions,
    progress: Optional[Callable[[float, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> dict:
    """Filter a point cloud and write an elevation GeoTIFF.

    The cloud is streamed through the grid a chunk at a time, so only the
    grid has to fit in memory.
    """
    from .lidar_tiles import arrays, ensure_memory, header_of, stream

    header = header_of(path)
    crs_name = _resolve_crs(header, options.crs)
    total_points = int(header.point_count)
    cell = options.cell_size
    # Cell edges on whole multiples of the cell size, so the surface lines up
    # with every other raster made at that size.
    bounds = _snapped_bounds(np.array([header.x_min, header.x_max]),
                             np.array([header.y_min, header.y_max]), cell)
    rows, cols = _grid_shape(bounds, cell)
    ensure_memory(rows * cols * 8 * 6, "A raster at this cell size")

    grid = _Grid(bounds, cell, options.cell_assignment, options.idw_power)
    kept = 0
    seen: set = set()
    for start, points in stream(path):
        if should_cancel and should_cancel():
            raise InterruptedError("Cancelled")
        a = arrays(points)
        keep = _keep(a, options.classes, options.returns)
        grid.add(a["x"][keep], a["y"][keep], a["z"][keep])
        kept += int(keep.sum())
        seen.update(np.unique(a["classification"]).tolist())
        if progress:
            progress(0.05 + 0.6 * (start + keep.size) / max(total_points, 1),
                     f"Binning {start + keep.size:,} of {total_points:,} points at {cell} m")
    if kept == 0:
        raise ValueError(
            "No points survived the class and return filter. "
            f"Requested classes {list(options.classes)}; "
            f"file contains {sorted(seen)}."
        )

    raw = grid.result()
    filled_mask = np.isfinite(raw)
    void_count = int((~filled_mask).sum())
    if progress:
        progress(0.75, f"Filling {void_count:,} empty cells")
    result = _fill_voids(raw, options.void_fill, options.max_void_radius_cells)

    if progress:
        progress(0.9, "Writing raster")
    height, width = result.shape
    _write_raster(options.output_path, result, bounds, cell, crs_name, options.nodata)
    if progress:
        progress(1.0, "Complete")

    valid = result[np.isfinite(result)]
    return {
        "outputPath": options.output_path,
        "width": width,
        "height": height,
        "cellSize": options.cell_size,
        "bounds": list(bounds),
        "crs": crs_name,
        "pointsTotal": total_points,
        "pointsUsed": kept,
        "pointsRejected": total_points - kept,
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
