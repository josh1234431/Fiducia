"""Mosaicking: cutlines, colour balancing and feathered blending.

The approach is two-resolution, which is what keeps it both good-looking and
memory-bounded on blocks that would not fit in RAM:

1. A decimated overview of every input is used to derive the global
   decisions -- where the seams fall and what gain and offset each image needs
   so its tones agree with its neighbours.
2. The full-resolution pass then walks output tiles, reads only the inputs that
   touch each tile, and blends them using weights interpolated from the
   low-resolution seam map.

Seams are placed by distance transform: each pixel is claimed by whichever
image it sits deepest inside. That naturally keeps cutlines away from image
edges, where radiometric falloff and geometric error are worst, without
needing the operator to draw anything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import reproject
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds
from scipy.ndimage import distance_transform_edt, zoom

from .geodesy import file_crs

__all__ = ["MosaicSpec", "MosaicResult", "generate_mosaic", "mosaic_preview"]

# none: as delivered. linear: one offset per image and band, so overlaps
# agree in level. histogram: a gain and an offset, so they agree in level and
# spread (mean and standard deviation). histogram_match: full histogram
# matching, image by image in the mosaic order, to what is already placed.
COLOR_BALANCE_METHODS = ("none", "linear", "histogram", "histogram_match")
# distance: each pixel to the image it is deepest inside. nearest_center: to
# the image whose centre is nearest. order: painter's order. min_difference:
# seams routed where overlapping images agree, the least-cost path through a
# surface of their disagreement, which keeps cuts out of buildings and trees.
CUTLINE_METHODS = ("distance", "nearest_center", "order", "min_difference")
# Brightness variation within each image, removed before balancing between
# images (dodging). across_image: a second-order trend surface, for the
# fall-off from centre to edge. hotspot: a broad low-pass surface, which also
# follows the bright patch opposite the sun.
NORMALIZATION_METHODS = ("none", "hotspot", "across_image")


@dataclass
class MosaicSpec:
    inputs: list[str]
    output_path: str
    bounds: Optional[Sequence[float]] = None     # west, south, east, north
    pixel_size: Optional[float] = None
    output_crs: Optional[str] = None
    color_balance: str = "linear"
    cutline_method: str = "distance"
    normalization: str = "none"
    blend_width_px: float = 40.0
    starting_image: Optional[str] = None
    sort_method: str = "nearest_center"          # or "maximum_intersection"
    resampling: str = "nearest"
    nodata: float = 0.0
    compress: str = "DEFLATE"
    tile_size: int = 1024
    overview_max_px: int = 2400

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class MosaicResult:
    output_path: str
    width: int
    height: int
    bounds: tuple
    pixel_size: float
    band_count: int
    inputs_used: list[str]
    gains: dict
    offsets: dict
    elapsed_seconds: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "outputPath": self.output_path,
            "width": self.width,
            "height": self.height,
            "bounds": list(self.bounds),
            "pixelSize": self.pixel_size,
            "bandCount": self.band_count,
            "inputsUsed": self.inputs_used,
            "gains": self.gains,
            "offsets": self.offsets,
            "elapsedSeconds": self.elapsed_seconds,
            "warnings": self.warnings,
        }


def _describe_inputs(paths: Sequence[str]) -> list[dict]:
    described = []
    for path in paths:
        with rasterio.open(path) as dataset:
            described.append(
                {
                    "path": str(path),
                    "bounds": tuple(dataset.bounds),
                    "crs": dataset.crs.to_string() if dataset.crs else None,
                    "count": dataset.count,
                    "dtype": dataset.dtypes[0],
                    "nodata": dataset.nodata,
                    "res": dataset.res,
                    "center": (
                        (dataset.bounds.left + dataset.bounds.right) / 2.0,
                        (dataset.bounds.bottom + dataset.bounds.top) / 2.0,
                    ),
                }
            )
    return described


def _union_bounds(described: list[dict]) -> tuple:
    west = min(d["bounds"][0] for d in described)
    south = min(d["bounds"][1] for d in described)
    east = max(d["bounds"][2] for d in described)
    north = max(d["bounds"][3] for d in described)
    return west, south, east, north


def _order_inputs(described: list[dict], spec: MosaicSpec) -> list[dict]:
    """Order images in the conventional way, since it drives balancing."""
    if not described:
        return []

    remaining = list(described)
    if spec.starting_image:
        for i, entry in enumerate(remaining):
            if Path(entry["path"]).name == Path(spec.starting_image).name:
                first = remaining.pop(i)
                break
        else:
            first = remaining.pop(0)
    else:
        # "Auto": the most central scene starts.
        centre_x = np.mean([d["center"][0] for d in described])
        centre_y = np.mean([d["center"][1] for d in described])
        distances = [
            math.hypot(d["center"][0] - centre_x, d["center"][1] - centre_y) for d in remaining
        ]
        first = remaining.pop(int(np.argmin(distances)))

    ordered = [first]
    while remaining:
        if spec.sort_method == "maximum_intersection":
            def score(entry):
                return max(_overlap_area(entry["bounds"], done["bounds"]) for done in ordered)
            remaining.sort(key=score, reverse=True)
        else:
            anchor = ordered[0]["center"]
            remaining.sort(
                key=lambda e: math.hypot(e["center"][0] - anchor[0], e["center"][1] - anchor[1])
            )
        ordered.append(remaining.pop(0))
    return ordered


def _overlap_area(a: tuple, b: tuple) -> float:
    west = max(a[0], b[0])
    south = max(a[1], b[1])
    east = min(a[2], b[2])
    north = min(a[3], b[3])
    if east <= west or north <= south:
        return 0.0
    return (east - west) * (north - south)


def _read_to_grid(
    path: str,
    transform: Affine,
    width: int,
    height: int,
    crs,
    bands: Sequence[int],
    resampling: str,
    nodata: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject/resample one input onto the mosaic grid. Returns (data, mask)."""
    method = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
    }.get(resampling, Resampling.nearest)

    with rasterio.open(path) as source:
        destination = np.full((len(bands), height, width), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(source, list(bands)),
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=transform,
            dst_crs=crs,
            dst_nodata=np.nan,
            resampling=method,
        )
        source_nodata = source.nodata

    mask = np.isfinite(destination).all(axis=0)
    if source_nodata is not None:
        mask &= ~np.all(destination == source_nodata, axis=0)
    # An all-zero pixel in an ortho is background, not black ground.
    mask &= ~np.all(np.nan_to_num(destination) == nodata, axis=0)
    return np.nan_to_num(destination, nan=nodata), mask


def _read_hidden(path: str, transform: Affine, width: int, height: int, crs,
                 resampling=Resampling.nearest) -> Optional[np.ndarray]:
    """An ortho's hidden-ground mask on the mosaic grid, or None if it has none."""
    sidecar = str(path) + ".hidden.tif"
    if not Path(sidecar).exists():
        return None
    with rasterio.open(sidecar) as source:
        destination = np.zeros((height, width), dtype=np.uint8)
        reproject(source=rasterio.band(source, 1), destination=destination,
                  src_transform=source.transform, src_crs=source.crs,
                  dst_transform=transform, dst_crs=crs, resampling=resampling)
    return destination > 0


def _solve_color_balance(
    overviews: list[np.ndarray],
    masks: list[np.ndarray],
    method: str,
    band_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-image, per-band gain and offset that make overlaps agree.

    Solves the classic global formulation: for every overlapping pair, the
    corrected means should match and the corrected spreads should match. The
    first image is pinned to gain 1 / offset 0 so the whole block does not
    drift, and the system is solved in the log domain for gains, which keeps
    them positive without needing a constrained solver.
    """
    n = len(overviews)
    gains = np.ones((n, band_count))
    offsets = np.zeros((n, band_count))

    if method == "none" or n < 2:
        return gains, offsets

    for band in range(band_count):
        pairs: list[tuple[int, int, float, float, float, float]] = []
        for i in range(n):
            for j in range(i + 1, n):
                overlap = masks[i] & masks[j]
                count = int(overlap.sum())
                if count < 50:
                    continue
                a = overviews[i][band][overlap]
                b = overviews[j][band][overlap]
                pairs.append(
                    (i, j, float(a.mean()), float(b.mean()),
                     float(a.std()) or 1.0, float(b.std()) or 1.0)
                )

        if not pairs:
            continue

        if method == "histogram":
            # Match spread as well as level.
            rows = np.zeros((len(pairs) + 1, n))
            rhs = np.zeros(len(pairs) + 1)
            for k, (i, j, _, _, sa, sb) in enumerate(pairs):
                rows[k, i] = 1.0
                rows[k, j] = -1.0
                rhs[k] = math.log(max(sb, 1e-6)) - math.log(max(sa, 1e-6))
            rows[-1, 0] = 1.0
            rhs[-1] = 0.0
            log_gain, *_ = np.linalg.lstsq(rows, rhs, rcond=None)
            gains[:, band] = np.clip(np.exp(log_gain), 0.25, 4.0)

        # Offsets: corrected means agree, with the first image pinned.
        rows = np.zeros((len(pairs) + 1, n))
        rhs = np.zeros(len(pairs) + 1)
        for k, (i, j, ma, mb, _, _) in enumerate(pairs):
            rows[k, i] = 1.0
            rows[k, j] = -1.0
            rhs[k] = gains[j, band] * mb - gains[i, band] * ma
        rows[-1, 0] = 1.0
        rhs[-1] = 0.0
        solved, *_ = np.linalg.lstsq(rows, rhs, rcond=None)
        offsets[:, band] = solved

    return gains, offsets


def _normalisation_factors(
    overviews: list[np.ndarray],
    masks: list[np.ndarray],
    method: str,
) -> Optional[list[np.ndarray]]:
    """Per-image multiplicative surfaces that flatten brightness within it.

    Returns, per image, a (bands, h, w) factor on the overview grid, or None
    for no normalisation. The image's mean is kept: only the variation across
    it is removed, so balancing between images still has the levels to work
    with.
    """
    from scipy.ndimage import gaussian_filter

    if method not in ("hotspot", "across_image"):
        return None

    factors = []
    for data, mask in zip(overviews, masks):
        bands, h, w = data.shape
        factor = np.ones((bands, h, w), dtype=np.float32)
        if mask.sum() < 100:
            factors.append(factor)
            continue
        rows, cols = np.nonzero(mask)
        # Coordinates scaled to about unity, for a well-conditioned fit.
        cr, cc = rows.mean(), cols.mean()
        span = max(np.ptp(rows), np.ptp(cols), 1)
        for band in range(bands):
            values = data[band][mask].astype(np.float64)
            mean = float(values.mean())
            if mean <= 0:
                continue
            if method == "across_image":
                y, x = (rows - cr) / span, (cols - cc) / span
                design = np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])
                # Ignore the extremes (water, cloud, roofs) in the fit.
                lo, hi = np.percentile(values, [5, 95])
                keep = (values >= lo) & (values <= hi)
                coeffs, *_ = np.linalg.lstsq(design[keep], values[keep], rcond=None)
                grid_r, grid_c = np.mgrid[0:h, 0:w]
                gy, gx = (grid_r - cr) / span, (grid_c - cc) / span
                surface = (coeffs[0] + coeffs[1] * gx + coeffs[2] * gy + coeffs[3] * gx * gx
                           + coeffs[4] * gx * gy + coeffs[5] * gy * gy)
            else:
                # Normalised convolution: a wide blur of the image over its own
                # footprint only, so the background does not darken its edges.
                sigma = max(h, w) / 8.0
                filled = np.where(mask, data[band], 0.0).astype(np.float64)
                weight = gaussian_filter(mask.astype(np.float64), sigma)
                surface = gaussian_filter(filled, sigma) / np.maximum(weight, 1e-6)
            band_factor = np.clip(mean / np.maximum(surface, 1e-6), 0.5, 2.0)
            factor[band] = np.where(mask, band_factor, 1.0).astype(np.float32)
        factors.append(factor)
    return factors


def _histogram_tables(
    overviews: list[np.ndarray],
    masks: list[np.ndarray],
    band_count: int,
    quantiles: int = 101,
) -> list[Optional[list[tuple[np.ndarray, np.ndarray]]]]:
    """Histogram matching, image by image in the mosaic order.

    The first image is the reference. Each later image is matched, in its
    overlap, to what the images before it already show there (after their own
    matching), by mapping its quantiles onto theirs. Returns per image a list
    of (source quantiles, target quantiles) per band, or None where an image
    overlaps nothing placed before it.
    """
    probabilities = np.linspace(0.0, 100.0, quantiles)
    placed_sum = np.zeros_like(overviews[0], dtype=np.float64)
    placed_count = np.zeros(masks[0].shape, dtype=np.float64)
    tables: list = []
    for index, (data, mask) in enumerate(zip(overviews, masks)):
        table = None
        overlap = mask & (placed_count > 0)
        if index > 0 and overlap.sum() >= 200:
            table = []
            for band in range(band_count):
                source = np.percentile(data[band][overlap], probabilities)
                target = np.percentile(placed_sum[band][overlap] / placed_count[overlap],
                                       probabilities)
                # Strictly increasing, as interpolation needs.
                source = np.maximum.accumulate(source + np.arange(quantiles) * 1e-6)
                table.append((source, target))
        tables.append(table)
        matched = _apply_tables(data, table)
        placed_sum += np.where(mask[None], matched, 0.0)
        placed_count += mask
    return tables


def _apply_tables(data: np.ndarray, table) -> np.ndarray:
    if not table:
        return data
    out = np.empty_like(data, dtype=np.float32)
    for band, (source, target) in enumerate(table):
        out[band] = np.interp(data[band], source, target).astype(np.float32)
    return out


def _min_difference_labels(
    overviews: list[np.ndarray],
    masks: list[np.ndarray],
    max_px: int = 700,
) -> np.ndarray:
    """Which image each overview pixel takes, with seams on low disagreement.

    A geodesic Voronoi partition: every image grows from its own core
    (where it is alone, or deepest inside its footprint) across a cost
    surface that is cheap where the overlapping images agree and expensive
    where they differ. Where two growths meet is the seam, so seams run along
    roads and open ground rather than through buildings, whose lean differs
    between photos.
    """
    from scipy.ndimage import zoom as nd_zoom
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    n = len(masks)
    shape = masks[0].shape
    step = max(1, int(math.ceil(max(shape) / max_px)))
    small_masks = [m[::step, ::step] for m in masks]
    small_shape = small_masks[0].shape
    luminance = [d.mean(axis=0)[::step, ::step] for d in overviews]

    # Disagreement: mean absolute difference to the other images covering
    # each pixel, relative to its typical size.
    stack = np.stack([np.where(m, l, np.nan) for m, l in zip(small_masks, luminance)])
    count = np.isfinite(stack).sum(axis=0)
    with np.errstate(invalid="ignore"), np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            centre = np.nanmedian(stack, axis=0)
            spread = np.nanmean(np.abs(stack - centre[None]), axis=0)
    spread = np.where(count >= 2, np.nan_to_num(spread), 0.0)
    typical = float(np.median(spread[count >= 2])) if (count >= 2).any() else 1.0
    cost = 1.0 + 8.0 * spread / max(typical, 1e-6)

    h, w = small_shape
    index = np.arange(h * w).reshape(h, w)
    best = np.full((h, w), np.inf)
    labels = np.full((h, w), -1, dtype=np.int32)
    for i, mask in enumerate(small_masks):
        if not mask.any():
            continue
        # Edges between 4-neighbours that both lie in this image.
        rows, cols, weights = [], [], []
        for a, b in (((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                     ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
            both = mask[a] & mask[b]
            rows.append(index[a][both])
            cols.append(index[b][both])
            weights.append(0.5 * (cost[a][both] + cost[b][both]))
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        weights = np.concatenate(weights)
        graph = coo_matrix((weights, (rows, cols)), shape=(h * w, h * w)).tocsr()

        alone = mask & (count == 1)
        depth = distance_transform_edt(mask)
        core = alone | (depth >= 0.8 * depth.max())
        sources = index[core]
        if sources.size == 0:
            continue
        distance = dijkstra(graph, directed=False, indices=sources, min_only=True)
        distance = distance.reshape(h, w)
        better = mask & (distance < best)
        best[better] = distance[better]
        labels[better] = i

    if step > 1:
        labels = nd_zoom(labels, (shape[0] / h, shape[1] / w), order=0)[:shape[0], :shape[1]]
        if labels.shape != shape:
            padded = np.full(shape, -1, dtype=np.int32)
            padded[:labels.shape[0], :labels.shape[1]] = labels
            labels = padded
    # Any pixel the coarse grid missed goes to the deepest covering image.
    missing = (labels < 0) & np.any(np.stack(masks), axis=0)
    if missing.any():
        depths = np.stack([distance_transform_edt(m) for m in masks])
        labels[missing] = np.argmax(depths, axis=0)[missing]
    return labels


def _seam_weights(
    masks: list[np.ndarray],
    method: str,
    blend_px: float,
    order_index: list[int],
    overviews: Optional[list[np.ndarray]] = None,
) -> np.ndarray:
    """Per-image blend weights over the low-resolution grid.

    Weight is driven by how deep a pixel sits inside each image's valid area,
    so the seam lands on the locus where two images are equally interior.
    ``blend_px`` controls how sharply the handover happens.
    """
    n = len(masks)
    shape = masks[0].shape
    distances = np.zeros((n, *shape), dtype=np.float32)

    for i, mask in enumerate(masks):
        if not mask.any():
            continue
        distances[i] = distance_transform_edt(mask).astype(np.float32)

    if method == "min_difference" and overviews is not None:
        labels = _min_difference_labels(overviews, masks)
        # Feathered across the seam by the blend width: each image's weight
        # rises from 0 to 1 over the blend width centred on its boundary.
        sharpness = max(blend_px, 1e-3)
        weights = np.zeros_like(distances)
        for i, mask in enumerate(masks):
            region = labels == i
            if not region.any():
                continue
            inside = distance_transform_edt(region)
            outside = distance_transform_edt(~region)
            signed = np.where(region, inside, -outside)
            weights[i] = np.where(mask, np.clip(0.5 + signed / (2.0 * sharpness), 0.0, 1.0), 0.0)
        total = weights.sum(axis=0)
        orphan = total <= 0
        if orphan.any():
            # Covered but outside every feather: the labelled image alone.
            for i in range(n):
                weights[i][orphan & (labels == i)] = 1.0
            total = weights.sum(axis=0)
        total[total <= 0] = 1.0
        return weights / total

    if method == "order":
        # Strict painter's algorithm: later images win outright.
        weights = np.zeros_like(distances)
        claimed = np.zeros(shape, dtype=bool)
        for i in order_index:
            take = masks[i] & ~claimed
            weights[i][take] = 1.0
            claimed |= take
        return weights

    if method == "nearest_center":
        centers = []
        for mask in masks:
            if mask.any():
                rows, cols = np.nonzero(mask)
                centers.append((rows.mean(), cols.mean()))
            else:
                centers.append((0.0, 0.0))
        grid_r, grid_c = np.indices(shape)
        proximity = np.zeros_like(distances)
        for i, (cr, cc) in enumerate(centers):
            d = np.hypot(grid_r - cr, grid_c - cc)
            proximity[i] = np.where(masks[i], 1.0 / (1.0 + d), 0.0)
        distances = proximity * 1000.0

    sharpness = max(blend_px, 1e-3)
    leader = distances.max(axis=0)
    weights = np.exp((distances - leader) / sharpness)
    for i, mask in enumerate(masks):
        weights[i][~mask] = 0.0

    total = weights.sum(axis=0)
    total[total <= 0] = 1.0
    return weights / total


def _plan(spec: MosaicSpec):
    """Shared setup for both preview and full generation."""
    described = _describe_inputs(spec.inputs)
    if not described:
        raise ValueError("No input images supplied to the mosaic")

    ordered = _order_inputs(described, spec)
    paths = [d["path"] for d in ordered]

    crs_name = spec.output_crs or ordered[0]["crs"]
    if not crs_name:
        raise ValueError("Mosaic inputs have no coordinate system and none was supplied")
    crs = file_crs(crs_name)

    pixel = spec.pixel_size or min(abs(d["res"][0]) for d in ordered)
    if spec.bounds:
        bounds = tuple(spec.bounds)
    else:
        # On whole multiples of the pixel size, the grid the orthos are made
        # on: each input pixel then lands on exactly one mosaic pixel.
        west, south, east, north = _union_bounds(ordered)
        bounds = (math.floor(west / pixel + 1e-9) * pixel, math.floor(south / pixel + 1e-9) * pixel,
                  math.ceil(east / pixel - 1e-9) * pixel, math.ceil(north / pixel - 1e-9) * pixel)

    band_count = min(d["count"] for d in ordered)
    dtype = ordered[0]["dtype"]

    return ordered, paths, crs, crs_name, bounds, pixel, band_count, dtype


def _overview_grid(bounds, pixel, max_px):
    west, south, east, north = bounds
    full_w = max(1, int(math.ceil((east - west) / pixel)))
    full_h = max(1, int(math.ceil((north - south) / pixel)))
    decimation = max(1, int(math.ceil(max(full_w, full_h) / max_px)))
    ov_w = max(1, full_w // decimation)
    ov_h = max(1, full_h // decimation)
    transform = Affine(
        (east - west) / ov_w, 0.0, west, 0.0, -(north - south) / ov_h, north
    )
    return ov_w, ov_h, transform, decimation, full_w, full_h


def _build_overviews(spec, paths, crs, bounds, pixel, band_count):
    ov_w, ov_h, ov_transform, decimation, full_w, full_h = _overview_grid(
        bounds, pixel, spec.overview_max_px
    )
    bands = list(range(1, band_count + 1))

    overviews, masks, coverage = [], [], []
    for path in paths:
        data, mask = _read_to_grid(
            path, ov_transform, ov_w, ov_h, crs, bands, "average", spec.nodata
        )
        overviews.append(data)
        coverage.append(mask)
        # Ground a photo could not see (behind a tree, a wall) is not its to
        # show: seams route around it to a photo that does see it.
        hidden = _read_hidden(path, ov_transform, ov_w, ov_h, crs, Resampling.max)
        masks.append(mask & ~hidden if hidden is not None else mask)

    return overviews, masks, coverage, ov_w, ov_h, ov_transform, decimation, full_w, full_h


def _radiometry(spec: MosaicSpec, overviews, masks, band_count: int):
    """Every tonal decision, made once on the overviews.

    In order: flatten each image (normalisation), balance images against one
    another (gain and offset), then, for histogram matching, map each image's
    histogram onto its predecessors'. Returns the per-image factors, gains,
    offsets and tables, and the corrected overviews.
    """
    n = len(overviews)
    factors = _normalisation_factors(overviews, masks, spec.normalization)
    flattened = [d * f for d, f in zip(overviews, factors)] if factors else list(overviews)
    method = spec.color_balance
    gains, offsets = _solve_color_balance(
        flattened, masks, "none" if method == "histogram_match" else method, band_count)
    corrected = [d * gains[i][:, None, None] + offsets[i][:, None, None]
                 for i, d in enumerate(flattened)]
    tables = (_histogram_tables(corrected, masks, band_count) if method == "histogram_match"
              else [None] * n)
    corrected = [_apply_tables(c, t) for c, t in zip(corrected, tables)]
    return factors, gains, offsets, tables, corrected


def _upsample_patch(layer: np.ndarray, c: int, r: int, w: int, h: int,
                    width: int, height: int, ov_w: int, ov_h: int) -> Optional[np.ndarray]:
    """A low-resolution layer's patch under one output tile, at tile size."""
    ov_c0 = int(c / width * ov_w)
    ov_r0 = int(r / height * ov_h)
    ov_c1 = max(ov_c0 + 1, int((c + w) / width * ov_w))
    ov_r1 = max(ov_r0 + 1, int((r + h) / height * ov_h))
    patch = layer[ov_r0:ov_r1, ov_c0:ov_c1]
    if patch.size == 0:
        return None
    up = zoom(patch, (h / patch.shape[0], w / patch.shape[1]), order=1)[:h, :w]
    if up.shape != (h, w):
        padded = np.empty((h, w), dtype=np.float32)
        padded[...] = up[-1:, -1:].mean() if up.size else 0.0
        padded[: up.shape[0], : up.shape[1]] = up
        up = padded
    return up


def mosaic_preview(spec: MosaicSpec, max_px: int = 900) -> dict:
    """Render a fast, low-resolution preview plus the seam map, as PNG bytes.

    This is what makes parameter tuning bearable: the operator changes colour
    balance or blend width and sees the result in under a second, instead of
    regenerating a full mosaic to find out.
    """
    import base64
    import io
    from PIL import Image

    preview_spec = MosaicSpec(**{**spec.to_dict(), "overview_max_px": max_px})
    ordered, paths, crs, crs_name, bounds, pixel, band_count, dtype = _plan(preview_spec)

    overviews, masks, _, ov_w, ov_h, _, _, full_w, full_h = _build_overviews(
        preview_spec, paths, crs, bounds, pixel, band_count
    )

    _, gains, offsets, _, corrected = _radiometry(spec, overviews, masks, band_count)
    scale = max(1.0, max(full_w, full_h) / max(ov_w, ov_h))
    weights = _seam_weights(
        masks, spec.cutline_method, max(spec.blend_width_px / scale, 0.5), list(range(len(masks))),
        corrected,
    )

    blended = np.zeros((band_count, ov_h, ov_w), dtype=np.float32)
    for i, data in enumerate(corrected):
        blended += data * weights[i][None, :, :]

    coverage = np.any(np.stack(masks), axis=0)

    def to_png(array_u8: np.ndarray) -> str:
        buffer = io.BytesIO()
        Image.fromarray(array_u8).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    valid = blended[:, coverage] if coverage.any() else blended.reshape(band_count, -1)
    lo = float(np.percentile(valid, 2)) if valid.size else 0.0
    hi = float(np.percentile(valid, 98)) if valid.size else 1.0
    span = max(hi - lo, 1e-6)
    display = np.clip((blended - lo) / span * 255.0, 0, 255).astype(np.uint8)

    if band_count >= 3:
        rgb = display[:3].transpose(1, 2, 0)
    else:
        rgb = np.repeat(display[:1], 3, axis=0).transpose(1, 2, 0)

    alpha = np.where(coverage, 255, 0).astype(np.uint8)
    rgba = np.dstack([rgb, alpha])

    # Seam map: which image dominates each pixel, in distinguishable hues.
    palette = np.array(
        [[27, 156, 133], [217, 119, 62], [92, 126, 214], [201, 84, 132],
         [140, 172, 70], [120, 108, 196], [214, 168, 58], [70, 160, 170]],
        dtype=np.uint8,
    )
    dominant = np.argmax(np.stack(weights), axis=0)
    seam_rgb = palette[dominant % len(palette)]
    seam_alpha = np.where(coverage, 190, 0).astype(np.uint8)
    seam = np.dstack([seam_rgb, seam_alpha])

    return {
        "previewPng": to_png(rgba),
        "seamPng": to_png(seam),
        "width": ov_w,
        "height": ov_h,
        "bounds": list(bounds),
        "fullWidth": full_w,
        "fullHeight": full_h,
        "pixelSize": pixel,
        "order": [Path(p).name for p in paths],
        "gains": {Path(p).name: gains[i].tolist() for i, p in enumerate(paths)},
        "offsets": {Path(p).name: offsets[i].tolist() for i, p in enumerate(paths)},
        "bandCount": band_count,
        "crs": crs_name,
    }


def generate_mosaic(
    spec: MosaicSpec,
    progress: Optional[Callable[[float, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> MosaicResult:
    """Write the full-resolution mosaic."""
    import time

    started = time.time()
    warnings: list[str] = []

    ordered, paths, crs, crs_name, bounds, pixel, band_count, dtype = _plan(spec)
    bands = list(range(1, band_count + 1))

    if progress:
        progress(0.02, "Analysing overlaps and tone")

    overviews, masks, coverage, ov_w, ov_h, ov_transform, decimation, width, height = \
        _build_overviews(spec, paths, crs, bounds, pixel, band_count)
    hidden_any = any(m is not c for m, c in zip(masks, coverage))

    factors, gains, offsets, tables, corrected_overviews = _radiometry(
        spec, overviews, masks, band_count)
    scale = max(1.0, max(width, height) / max(ov_w, ov_h))
    if progress and spec.cutline_method == "min_difference":
        progress(0.04, "Routing seams where the images agree")
    weights = _seam_weights(
        masks, spec.cutline_method, max(spec.blend_width_px / scale, 0.5), list(range(len(masks))),
        corrected_overviews,
    )
    # Where no photo sees the ground at all, the best of what they show
    # instead -- never a hole in the mosaic.
    fallback_weights = (_seam_weights(coverage, "distance", max(spec.blend_width_px / scale, 0.5),
                                      list(range(len(coverage))))
                        if hidden_any else None)
    del overviews, corrected_overviews

    west, south, east, north = bounds
    transform = Affine(pixel, 0.0, west, 0.0, -pixel, north)

    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": band_count,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": spec.nodata,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": spec.compress,
        "BIGTIFF": "IF_SAFER",
    }

    Path(spec.output_path).parent.mkdir(parents=True, exist_ok=True)

    info = np.iinfo(np.dtype(dtype)) if np.dtype(dtype).kind in "iu" else None
    tile = max(256, spec.tile_size)
    tiles = [
        (c, r, min(tile, width - c), min(tile, height - r))
        for r in range(0, height, tile)
        for c in range(0, width, tile)
    ]

    with rasterio.open(spec.output_path, "w", **profile) as sink:
        for index, (c, r, w, h) in enumerate(tiles):
            if should_cancel and should_cancel():
                Path(spec.output_path).unlink(missing_ok=True)
                raise InterruptedError("Mosaic generation cancelled")

            tile_transform = transform * Affine.translation(c, r)
            tile_bounds = (
                west + c * pixel,
                north - (r + h) * pixel,
                west + (c + w) * pixel,
                north - r * pixel,
            )

            accumulator = np.zeros((band_count, h, w), dtype=np.float32)
            weight_sum = np.zeros((h, w), dtype=np.float32)
            spare = np.zeros((band_count, h, w), dtype=np.float32) if hidden_any else None
            spare_sum = np.zeros((h, w), dtype=np.float32) if hidden_any else None

            for i, path in enumerate(paths):
                if _overlap_area(tile_bounds, ordered[i]["bounds"]) <= 0:
                    continue

                data, mask = _read_to_grid(
                    path, tile_transform, w, h, crs, bands, spec.resampling, spec.nodata
                )
                if not mask.any():
                    continue

                # This image's low-resolution seam weight, upsampled onto the tile.
                tile_weight = _upsample_patch(weights[i], c, r, w, h, width, height, ov_w, ov_h)
                if tile_weight is None:
                    continue

                hidden = (_read_hidden(path, tile_transform, w, h, crs) if hidden_any else None)
                if hidden is not None:
                    fallback = _upsample_patch(fallback_weights[i], c, r, w, h, width, height,
                                               ov_w, ov_h)
                    fallback = np.where(mask, np.clip(fallback, 0.0, None), 0.0) \
                        if fallback is not None else np.zeros((h, w), np.float32)
                    mask = mask & ~hidden
                elif hidden_any:
                    fallback = _upsample_patch(fallback_weights[i], c, r, w, h, width, height,
                                               ov_w, ov_h)
                    fallback = np.where(mask, np.clip(fallback, 0.0, None), 0.0) \
                        if fallback is not None else np.zeros((h, w), np.float32)
                tile_weight = np.where(mask, np.clip(tile_weight, 0.0, None), 0.0)
                if factors is not None:
                    for band in range(band_count):
                        flat = _upsample_patch(factors[i][band], c, r, w, h, width, height,
                                               ov_w, ov_h)
                        if flat is not None:
                            data[band] = data[band] * flat
                corrected = data * gains[i][:, None, None] + offsets[i][:, None, None]
                corrected = _apply_tables(corrected, tables[i])
                accumulator += corrected * tile_weight[None, :, :]
                weight_sum += tile_weight
                if hidden_any:
                    spare += corrected * fallback[None, :, :]
                    spare_sum += fallback

            if hidden_any:
                # Seen by no photo: take what they show rather than leave a hole.
                unseen = (weight_sum <= 1e-6) & (spare_sum > 1e-6)
                accumulator[:, unseen] = spare[:, unseen]
                weight_sum[unseen] = spare_sum[unseen]
            covered = weight_sum > 1e-6
            output = np.full((band_count, h, w), spec.nodata, dtype=np.float32)
            for band in range(band_count):
                channel = accumulator[band]
                output[band] = np.where(covered, channel / np.where(covered, weight_sum, 1.0),
                                        spec.nodata)

            if info is not None:
                output = np.clip(output, info.min, info.max)

            sink.write(output.astype(dtype), window=Window(c, r, w, h))

            if progress:
                progress(
                    0.05 + 0.93 * (index + 1) / len(tiles),
                    f"Tile {index + 1} of {len(tiles)}",
                )

    try:
        from . import raster

        raster.build_overviews(spec.output_path)
    except Exception:
        warnings.append("Mosaic written, but overview pyramid could not be built")

    if progress:
        progress(1.0, "Mosaic complete")

    return MosaicResult(
        output_path=spec.output_path,
        width=width,
        height=height,
        bounds=bounds,
        pixel_size=pixel,
        band_count=band_count,
        inputs_used=paths,
        gains={Path(p).name: gains[i].tolist() for i, p in enumerate(paths)},
        offsets={Path(p).name: offsets[i].tolist() for i, p in enumerate(paths)},
        elapsed_seconds=time.time() - started,
        warnings=warnings,
    )
