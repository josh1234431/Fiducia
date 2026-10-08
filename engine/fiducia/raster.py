"""Raster access, overview pyramids and tile rendering.

The viewer never loads a whole image. It requests 256-pixel tiles at a zoom
level, and this module serves them from the closest overview level, so panning
around a 12000 x 12000 scanned aerial costs the same as panning around a
thumbnail. That single decision is most of the difference between an interface
that feels instant and one that stalls every time the operator moves.

Formats come through GDAL via rasterio, so PCIDSK (.pix) files open
directly alongside GeoTIFF, JPEG and PNG -- existing
projects do not have to be converted before they can be used.
"""

from __future__ import annotations

import io
import logging
import math
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds

__all__ = [
    "RasterInfo",
    "open_raster",
    "raster_info",
    "read_window",
    "render_tile",
    "sample_elevation",
    "build_overviews",
    "ENHANCEMENTS",
]

_LOG = logging.getLogger("fiducia.raster")

TILE_SIZE = 256

ENHANCEMENTS = ("none", "linear", "linear2pct", "stddev", "equalize", "root")

_dataset_lock = threading.Lock()
_open_datasets: dict[tuple[str, int], rasterio.DatasetReader] = {}


def open_raster(path: str | Path) -> rasterio.DatasetReader:
    """Open (and cache) a dataset handle for the calling thread.

    GDAL dataset objects are expensive to create and cheap to keep, and the
    viewer hits the same handful of files thousands of times per session.
    A GDAL dataset must not be read from two threads at once, and tiles are
    served from a thread pool, so each thread gets its own handle. Sharing
    one crashed the process under concurrent tile requests.
    """
    key = (str(Path(path).resolve()), threading.get_ident())
    with _dataset_lock:
        dataset = _open_datasets.get(key)
        if dataset is not None and not dataset.closed:
            return dataset
    dataset = rasterio.open(key[0])
    with _dataset_lock:
        _open_datasets[key] = dataset
    return dataset


def close_raster(path: str | Path) -> None:
    """Close every thread's handle on this file, e.g. before it is rewritten."""
    resolved = str(Path(path).resolve())
    with _dataset_lock:
        keys = [key for key in _open_datasets if key[0] == resolved]
        datasets = [_open_datasets.pop(key) for key in keys]
    for dataset in datasets:
        if not dataset.closed:
            dataset.close()
    # Its display statistics describe the old contents too.
    for key in [k for k in list(_stats_cache) if k[0] == resolved]:
        _stats_cache.pop(key, None)


def close_all() -> None:
    with _dataset_lock:
        datasets = list(_open_datasets.values())
        _open_datasets.clear()
    for dataset in datasets:
        if not dataset.closed:
            dataset.close()


@dataclass
class RasterInfo:
    path: str
    width: int
    height: int
    band_count: int
    dtype: str
    crs: Optional[str]
    transform: tuple
    bounds: tuple
    nodata: Optional[float]
    overview_levels: list[int]
    has_georeferencing: bool
    driver: str

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "bandCount": self.band_count,
            "dtype": self.dtype,
            "crs": self.crs,
            "transform": list(self.transform),
            "bounds": list(self.bounds),
            "nodata": self.nodata,
            "overviewLevels": self.overview_levels,
            "hasGeoreferencing": self.has_georeferencing,
            "driver": self.driver,
            "maxZoom": max(0, math.ceil(math.log2(max(self.width, self.height) / TILE_SIZE)))
            if max(self.width, self.height) > TILE_SIZE
            else 0,
        }


def raster_info(path: str | Path) -> RasterInfo:
    dataset = open_raster(path)
    try:
        overviews = dataset.overviews(1)
    except Exception:
        overviews = []

    transform = dataset.transform
    # A raster with the identity transform is un-georeferenced -- a raw scan.
    identity = (
        abs(transform.a - 1.0) < 1e-12
        and abs(transform.e - 1.0) < 1e-12
        and abs(transform.c) < 1e-12
        and abs(transform.f) < 1e-12
    )

    return RasterInfo(
        path=str(Path(path).resolve()),
        width=dataset.width,
        height=dataset.height,
        band_count=dataset.count,
        dtype=str(dataset.dtypes[0]),
        crs=dataset.crs.to_string() if dataset.crs else None,
        transform=tuple(transform)[:6],
        bounds=tuple(dataset.bounds),
        nodata=dataset.nodata,
        overview_levels=list(overviews),
        has_georeferencing=bool(dataset.crs) and not identity,
        driver=dataset.driver,
    )


def build_overviews(path: str | Path, levels: Sequence[int] = (2, 4, 8, 16, 32)) -> list[int]:
    """Add internal overviews so the viewer can render at any zoom instantly.

    Falls back to an external .ovr sidecar when the format cannot carry
    internal overviews or the file is read-only -- the original is never
    modified in a way that could lose data.
    """
    target = str(Path(path).resolve())

    # A cached read handle blocks opening the same file for update on Windows,
    # and this is routinely called right after raster_info() has opened one.
    close_raster(target)

    try:
        with rasterio.open(target, "r+") as dataset:
            existing = dataset.overviews(1)
            if existing:
                return list(existing)
            useful = [
                lvl for lvl in levels
                if dataset.width // lvl >= TILE_SIZE or dataset.height // lvl >= TILE_SIZE
            ] or [2]
            dataset.build_overviews(useful, Resampling.average)
            dataset.update_tags(ns="rio_overview", resampling="average")
            return useful
    except Exception as exc:
        internal_error = exc

    # Read-only source, or a format that cannot carry internal overviews:
    # write an external .ovr sidecar instead. The original file is never
    # modified in a way that could lose data.
    close_raster(target)
    try:
        with rasterio.Env(TIFF_USE_OVR=True):
            with rasterio.open(target, "r") as dataset:
                useful = [
                    lvl for lvl in levels
                    if dataset.width // lvl >= TILE_SIZE or dataset.height // lvl >= TILE_SIZE
                ] or [2]
                dataset.build_overviews(useful, Resampling.average)
        return useful
    except Exception as exc:
        # Overviews are an optimisation, never a correctness requirement, so
        # this stays non-fatal -- but it must not be silent, or a viewer that
        # is mysteriously slow has no explanation anywhere.
        _LOG.warning(
            "Could not build overviews for %s (internal: %s; sidecar: %s)",
            target, internal_error, exc,
        )
        return []


_stats_guard = threading.Lock()
_stats_locks: dict[tuple, threading.Lock] = {}


def _band_statistics(path: str, band: int, sample_limit: int = 2_000_000) -> tuple:
    """Min/max/mean/std plus 2nd and 98th percentiles, from a decimated read.

    Sampling rather than reading every pixel keeps this instant on large
    scans; the percentile estimates are stable well below the sample limit.

    Guarded against a thundering herd. Opening an image fires a screenful of
    tile requests at once, and a plain ``lru_cache`` lets every one of them
    miss simultaneously and redo the same decimated read -- which is exactly
    why the first view of a new image used to arrive in pieces. One thread
    computes; the rest wait and take its answer.
    """
    key = (path, band, sample_limit)
    cached = _stats_cache.get(key)
    if cached is not None:
        return cached

    with _stats_guard:
        lock = _stats_locks.setdefault(key, threading.Lock())

    with lock:
        cached = _stats_cache.get(key)
        if cached is not None:
            return cached
        result = _compute_band_statistics(path, band, sample_limit)
        _stats_cache[key] = result

    with _stats_guard:
        _stats_locks.pop(key, None)
    return result


_stats_cache: dict[tuple, tuple] = {}


def _compute_band_statistics(path: str, band: int, sample_limit: int) -> tuple:
    dataset = open_raster(path)
    total = dataset.width * dataset.height
    step = max(1, int(math.sqrt(total / sample_limit))) if total > sample_limit else 1
    out_shape = (max(1, dataset.height // step), max(1, dataset.width // step))

    data = dataset.read(band, out_shape=out_shape, resampling=Resampling.average).astype(np.float64)
    if dataset.nodata is not None:
        data = data[data != dataset.nodata]
    data = data[np.isfinite(data)]
    if data.size == 0:
        return (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)

    return (
        float(np.min(data)),
        float(np.max(data)),
        float(np.mean(data)),
        float(np.std(data)) or 1.0,
        float(np.percentile(data, 2)),
        float(np.percentile(data, 98)),
    )


def band_statistics(path: str | Path, band: int) -> dict:
    lo, hi, mean, std, p2, p98 = _band_statistics(str(Path(path).resolve()), band)
    return {"min": lo, "max": hi, "mean": mean, "std": std, "p2": p2, "p98": p98}


def _stretch(data: np.ndarray, mode: str, stats: tuple, gamma: float = 1.0) -> np.ndarray:
    """Map raw band values onto 0-255 for display."""
    lo, hi, mean, std, p2, p98 = stats
    data = data.astype(np.float32)

    if mode == "none":
        low, high = lo, hi
    elif mode == "linear2pct":
        low, high = p2, p98
    elif mode == "stddev":
        low, high = mean - 2.0 * std, mean + 2.0 * std
    elif mode == "root":
        low, high = lo, hi
        data = np.sqrt(np.clip(data - low, 0, None))
        high = math.sqrt(max(high - low, 1e-6))
        low = 0.0
    elif mode == "equalize":
        finite = data[np.isfinite(data)]
        if finite.size:
            # Histogram equalisation via the empirical CDF of this tile.
            sorted_vals = np.sort(finite)
            ranks = np.searchsorted(sorted_vals, data, side="left")
            return np.clip(ranks / max(sorted_vals.size - 1, 1) * 255.0, 0, 255).astype(np.uint8)
        low, high = lo, hi
    else:  # "linear"
        low, high = lo, hi

    span = high - low
    if abs(span) < 1e-9:
        span = 1.0
    scaled = (data - low) / span
    if gamma and abs(gamma - 1.0) > 1e-6:
        scaled = np.clip(scaled, 0.0, 1.0) ** (1.0 / gamma)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def read_window(
    path: str | Path,
    col_off: int,
    row_off: int,
    width: int,
    height: int,
    bands: Optional[Sequence[int]] = None,
    out_shape: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """Read a pixel window, letting GDAL pick the best overview level."""
    dataset = open_raster(path)
    band_list = list(bands) if bands else list(range(1, min(dataset.count, 3) + 1))
    window = Window(col_off, row_off, width, height)
    shape = (len(band_list), out_shape[0], out_shape[1]) if out_shape else None
    return dataset.read(
        band_list,
        window=window,
        out_shape=shape,
        boundless=True,
        fill_value=0,
        resampling=Resampling.average,
    )


def render_tile(
    path: str | Path,
    zoom: int,
    tile_x: int,
    tile_y: int,
    bands: Optional[Sequence[int]] = None,
    enhancement: str = "linear2pct",
    gamma: float = 1.0,
    colormap: Optional[str] = None,
) -> bytes:
    """Render one 256px display tile as PNG bytes.

    Tiles are addressed in a simple pyramid over the image's own pixel grid:
    at ``zoom`` z, the image is treated as ``width / 2**z`` pixels wide. Zoom 0
    is the full overview.
    """
    from PIL import Image

    resolved = str(Path(path).resolve())
    dataset = open_raster(resolved)
    scale = 2 ** max(0, zoom)

    # Which source pixels this tile covers.
    src_size = TILE_SIZE * (max(dataset.width, dataset.height) / (TILE_SIZE * scale))
    src_size = max(1.0, src_size)
    col_off = tile_x * src_size
    row_off = tile_y * src_size

    if col_off >= dataset.width or row_off >= dataset.height:
        return _blank_tile()

    band_list = list(bands) if bands else list(range(1, min(dataset.count, 3) + 1))
    band_list = [b for b in band_list if 1 <= b <= dataset.count] or [1]

    # Read only the part of the tile that lies on the image, then place it in
    # a full tile. A boundless read would be simpler, but rasterio serves it
    # through a temporary VRT that hides the file's overviews, so a zoomed-out
    # tile was averaged down from full resolution: about a minute for the
    # overview tile of a 260-megapixel scan instead of milliseconds.
    in_w = min(src_size, dataset.width - col_off)
    in_h = min(src_size, dataset.height - row_off)
    out_w = max(1, min(TILE_SIZE, int(round(TILE_SIZE * in_w / src_size))))
    out_h = max(1, min(TILE_SIZE, int(round(TILE_SIZE * in_h / src_size))))
    part = dataset.read(
        band_list,
        window=Window(col_off, row_off, in_w, in_h),
        out_shape=(len(band_list), out_h, out_w),
        resampling=Resampling.average,
    )
    data = np.zeros((len(band_list), TILE_SIZE, TILE_SIZE), dtype=part.dtype)
    data[:, :out_h, :out_w] = part

    stretched = np.stack(
        [
            _stretch(data[i], enhancement, _band_statistics(resolved, band), gamma)
            for i, band in enumerate(band_list)
        ]
    )

    if colormap and stretched.shape[0] == 1:
        rgb = _apply_colormap(stretched[0], colormap)
    elif stretched.shape[0] == 1:
        rgb = np.repeat(stretched, 3, axis=0).transpose(1, 2, 0)
    else:
        rgb = stretched[:3].transpose(1, 2, 0)
        if rgb.shape[2] == 2:
            rgb = np.dstack([rgb, rgb[:, :, :1]])

    alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    if dataset.nodata is not None:
        raw = data[0]
        alpha[raw == dataset.nodata] = 0

    # Pixels beyond the image edge should be transparent, not black.
    end_col = min(dataset.width, col_off + src_size)
    end_row = min(dataset.height, row_off + src_size)
    frac_x = (end_col - col_off) / src_size
    frac_y = (end_row - row_off) / src_size
    if frac_x < 1.0:
        alpha[:, int(TILE_SIZE * frac_x):] = 0
    if frac_y < 1.0:
        alpha[int(TILE_SIZE * frac_y):, :] = 0

    image = Image.fromarray(np.dstack([rgb, alpha]), mode="RGBA")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()


@lru_cache(maxsize=1)
def _blank_tile() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


_COLORMAPS = {
    # Sampled control points; interpolated to 256 entries on first use.
    "elevation": [(0, 40, 90), (30, 120, 140), (120, 180, 120), (220, 210, 140),
                  (170, 120, 80), (245, 245, 245)],
    "precipitation": [(245, 245, 245), (170, 120, 80), (220, 210, 140),
                      (120, 180, 120), (30, 120, 140), (0, 40, 90)],
    "greyscale": [(0, 0, 0), (255, 255, 255)],
    "teal": [(8, 22, 20), (27, 156, 133), (235, 250, 246)],
}


@lru_cache(maxsize=16)
def _colormap_lut(name: str) -> np.ndarray:
    control = _COLORMAPS.get(name, _COLORMAPS["greyscale"])
    positions = np.linspace(0, 255, len(control))
    lut = np.zeros((256, 3), dtype=np.uint8)
    for channel in range(3):
        lut[:, channel] = np.interp(
            np.arange(256), positions, [c[channel] for c in control]
        ).astype(np.uint8)
    return lut


def _apply_colormap(band: np.ndarray, name: str) -> np.ndarray:
    return _colormap_lut(name)[band]


def sample_elevation(
    dem_path: str | Path,
    xs: Iterable[float],
    ys: Iterable[float],
    band: int = 1,
    bilinear: bool = True,
) -> np.ndarray:
    """Sample a DEM at map coordinates.

    Bilinear by default: nearest-neighbour sampling of a 5 m DEM introduces up
    to half a pixel of horizontal bias into every extracted GCP elevation,
    which shows up directly as a vertical residual in the adjustment.
    """
    dataset = open_raster(dem_path)
    xs = np.asarray(list(xs), dtype=float)
    ys = np.asarray(list(ys), dtype=float)

    inverse = ~dataset.transform
    cols, rows = inverse * (xs, ys)
    cols = np.asarray(cols, dtype=float) - 0.5
    rows = np.asarray(rows, dtype=float) - 0.5

    out = np.full(xs.shape, np.nan, dtype=float)
    if xs.size == 0:
        return out

    pad = 2
    col_lo = int(np.floor(np.nanmin(cols))) - pad
    row_lo = int(np.floor(np.nanmin(rows))) - pad
    col_hi = int(np.ceil(np.nanmax(cols))) + pad
    row_hi = int(np.ceil(np.nanmax(rows))) + pad

    window = Window(col_lo, row_lo, max(1, col_hi - col_lo), max(1, row_hi - row_lo))
    block = dataset.read(band, window=window, boundless=True, fill_value=np.nan).astype(float)
    if dataset.nodata is not None:
        block[block == dataset.nodata] = np.nan

    local_c = cols - col_lo
    local_r = rows - row_lo

    if not bilinear:
        ci = np.round(local_c).astype(int)
        ri = np.round(local_r).astype(int)
        valid = (ci >= 0) & (ri >= 0) & (ci < block.shape[1]) & (ri < block.shape[0])
        out[valid] = block[ri[valid], ci[valid]]
        return out

    c0 = np.floor(local_c).astype(int)
    r0 = np.floor(local_r).astype(int)
    fc = local_c - c0
    fr = local_r - r0

    valid = (c0 >= 0) & (r0 >= 0) & (c0 + 1 < block.shape[1]) & (r0 + 1 < block.shape[0])
    if not np.any(valid):
        return out

    c0v, r0v, fcv, frv = c0[valid], r0[valid], fc[valid], fr[valid]
    v00 = block[r0v, c0v]
    v01 = block[r0v, c0v + 1]
    v10 = block[r0v + 1, c0v]
    v11 = block[r0v + 1, c0v + 1]

    top = v00 * (1 - fcv) + v01 * fcv
    bottom = v10 * (1 - fcv) + v11 * fcv
    out[valid] = top * (1 - frv) + bottom * frv
    return out


def pixel_to_map(path: str | Path, cols, rows):
    dataset = open_raster(path)
    xs, ys = dataset.transform * (np.asarray(cols, float) + 0.5, np.asarray(rows, float) + 0.5)
    return np.asarray(xs), np.asarray(ys)


def map_to_pixel(path: str | Path, xs, ys):
    dataset = open_raster(path)
    cols, rows = (~dataset.transform) * (np.asarray(xs, float), np.asarray(ys, float))
    return np.asarray(cols) - 0.5, np.asarray(rows) - 0.5
