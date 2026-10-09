"""Point clouds of any size in bounded memory.

A drone LiDAR survey runs to hundreds of millions of points, far more than a
laptop can hold at once. So nothing here reads a whole cloud:

* Work that only builds up a grid (a terrain model, a minimum surface, an
  overview) streams the file in chunks.
* Work that needs each point's neighbours (noise, heights, learned classes)
  goes tile by tile, each tile read with a margin of its neighbours' points so
  that a point at a tile's edge sees everything around it.
* A result is a new class for some points. It is kept in a disk-backed array
  indexed by the point's place in the file, and written by streaming the
  source once more with the classes swapped in, so the output keeps every
  point, field and its order.

The tiles are made once per cloud, in one pass, and kept in a cache. Each
point takes 21 bytes there: its place in the file, its position relative to
its tile, its class, returns and intensity. Positions are the file's own
integer coordinates, so every point comes back exactly as the file holds it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np

from . import memory_budget

CHUNK = 2_000_000          # points per streamed read
VERSION = 2
NOISE_CLASSES = (7, 18)
OVERVIEW_SIZE = 1024
KEEP_CACHES = 8            # clouds whose tiles are kept

RECORD = np.dtype([
    ("index", "<u4"), ("X", "<i4"), ("Y", "<i4"), ("Z", "<i4"),
    ("classification", "u1"), ("return_number", "u1"), ("number_of_returns", "u1"),
    ("intensity", "<u2"),
])

# Peak bytes per point loaded, by the heaviest tool, measured: a tile plus its
# margin must fit in memory with room to spare.
BYTES_PER_POINT = 900

_cache_root: Optional[Path] = None
_building = threading.Lock()     # guards the table of per-cloud locks
_locks: dict = {}
TILE_POINTS: Optional[int] = None   # fixed points per tile, for tests; None: from free memory


def set_cache_root(path: Optional[str]) -> None:
    """Keep tiles inside the open project, or in the temporary folder."""
    global _cache_root
    _cache_root = Path(path) if path else None


def cache_root() -> Path:
    root = _cache_root or Path(tempfile.gettempdir()) / "fiducia-lidar"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _key(path: str) -> tuple[str, Path]:
    source = str(Path(path).resolve())
    stat = Path(source).stat()
    key = hashlib.sha1(f"{source.lower()}|{stat.st_size}|{stat.st_mtime_ns}|{VERSION}".encode()).hexdigest()[:20]
    return source, cache_root() / key


def is_indexed(path: str) -> bool:
    """Whether a cloud's tiles are made, so looking at it is instant."""
    return (_key(path)[1] / "manifest.json").exists()


def tile_target() -> int:
    """Points per tile: as many as a quarter of the free memory allows, within reason."""
    if TILE_POINTS:
        return TILE_POINTS
    try:
        free = memory_budget.available_bytes()
    except Exception:
        free = 4 * 1024 ** 3
    return int(np.clip(free * 0.25 / (BYTES_PER_POINT * 1.5), 200_000, 1_000_000))


def ensure_memory(needed: int, what: str) -> None:
    """Refuse clearly rather than let Windows page the machine to a standstill."""
    try:
        free = memory_budget.available_bytes()
    except Exception:
        return
    if free and needed > free * 0.8:
        raise MemoryError(
            f"{what} needs about {needed / 1024 ** 3:.1f} GB of memory and "
            f"{free / 1024 ** 3:.1f} GB is free. Close other programs, or use a larger cell size.")


# -- streaming ------------------------------------------------------------------

def stream(path: str, chunk: int = CHUNK) -> Iterator[tuple[int, object]]:
    """(index of the first point, points) for each chunk of the file, in order."""
    import laspy

    with laspy.open(path) as reader:
        start = 0
        for points in reader.chunk_iterator(chunk):
            yield start, points
            start += len(points)


def arrays(points) -> dict:
    """The fields every tool uses, as plain arrays."""
    out = {
        "x": np.asarray(points.x, dtype=np.float64),
        "y": np.asarray(points.y, dtype=np.float64),
        "z": np.asarray(points.z, dtype=np.float64),
        "classification": np.asarray(points.classification).astype(np.int64),
        "return_number": np.asarray(points.return_number).astype(np.int64),
        "number_of_returns": np.asarray(points.number_of_returns).astype(np.int64),
    }
    try:
        out["intensity"] = np.asarray(points.intensity).astype(np.float64)
    except Exception:
        out["intensity"] = np.zeros(out["x"].size)
    return out


def header_of(path: str):
    import laspy

    with laspy.open(path) as reader:
        return reader.header


def scratch(n: int, dtype, fill=None) -> np.memmap:
    """A disk-backed array for per-point results, deleted with its folder."""
    folder = cache_root() / "work"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{uuid.uuid4().hex}.dat"
    array = np.memmap(path, dtype=dtype, mode="w+", shape=(max(n, 1),))
    if fill is not None:
        array[:] = fill
    return array


def release(array: np.memmap) -> None:
    filename = getattr(array, "filename", None)
    del array
    if filename:
        try:
            os.remove(filename)
        except OSError:
            pass


def write_classified(path: str, output: str, classification: np.ndarray,
                     progress: Optional[Callable[[float], None]] = None) -> None:
    """Copy a cloud, chunk by chunk, with its classes replaced."""
    import laspy

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with laspy.open(path) as reader:
        total = max(int(reader.header.point_count), 1)
        with laspy.open(output, mode="w", header=reader.header) as writer:
            start = 0
            for points in reader.chunk_iterator(CHUNK):
                end = start + len(points)
                points.classification = np.asarray(classification[start:end]).astype(
                    np.asarray(points.classification).dtype)
                writer.write_points(points)
                start = end
                if progress:
                    progress(start / total)


# -- tiles ------------------------------------------------------------------------

class TiledCloud:
    """A cloud's points sorted into square tiles on disk, made on first use."""

    def __init__(self, path: str, progress: Optional[Callable[[float, str], None]] = None,
                 should_cancel: Optional[Callable[[], bool]] = None):
        self.source, self.folder = _key(path)
        manifest = self.folder / "manifest.json"
        # One build per cloud at a time: a second caller waits for the first.
        with _building:
            lock = _locks.setdefault(str(self.folder), threading.Lock())
        with lock:
            if not manifest.exists():
                self._build(progress, should_cancel)
            else:
                os.utime(self.folder)   # recently used, so pruned last
        self.manifest = json.loads(manifest.read_text(encoding="utf-8"))
        m = self.manifest
        self.n = m["points"]
        self.bounds = tuple(m["bounds"])
        self.tile_size = m["tileSize"]
        self.origin = tuple(m["origin"])
        self.z_base = m["zBase"]
        self.spacing = m["spacing"]
        self.scales, self.offsets = m["scales"], m["offsets"]
        self.tiles = {tuple(int(v) for v in key.split(",")): count for key, count in m["tiles"].items()}
        self.largest_tile = max(self.tiles.values()) if self.tiles else 0

    # -- making the tiles ------------------------------------------------------

    def _build(self, progress, should_cancel) -> None:
        header = header_of(self.source)
        n = int(header.point_count)
        west, south = float(header.x_min), float(header.y_min)
        east, north = float(header.x_max), float(header.y_max)
        area = max((east - west) * (north - south), 1e-6)
        density = n / area
        size = math.sqrt(tile_target() / max(density, 1e-9))
        size = float(np.clip(math.ceil(size / 5.0) * 5.0, 10.0, 5000.0))
        nx = max(1, int(math.ceil((east - west) / size)) + (1 if (east - west) % size == 0 else 0))
        ny = max(1, int(math.ceil((north - south) / size)) + (1 if (north - south) % size == 0 else 0))
        z_base = float(header.z_min)

        spacing = math.sqrt(area / max(n, 1))
        cell = max(max(east - west, north - south) / OVERVIEW_SIZE, 2.0 * spacing, 1e-3)
        ow = max(1, int(math.ceil((east - west) / cell)) + 1)
        oh = max(1, int(math.ceil((north - south) / cell)) + 1)
        top_z = np.full(ow * oh, -np.inf, dtype=np.float64)
        top_class = np.zeros(ow * oh, dtype=np.uint8)

        building = self.folder.with_name(self.folder.name + ".building")
        shutil.rmtree(building, ignore_errors=True)
        building.mkdir(parents=True)
        counts: dict = {}
        classes: dict = {}
        returns: dict = {}
        z_low, z_high = np.inf, -np.inf
        intensity_sample = []

        for start, points in stream(self.source):
            if should_cancel and should_cancel():
                shutil.rmtree(building, ignore_errors=True)
                raise InterruptedError("Cancelled")
            a = arrays(points)
            m = a["x"].size
            ix = np.clip(((a["x"] - west) / size).astype(np.int64), 0, nx - 1)
            iy = np.clip(((a["y"] - south) / size).astype(np.int64), 0, ny - 1)
            tile = iy * nx + ix
            record = np.empty(m, dtype=RECORD)
            record["index"] = np.arange(start, start + m, dtype=np.uint32)
            record["X"] = np.asarray(points.X)
            record["Y"] = np.asarray(points.Y)
            record["Z"] = np.asarray(points.Z)
            record["classification"] = a["classification"]
            record["return_number"] = a["return_number"]
            record["number_of_returns"] = a["number_of_returns"]
            record["intensity"] = np.clip(a["intensity"], 0, 65535)
            order = np.argsort(tile, kind="stable")
            ids, first = np.unique(tile[order], return_index=True)
            edges = list(first[1:]) + [m]
            for t, lo, hi in zip(ids, first, edges):
                tx, ty = int(t % nx), int(t // nx)
                with open(building / f"{tx}_{ty}.bin", "ab") as handle:
                    record[order[lo:hi]].tofile(handle)
                counts[(tx, ty)] = counts.get((tx, ty), 0) + int(hi - lo)

            # The overview: the highest real point in each pixel.
            real = ~np.isin(a["classification"], NOISE_CLASSES)
            cols = np.clip(((a["x"][real] - west) / cell).astype(np.int64), 0, ow - 1)
            rows = np.clip(((north - a["y"][real]) / cell).astype(np.int64), 0, oh - 1)
            flat = rows * ow + cols
            zr = a["z"][real]
            o = np.lexsort((-zr, flat))
            keep = np.ones(o.size, dtype=bool)
            keep[1:] = flat[o][1:] != flat[o][:-1]
            o = o[keep]
            higher = zr[o] > top_z[flat[o]]
            top_z[flat[o][higher]] = zr[o][higher]
            top_class[flat[o][higher]] = a["classification"][real][o][higher].clip(0, 255)

            for field, table in (("classification", classes), ("return_number", returns)):
                values, n_values = np.unique(a[field], return_counts=True)
                for v, c in zip(values, n_values):
                    table[str(int(v))] = table.get(str(int(v)), 0) + int(c)
            intensity_sample.append(a["intensity"][:: max(1, m // 20000)])
            if real.any():
                z_low, z_high = min(z_low, float(zr.min())), max(z_high, float(zr.max()))
            if progress:
                progress(min(0.99, (start + m) / max(n, 1)), f"Indexing {start + m:,} of {n:,} points")

        np.save(building / "overview_z.npy", top_z.reshape(oh, ow).astype(np.float32))
        np.save(building / "overview_class.npy", top_class.reshape(oh, ow))
        intensities = np.concatenate(intensity_sample) if intensity_sample else np.zeros(1)
        crs = None
        try:
            parsed = header.parse_crs()
            crs = parsed.to_string() if parsed is not None else None
        except Exception:
            pass
        manifest = {
            "version": VERSION,
            "source": self.source,
            "points": n,
            "bounds": [west, south, east, north],
            "origin": [west, south],
            "tileSize": size,
            "tiles": {f"{x},{y}": c for (x, y), c in counts.items()},
            "zBase": z_base,
            "scales": [float(v) for v in header.scales],
            "offsets": [float(v) for v in header.offsets],
            "spacing": spacing,
            "crs": crs,
            "lasVersion": f"{header.version.major}.{header.version.minor}",
            "pointFormat": int(header.point_format.id),
            "classHistogram": classes,
            "returnHistogram": returns,
            # One scale for intensity across every tile, so a feature means the same everywhere.
            "intensity": {"p99": float(np.percentile(intensities, 99)),
                          "varies": bool(intensities.max() > intensities.min())},
            "elevationRange": [z_low if np.isfinite(z_low) else float(header.z_min),
                               z_high if np.isfinite(z_high) else float(header.z_max)],
            "overview": {"cell": cell, "bounds": [west, north - oh * cell, west + ow * cell, north],
                         "width": ow, "height": oh},
        }
        (building / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        shutil.rmtree(self.folder, ignore_errors=True)
        os.replace(building, self.folder)
        _prune(keep=self.folder)

    # -- reading tiles ---------------------------------------------------------

    def tile_bounds(self, tile: tuple) -> tuple:
        x0 = self.origin[0] + tile[0] * self.tile_size
        y0 = self.origin[1] + tile[1] * self.tile_size
        return (x0, y0, x0 + self.tile_size, y0 + self.tile_size)

    def _read(self, tile: tuple) -> dict:
        path = self.folder / f"{tile[0]}_{tile[1]}.bin"
        if not path.exists():
            return None
        record = np.fromfile(path, dtype=RECORD)
        (sx, sy, sz), (ox, oy, oz) = self.scales, self.offsets
        return {
            "index": record["index"].astype(np.int64),
            "x": record["X"] * sx + ox,
            "y": record["Y"] * sy + oy,
            "z": record["Z"] * sz + oz,
            "classification": record["classification"].astype(np.int64),
            "return_number": record["return_number"].astype(np.int64),
            "number_of_returns": record["number_of_returns"].astype(np.int64),
            "intensity": record["intensity"].astype(np.float64),
        }

    def load(self, tile: tuple, margin: float = 0.0) -> dict:
        """A tile's points, plus its neighbours' within the margin; `core` marks the tile's own."""
        parts, cores = [], []
        reach = max(0, int(math.ceil(margin / self.tile_size)))
        west, south, east, north = self.tile_bounds(tile)
        for dx in range(-reach, reach + 1):
            for dy in range(-reach, reach + 1):
                other = (tile[0] + dx, tile[1] + dy)
                if other not in self.tiles:
                    continue
                part = self._read(other)
                if part is None:
                    continue
                if other != tile:
                    near = ((part["x"] >= west - margin) & (part["x"] < east + margin)
                            & (part["y"] >= south - margin) & (part["y"] < north + margin))
                    part = {k: v[near] for k, v in part.items()}
                parts.append(part)
                cores.append(np.full(part["x"].size, other == tile))
        if not parts:
            return {"index": np.zeros(0, np.int64), "core": np.zeros(0, bool), "x": np.zeros(0),
                    "y": np.zeros(0), "z": np.zeros(0), "classification": np.zeros(0, np.int64),
                    "return_number": np.zeros(0, np.int64), "number_of_returns": np.zeros(0, np.int64),
                    "intensity": np.zeros(0)}
        out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        out["core"] = np.concatenate(cores)
        return out

    def load_box(self, west: float, south: float, east: float, north: float) -> dict:
        """Every point inside a box, from the tiles it touches."""
        x0, y0 = self.origin
        tx0, tx1 = int((west - x0) // self.tile_size), int((east - x0) // self.tile_size)
        ty0, ty1 = int((south - y0) // self.tile_size), int((north - y0) // self.tile_size)
        parts = []
        for tx in range(tx0, tx1 + 1):
            for ty in range(ty0, ty1 + 1):
                if (tx, ty) not in self.tiles:
                    continue
                part = self._read((tx, ty))
                inside = ((part["x"] >= west) & (part["x"] <= east)
                          & (part["y"] >= south) & (part["y"] <= north))
                parts.append({k: v[inside] for k, v in part.items()})
        if not parts:
            return self.load((10 ** 9, 10 ** 9))
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

    def overview_grids(self) -> tuple[np.ndarray, np.ndarray]:
        z = np.load(self.folder / "overview_z.npy").astype(np.float64)
        z[~np.isfinite(z)] = np.nan
        return z, np.load(self.folder / "overview_class.npy")


def _prune(keep: Path) -> None:
    """Keep the tiles of the most recently used clouds only."""
    root = cache_root()
    folders = [p for p in root.iterdir() if p.is_dir() and p.name not in ("work",) and not p.name.endswith(".building")]
    folders.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for old in folders[KEEP_CACHES:]:
        if old != keep:
            shutil.rmtree(old, ignore_errors=True)
