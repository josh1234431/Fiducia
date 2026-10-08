"""Swap -- getting data in and out.

Until now the only way to put a control point into Fiducia was to click it.
That is fine for a teaching exercise and unusable in practice: a surveyor
hands you a text file of coordinates, a UAV hands you a flight log with the
exterior orientation of every frame already solved by its GNSS and IMU, and a
camera calibration is re-used across every project flown with that camera.

So this module reads and writes the four things that actually cross the
boundary of a photogrammetric project:

    control points      CSV or whitespace text, in whatever column order
    exterior orientation    a flight log, which can skip ground control entirely
    camera calibrations     a library, so a camera is entered once ever
    image footprints        GeoJSON, so the block can be checked in any GIS

The import side is deliberately forgiving. Survey files arrive with headers or
without, comma or tab or semicolon separated, degrees or radians, in any column
order, often with a byte-order mark and Windows line endings. Refusing to read
one because it has a BOM is not a principled stand, it is just rude.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

__all__ = [
    "ExchangeError",
    "sniff_table",
    "read_control_points",
    "write_control_points",
    "read_exterior_orientation",
    "write_exterior_orientation",
    "save_camera_library",
    "load_camera_library",
    "write_footprints_geojson",
    "write_points_geojson",
]


class ExchangeError(RuntimeError):
    """Raised with a message intended to be shown to the operator as-is."""


# -- header guessing -------------------------------------------------------
#
# The names people actually use, gathered from survey deliverables rather than
# from a specification. Matching is case-insensitive and punctuation-blind.

_ALIASES = {
    "id": ("id", "point", "pointid", "pointno", "pointnum", "pointnumber",
           "pointname", "name", "station", "stationid", "stationno", "no",
           "number", "label", "pt", "ptid", "ptno", "beacon", "mark", "target",
           "gcp", "gcpid", "code", "descriptor"),
    "x": ("x", "easting", "east", "e", "lon", "long", "longitude", "y_lo", "xcoord"),
    "y": ("y", "northing", "north", "n", "lat", "latitude", "x_lo", "ycoord"),
    "z": ("z", "elevation", "elev", "height", "h", "alt", "altitude", "ortho", "orthoheight"),
    "image": ("image", "photo", "frame", "img", "picture", "file", "filename", "photoid"),
    "col": ("col", "column", "sample", "px", "pixel", "x_image", "ximage", "i"),
    "row": ("row", "line", "py", "y_image", "yimage", "j"),
    "omega": ("omega", "om", "roll", "w"),
    "phi": ("phi", "ph", "pitch", "p"),
    "kappa": ("kappa", "ka", "kap", "yaw", "heading", "k"),
    "sigma": ("sigma", "sd", "stddev", "accuracy", "precision", "weight"),
    "type": ("type", "kind", "class", "use", "role"),
}

_NORMALISE = re.compile(r"[^a-z0-9]")


def _key(name: str) -> str:
    return _NORMALISE.sub("", str(name).strip().lower())


def _match_columns(header: Sequence[str]) -> dict[str, int]:
    """Map canonical field names onto column indices."""
    found: dict[str, int] = {}
    keys = [_key(h) for h in header]

    for field_name, aliases in _ALIASES.items():
        for index, key in enumerate(keys):
            if index in found.values():
                continue
            if key in aliases:
                found[field_name] = index
                break
    return found


def _looks_numeric(value: str) -> bool:
    try:
        float(value.replace(",", ".") if value.count(",") == 1 and "." not in value else value)
        return True
    except (TypeError, ValueError):
        return False


def _to_float(value: str) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().replace(" ", "")
    if not text or text.lower() in ("na", "n/a", "null", "none", "-", "nan"):
        return None
    # A European decimal comma, but only when it is clearly a decimal marker
    # and not a thousands separator.
    if text.count(",") == 1 and "." not in text:
        text = text.replace(",", ".")
    else:
        text = text.replace(" ", "")
    try:
        return float(text)
    except ValueError:
        return None


@dataclass
class TableSniff:
    delimiter: str
    has_header: bool
    header: list[str]
    rows: list[list[str]]
    columns: dict[str, int]
    encoding: str

    def to_dict(self) -> dict:
        return {
            "delimiter": self.delimiter,
            "hasHeader": self.has_header,
            "header": self.header,
            "columns": self.columns,
            "encoding": self.encoding,
            "rowCount": len(self.rows),
            "preview": self.rows[:8],
        }


def sniff_table(path: str, max_preview: int = 400) -> TableSniff:
    """Work out how a text table is shaped before reading it.

    Returns the delimiter, whether row one is a header, and the best guess at
    which column holds which field. The interface shows that guess and lets the
    operator correct it, which is far kinder than refusing a file whose columns
    are in an order nobody anticipated.
    """
    source = Path(path)
    if not source.exists():
        raise ExchangeError(f"No file at {path}")

    raw = source.read_bytes()
    encoding = "utf-8-sig" if raw[:3] == b"\xef\xbb\xbf" else "utf-8"
    try:
        text = raw.decode(encoding)
    except UnicodeDecodeError:
        encoding = "latin-1"
        text = raw.decode(encoding, errors="replace")

    lines = [line for line in text.splitlines() if line.strip()
             and not line.lstrip().startswith(("#", "//", ";;"))]
    if not lines:
        raise ExchangeError(f"{source.name} has no data in it")

    sample = "\n".join(lines[:40])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t| ")
        delimiter = dialect.delimiter
    except csv.Error:
        # Fall back to whichever candidate splits the first line into the most
        # pieces -- almost always right for a survey file.
        counts = {d: lines[0].count(d) for d in (",", ";", "\t", "|")}
        delimiter = max(counts, key=counts.get) if max(counts.values()) else " "

    def split(line: str) -> list[str]:
        if delimiter == " ":
            return [p for p in line.split() if p]
        return [p.strip().strip('"').strip("'") for p in line.split(delimiter)]

    parsed = [split(line) for line in lines]
    first = parsed[0]

    # A header is a first row where the numeric-looking fields are scarce but
    # the rows beneath it are numeric.
    numeric_first = sum(1 for v in first if _looks_numeric(v))
    numeric_next = (
        sum(1 for v in parsed[1] if _looks_numeric(v)) if len(parsed) > 1 else 0
    )
    has_header = numeric_first <= 1 and numeric_next >= 2

    header = first if has_header else [f"column {i + 1}" for i in range(len(first))]
    rows = parsed[1:] if has_header else parsed

    columns = _match_columns(header) if has_header else {}

    # With no header, fall back to the conventional survey order.
    if not columns and rows:
        width = len(rows[0])
        if width >= 4:
            columns = {"id": 0, "x": 1, "y": 2, "z": 3}
        elif width == 3:
            columns = {"x": 0, "y": 1, "z": 2}

    return TableSniff(delimiter, has_header, list(header), rows[:max_preview],
                      columns, encoding)


# -- control points --------------------------------------------------------


@dataclass
class ImportedPoint:
    id: str
    x: float
    y: float
    z: float
    is_check_point: bool = False
    sigma: Optional[float] = None
    image: Optional[str] = None
    col: Optional[float] = None
    row: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id, "x": self.x, "y": self.y, "z": self.z,
            "isCheckPoint": self.is_check_point,
            "sigma": self.sigma,
            "image": self.image, "col": self.col, "row": self.row,
        }


def read_control_points(
    path: str,
    mapping: Optional[dict[str, int]] = None,
    check_point_values: Sequence[str] = ("check", "chk", "c", "0", "false"),
) -> dict:
    """Read ground control from a text table.

    ``mapping`` overrides the sniffed column assignment. Rows that cannot be
    read are reported rather than silently dropped -- a survey file with three
    bad rows should tell you which three.
    """
    sniff = sniff_table(path)
    columns = {**sniff.columns, **(mapping or {})}

    for required in ("x", "y"):
        if required not in columns:
            raise ExchangeError(
                f"Could not work out which column holds {required.upper()}. "
                f"Columns found: {', '.join(sniff.header)}"
            )

    points: list[ImportedPoint] = []
    problems: list[dict] = []

    def cell(row: list[str], field_name: str) -> Optional[str]:
        index = columns.get(field_name)
        if index is None or index >= len(row):
            return None
        return row[index]

    for number, row in enumerate(sniff.rows, start=2 if sniff.has_header else 1):
        x = _to_float(cell(row, "x"))
        y = _to_float(cell(row, "y"))
        z = _to_float(cell(row, "z"))

        if x is None or y is None:
            problems.append({"line": number, "reason": "X or Y could not be read",
                             "row": row[:6]})
            continue

        identifier = cell(row, "id")
        if not identifier:
            identifier = f"G{len(points) + 1:04d}"

        kind = (cell(row, "type") or "").strip().lower()
        is_check = kind in check_point_values

        image = cell(row, "image")
        col = _to_float(cell(row, "col"))
        row_px = _to_float(cell(row, "row"))

        points.append(ImportedPoint(
            id=str(identifier).strip(),
            x=x, y=y, z=z if z is not None else 0.0,
            is_check_point=is_check,
            sigma=_to_float(cell(row, "sigma")),
            image=str(image).strip() if image else None,
            col=col, row=row_px,
        ))

    missing_z = sum(1 for p in points if p.z == 0.0 and "z" not in columns)

    notes = []
    if "z" not in columns:
        notes.append(
            "No elevation column was found, so every point came in at zero. "
            "Extract elevations from the DEM before solving, or re-import with "
            "the height column mapped."
        )
    if problems:
        notes.append(f"{len(problems)} row(s) could not be read and were skipped.")

    return {
        "points": [p.to_dict() for p in points],
        "problems": problems,
        "columns": columns,
        "header": sniff.header,
        "delimiter": sniff.delimiter,
        "hasHeader": sniff.has_header,
        "notes": notes,
        "missingZ": missing_z,
    }


def write_control_points(
    path: str,
    gcps: Sequence[dict],
    residuals: Optional[dict] = None,
    delimiter: str = ",",
) -> dict:
    """Write control points out, with residuals when the block has been solved."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    has_residuals = bool(residuals)
    header = ["id", "type", "x", "y", "z"]
    if has_residuals:
        header += ["dx", "dy", "dz", "residual"]

    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=delimiter)
        writer.writerow(header)
        for gcp in gcps:
            row = [
                gcp.get("id", ""),
                "check" if gcp.get("isCheckPoint") else "control",
                gcp.get("x", ""), gcp.get("y", ""), gcp.get("z", ""),
            ]
            if has_residuals:
                entry = (residuals or {}).get(gcp.get("id"))
                if entry:
                    dx, dy, dz = entry
                    row += [round(dx, 4), round(dy, 4), round(dz, 4),
                            round(math.sqrt(dx * dx + dy * dy + dz * dz), 4)]
                else:
                    row += ["", "", "", ""]
            writer.writerow(row)

    return {"path": str(target), "count": len(gcps), "withResiduals": has_residuals}


# -- exterior orientation --------------------------------------------------


def read_exterior_orientation(
    path: str,
    mapping: Optional[dict[str, int]] = None,
    angle_units: str = "auto",
) -> dict:
    """Read a flight log: one solved orientation per photograph.

    A UAV or a GNSS/IMU-equipped aerial camera delivers this directly, which
    means the block can be orthorectified with no ground control at all --
    accuracy then depends on the navigation solution rather than on survey.

    ``angle_units`` of ``"auto"`` decides by magnitude: anything beyond about
    six and a half is being quoted in degrees, because no sane omega or phi is
    that many radians.
    """
    sniff = sniff_table(path)
    columns = {**sniff.columns, **(mapping or {})}

    for required in ("image", "x", "y", "z", "omega", "phi", "kappa"):
        if required not in columns:
            raise ExchangeError(
                f"Could not find the {required} column. A flight log needs image, "
                f"X, Y, Z, omega, phi and kappa. Columns found: "
                f"{', '.join(sniff.header)}"
            )

    def cell(row, name):
        index = columns.get(name)
        return row[index] if index is not None and index < len(row) else None

    raw: list[dict] = []
    problems: list[dict] = []

    for number, row in enumerate(sniff.rows, start=2 if sniff.has_header else 1):
        values = {name: _to_float(cell(row, name))
                  for name in ("x", "y", "z", "omega", "phi", "kappa")}
        name = cell(row, "image")
        if not name or any(v is None for v in values.values()):
            problems.append({"line": number, "reason": "incomplete row", "row": row[:8]})
            continue
        raw.append({"image": str(name).strip(), **values})

    if not raw:
        raise ExchangeError("No complete rows were found in the flight log")

    if angle_units == "auto":
        widest = max(
            max(abs(entry["omega"]), abs(entry["phi"]), abs(entry["kappa"]))
            for entry in raw
        )
        units = "degrees" if widest > 6.5 else "radians"
    else:
        units = angle_units

    factor = math.pi / 180.0 if units == "degrees" else 1.0

    records = [{
        "image": entry["image"],
        "exterior": [
            entry["x"], entry["y"], entry["z"],
            entry["omega"] * factor, entry["phi"] * factor, entry["kappa"] * factor,
        ],
    } for entry in raw]

    return {
        "records": records,
        "angleUnits": units,
        "problems": problems,
        "columns": columns,
        "header": sniff.header,
        "notes": [
            f"Angles read as {units}." + (
                " Values were large enough that degrees is the only sensible reading."
                if units == "degrees" and angle_units == "auto" else ""
            )
        ],
    }


def write_exterior_orientation(
    path: str, images: Sequence[dict], angle_units: str = "degrees"
) -> dict:
    """Export the solved orientation of every photograph."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    factor = 180.0 / math.pi if angle_units == "degrees" else 1.0

    written = 0
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "X", "Y", "Z",
                         f"omega_{angle_units}", f"phi_{angle_units}",
                         f"kappa_{angle_units}", "source"])
        for image in images:
            eo = image.get("exterior")
            if not eo:
                continue
            writer.writerow([
                image.get("name", image.get("id")),
                round(eo[0], 4), round(eo[1], 4), round(eo[2], 4),
                round(eo[3] * factor, 7), round(eo[4] * factor, 7),
                round(eo[5] * factor, 7),
                image.get("exteriorSource", ""),
            ])
            written += 1

    return {"path": str(target), "count": written, "angleUnits": angle_units}


# -- camera library --------------------------------------------------------


def save_camera_library(path: str, camera: dict, label: Optional[str] = None) -> dict:
    """Add a camera calibration to a reusable library file.

    A camera is calibrated once and flown for years. Re-typing forty numbers
    per project is not a workflow, it is an opportunity to make a mistake.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    library = {"version": 1, "cameras": []}
    if target.exists():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(existing.get("cameras"), list):
                library = existing
        except (json.JSONDecodeError, OSError):
            pass

    name = label or camera.get("name") or "Unnamed camera"
    entry = {"label": name, "camera": camera}

    library["cameras"] = [c for c in library["cameras"] if c.get("label") != name]
    library["cameras"].append(entry)
    library["cameras"].sort(key=lambda c: c.get("label", ""))

    target.write_text(json.dumps(library, indent=2), encoding="utf-8")
    return {"path": str(target), "label": name, "count": len(library["cameras"])}


def load_camera_library(path: str) -> dict:
    """Read a camera library."""
    target = Path(path)
    if not target.exists():
        raise ExchangeError(f"No camera library at {path}")
    try:
        library = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExchangeError(f"{target.name} is not a readable camera library: {exc}") from exc

    cameras = library.get("cameras")
    if not isinstance(cameras, list):
        raise ExchangeError(f"{target.name} contains no cameras")

    return {"path": str(target), "cameras": cameras}


# -- GeoJSON ---------------------------------------------------------------


def _crs_block(identifier: Optional[str]) -> dict:
    """GeoJSON is WGS84 by specification, so a projected file says so plainly."""
    if not identifier:
        return {}
    return {"projectCrs": identifier}


def write_footprints_geojson(
    path: str,
    images: Sequence[dict],
    camera: dict,
    dem_path: Optional[str] = None,
    crs: Optional[str] = None,
) -> dict:
    """Export every photograph's ground footprint as a polygon layer.

    This is the flight index: opened in any GIS it shows coverage, overlap and
    any frame that has ended up somewhere impossible, which is the fastest
    visual check there is on a block.
    """
    from .camera import CameraModel, FiducialFit
    from .ortho import OrthoSpec, compute_footprint

    model = CameraModel.from_dict(camera or {})
    features = []
    skipped = []

    for image in images:
        if not image.get("exterior") or not image.get("online"):
            skipped.append({"name": image.get("name"), "reason": "no solved orientation"})
            continue
        try:
            spec = OrthoSpec(
                image_path=image["path"],
                output_path="",
                camera=camera,
                fiducial_fit=image.get("fiducialFit"),
                exterior=image["exterior"],
                output_crs=crs or "",
                pixel_size_x=1.0, pixel_size_y=1.0,
                dem_path=dem_path,
                clip_region=image.get("clipRegion"),
            )
            west, south, east, north = compute_footprint(spec)
        except Exception as exc:
            skipped.append({"name": image.get("name"), "reason": str(exc)[:120]})
            continue

        eo = image["exterior"]
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [west, north], [east, north], [east, south], [west, south],
                    [west, north],
                ]],
            },
            "properties": {
                "name": image.get("name"),
                "id": image.get("id"),
                "centreX": round(eo[0], 3),
                "centreY": round(eo[1], 3),
                "flyingHeight": round(eo[2], 2),
                "kappaDegrees": round(math.degrees(eo[5]), 3),
                "source": image.get("exteriorSource"),
            },
        })

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        "type": "FeatureCollection",
        **_crs_block(crs),
        "features": features,
    }, indent=2), encoding="utf-8")

    return {"path": str(target), "count": len(features), "skipped": skipped}


def write_points_geojson(
    path: str,
    gcps: Sequence[dict],
    residuals: Optional[dict] = None,
    crs: Optional[str] = None,
) -> dict:
    """Export control and check points, carrying their residuals."""
    features = []
    for gcp in gcps:
        if gcp.get("x") is None or gcp.get("y") is None:
            continue
        properties = {
            "id": gcp.get("id"),
            "type": "check" if gcp.get("isCheckPoint") else "control",
            "z": gcp.get("z"),
            "source": gcp.get("source", "manual"),
        }
        entry = (residuals or {}).get(gcp.get("id"))
        if entry:
            dx, dy, dz = entry
            properties.update({
                "dx": round(dx, 4), "dy": round(dy, 4), "dz": round(dz, 4),
                "residual": round(math.sqrt(dx * dx + dy * dy + dz * dz), 4),
            })
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [gcp["x"], gcp["y"]]},
            "properties": properties,
        })

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        "type": "FeatureCollection",
        **_crs_block(crs),
        "features": features,
    }, indent=2), encoding="utf-8")

    return {"path": str(target), "count": len(features)}
