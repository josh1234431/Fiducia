"""Project and residual reports in the classic text layout.

Years of photogrammetry work sit in text reports laid out a particular way:
marking rubrics, spreadsheet imports and parsing scripts all expect the same
columns in the same places. This module writes that layout so none of it has to
change when the processing moves to Fiducia.

This is a data layout, reproduced for interoperability from reports the
operator produced themselves. It carries no third-party name or branding, and
the header is the plain description the layout has always used.

Two halves, kept apart on purpose:

* **Formatters** take plain structures and produce text. They know nothing
  about Fiducia, which is what lets them be checked byte-for-byte against real
  reports.
* **Adapters** build those structures from a Fiducia project: the camera, the
  image list, the observations and the adjusted block.

Where the classic layout carries a quantity Fiducia defines for itself, the
definition is stated at the adapter, not left to be guessed.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .camera import CameraModel, FiducialFit
from .collinearity import rotation_matrix

__all__ = [
    "format_project_report",
    "format_residual_report",
    "project_document",
    "residual_document",
    "project_report",
    "residual_report",
    "NEWLINE",
    "ENCODING",
]

# The layout is a Windows text file: CRLF line endings, ANSI code page.
NEWLINE = "\r\n"
ENCODING = "cp1252"

# Fiducials are listed clockwise from the top left.
FIDUCIAL_ORDER = [
    ("top_left", "Top left"),
    ("top_middle", "Top middle"),
    ("top_right", "Top right"),
    ("right_middle", "Right middle"),
    ("bottom_right", "Bottom right"),
    ("bottom_middle", "Bottom middle"),
    ("bottom_left", "Bottom left"),
    ("left_middle", "Left middle"),
]

FIDUCIAL_TYPE = {"edge_and_corner": "EdgeCorner", "corner": "Corner", "edge": "Edge"}


def _join(lines: list[str]) -> str:
    return NEWLINE.join(lines) + NEWLINE


# =========================================================================
# Formatters
# =========================================================================


def _table(indent: str, columns: list[dict], rows: list[list]) -> list[str]:
    """A self-sizing table in the classic style.

    Each column is ``{"header", "kind", "decimals", "sigma"}``. A column is as
    wide as the wider of its header and its widest value; headers and values
    are right-aligned to that width, and a ``+/- sigma`` suffix follows the
    value when the column carries one. Cells are separated by one space.
    """
    cells: list[list[str]] = []
    widths: list[int] = []
    suffix_widths: list[int] = []

    for index, column in enumerate(columns):
        texts = []
        for row in rows:
            value = row[index]
            if value is None or value == "":
                texts.append(("", ""))
            elif column.get("kind") == "text":
                texts.append((str(value), ""))
            else:
                number, sigma = value if isinstance(value, tuple) else (value, None)
                decimals = column.get("decimals", 4)
                text = f"{number:.{decimals}f}"
                suffix = f" +/- {sigma:.{decimals}f}" if sigma is not None else ""
                texts.append((text, suffix))
        width = max([len(column["header"])] + [len(t) for t, _ in texts])
        suffix_width = max([0] + [len(s) for _, s in texts])
        widths.append(width)
        suffix_widths.append(suffix_width)
        cells.append(texts)

    header = indent + " ".join(
        column["header"].rjust(widths[i]) + " " * suffix_widths[i]
        for i, column in enumerate(columns)
    )
    lines = [header, indent + "-" * (len(header) - len(indent))]
    for r in range(len(rows)):
        parts = []
        for i in range(len(columns)):
            text, suffix = cells[i][r]
            parts.append(text.rjust(widths[i]) + suffix.ljust(suffix_widths[i])
                         if text else " " * (widths[i] + suffix_widths[i]))
        lines.append(indent + " ".join(parts))
    return lines


def _labelled(indent: str, pairs: list[tuple[str, str]]) -> list[str]:
    """``Label : value`` lines with the colons aligned within the group."""
    width = max(len(label) for label, _ in pairs)
    return [f"{indent}{label.ljust(width)} : {value}" for label, value in pairs]


def format_project_report(doc: dict) -> str:
    """The project report, from a plain structure (see :func:`project_document`)."""
    title = doc.get("title", "Project Report for Multi image orthorectification")
    camera = doc["camera"]

    lines = [
        title,
        "-" * len(title),
        "",
        "General project information",
        "",
        *_labelled("    ", [("Filename", doc.get("filename", "")),
                            ("Description", doc.get("description", ""))]),
        "",
        "Camera calibration",
        "",
        *_labelled("    ", [
            ("Focal length", f"{camera['focalMm']:.3f} mm"),
            ("Principal point", f"{camera['ppoXMm']:.3f} mm, {camera['ppoYMm']:.3f} mm"),
        ]),
        "",
        *_labelled("    ", [(f"Radial Lens Distortion R{i}", f"{value:.6e}")
                            for i, value in enumerate(camera["radial"])]),
        "",
        *_labelled("    ", [(f"Decentering Distortion P{i + 1}", f"{value:.6e}")
                            for i, value in enumerate(camera["decentering"])]),
        "",
    ]

    if camera.get("fiducials"):
        pairs = [("Fiducial Type", camera.get("fiducialType", "EdgeCorner"))]
        pairs += [(label, f"{x:9.3f} mm, {y:9.3f} mm") for label, x, y in camera["fiducials"]]
        lines += _labelled("    ", pairs) + [""]

    if camera.get("pixelSizeMm"):
        lines += _labelled("    ", [
            ("Pixel size", f"{camera['pixelSizeMm']:.6f} mm"),
            ("Sensor size", f"{camera['columns']} P x {camera['rows']} L"),
        ]) + [""]

    scale = camera.get("scale")
    radius = camera.get("earthRadiusM")
    lines += _labelled("    ", [
        ("Scale", f"1: {scale:.3f}" if scale else "Not Defined"),
        ("Earth Radius", f"{radius:.3f}" if radius else "Not Defined"),
    ])
    lines.append("")

    for image in doc.get("images", []):
        lines.append(
            f"Image {image['name']} GCPs : {image['gcpCount']}  TPs : {image['tpCount']}"
            f"  Check Pts : {image['checkCount']}"
        )
        lines.append("")
        lines += _labelled("    ", [("Date Added", image.get("dateAdded", "")),
                                   ("Date Updated", image.get("dateUpdated", ""))])
        lines.append("")

        # One alignment across these three groups, as in the layout.
        group = [
            ("Uncorrected File", image.get("path", "")),
            ("Channels", "".join(f"{c} " for c in image.get("channels", []))),
            ("Size", f"{image.get('width', 0)} P x {image.get('height', 0)} L"),
        ]
        ortho = image.get("ortho")
        ortho_pairs = [("Orthorectified File", ortho["path"] if ortho else "Not Defined")]
        if ortho:
            ortho_pairs += [
                ("Upper Left", f"{ortho['west']:.6f} {ortho['north']:.6f}"),
                ("Lower Right", f"{ortho['east']:.6f} {ortho['south']:.6f}"),
            ]
        ortho_pairs.append(("Status", "Ortho done" if ortho else "Ortho not done"))
        dem = image.get("dem") or {}
        dem_pairs = [
            ("DEM File", dem.get("path") or "Not Defined"),
            ("Channel", str(dem.get("channel", 1))),
            ("Background Elevation",
             f"{dem['background']:.3f}" if dem.get("background") is not None else "Not Defined"),
        ]
        width = max(len(label) for label, _ in group + ortho_pairs + dem_pairs)
        for block in (group, ortho_pairs, dem_pairs):
            lines += [f"    {label.ljust(width)} : {value}" for label, value in block]
            lines.append("")

        clip = image.get("clip")
        if clip:
            lines += _labelled("    ", [
                ("Clip Area Offset", f"({clip[0]}, {clip[1]})"),
                ("Clip Area Size", f"{clip[2]} P x {clip[3]} L"),
            ])
            lines.append("")

        if image.get("fiducials"):
            pairs = [("Fiducial Type", image.get("fiducialType", "EdgeCorner"))]
            pairs += [(label, f"{col:7.1f} P, {row:7.1f} L")
                      for label, col, row in image["fiducials"]]
            pairs.append(("Calibration edge", image.get("calibrationEdge", "Left")))
            lines += _labelled("    ", pairs)
            lines.append("")

        eo = image.get("exterior")
        if eo:
            names = ["Xo", "Yo", "Zo", "Omega", "Phi", "Kappa"]
            pairs = [(f"Exterior Orientation {n}", f"{v:.6f}") for n, v in zip(names, eo)]
        else:
            pairs = []
        pairs.append(("Status", image.get("modelStatus", "Model not computed")))
        lines += _labelled("    ", pairs)
        lines.append("")

        gcps = image.get("gcps", [])
        if gcps:
            lines += _table("    ", [
                {"header": "GCP ID", "kind": "text"},
                {"header": "Status", "kind": "text"},
                {"header": "Elev (m)", "decimals": 4},
                {"header": "Image X (P)", "decimals": 4},
                {"header": "Image Y (L)", "decimals": 4},
            ], [[g["id"], g["status"], (g["z"], g["zSigma"]),
                 (g["col"], g["pixelSigma"]), (g["row"], g["pixelSigma"])] for g in gcps])
            lines.append("")
            lines += _table("    ", [
                {"header": "GCP ID", "kind": "text"},
                {"header": "Georef", "kind": "text"},
                {"header": "Georef X", "decimals": 4},
                {"header": "Georef Y", "decimals": 4},
            ], [[g["id"], g["georef"], (g["x"], g["xySigma"]), (g["y"], g["xySigma"])]
                for g in gcps])
            lines.append("")

        tps = image.get("tiePoints", [])
        if tps:
            lines += _table("    ", [
                {"header": "TP ID", "kind": "text"},
                {"header": "Elev (m)", "decimals": 1},
                {"header": "Image X (P)", "decimals": 1},
                {"header": "Image Y (L)", "decimals": 1},
            ], [[t["id"], None, (t["col"], t["pixelSigma"]), (t["row"], t["pixelSigma"])]
                for t in tps])
            lines.append("")

    return _join(lines)


# -- residual report -------------------------------------------------------

_LISTING_HEADER = (
    "Point ID        Res XY        Res X       Res Y      Res Z  Type     Image ID       "
    "Ground X         Ground Y         Ground Z           Comp X           Comp Y     "
    "DS Res XY    DS Res X   DS Res Y "
)
_SUMMARY_WIDTH = 102
_SUMMARY_LABELS = [
    (" Active GCPs:", "gcp", True),
    ("  Check points", "check", True),
    ("  Active TPs:", "tie", True),
    ("Inactive GCPs:", "gcp", False),
    (" Inactive TPs:", "tie", False),
]


def _rms(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values) / len(values))


def _summary(rows: list[dict]) -> list[str]:
    lines = []
    for label, kind, active in _SUMMARY_LABELS:
        subset = [r for r in rows if r["kind"] == kind and r.get("active", True) == active]
        count = len({r["pointId"] for r in subset})
        line = f"{label}{count:>{18 - len(label)}d}"
        for axis in ("resX", "resY", "resZ"):
            value = f"{_rms([r[axis] for r in subset]):6.3f}" if subset else " " * 6
            line += f" {axis[-1]} RMS: {value}"
        lines.append(line.ljust(_SUMMARY_WIDTH))
    return lines


def format_residual_report(doc: dict) -> str:
    """The residual report, from a plain structure (see :func:`residual_document`)."""
    rows = list(doc["rows"])
    images = doc["images"]

    lines = [
        "Residual Error Report",
        "",
        f"Residual Units: {doc.get('units', 'Ground units')}",
        "",
        f"Residual Summary for {len(images)} Images",
        *_summary(rows),
        "",
        f"Listing: {doc.get('listing', 'All points All images')}",
        "",
    ]

    # Worst first. A stable sort, so rows with equal residuals keep the order
    # they were given in.
    rows.sort(key=lambda r: -round(r["resXY"], 3))

    name_width = max([17] + [len(r["imageName"]) for r in rows])
    header = _LISTING_HEADER
    if name_width > 17:
        header = header.replace("Image ID       ", "Image ID       " + " " * (name_width - 17), 1)
    lines.append(header)

    type_label = {"gcp": "GCP", "tie": "TP", "check": "Check"}
    for r in rows:
        lines.append(
            f"{r['pointId']:>9s}{r['resXY']:11.3f}{r['resX']:11.3f}{r['resY']:11.3f}"
            f"{r['resZ']:11.3f}{type_label.get(r['kind'], r['kind']):>6s}  "
            f"{r['imageName']:<{name_width}s}"
            f"{r['groundX']:17.3f}{r['groundY']:17.3f}{r['groundZ']:17.3f}"
            f"{r['compX']:17.3f}{r['compY']:17.3f}"
            f"{r['dsXY']:11.2f}{r['dsX']:11.2f}{r['dsY']:11.2f}{r['dsZ']:11.2f}"
        )

    if rows:
        worst = rows[:max(1, int(len(rows) * 0.05))]
        lines.append(
            "RMS (x, y, z) for worst 5% of points in list: "
            f"{_rms([r['resX'] for r in worst]):.2f}, {_rms([r['resY'] for r in worst]):.2f}, "
            f"{_rms([r['resZ'] for r in worst]):.2f}"
        )
    lines.append("")

    for name in images:
        lines.append(f"Residual Summary for {name}")
        lines += _summary([r for r in rows if r["imageName"] == name])

    return _join(lines)


# =========================================================================
# Adapters -- a Fiducia project into the structures above
# =========================================================================


def _date(stamp: Optional[str]) -> str:
    if not stamp:
        return ""
    try:
        return datetime.fromisoformat(str(stamp).replace("Z", "+00:00")).strftime("%m/%d/%Y")
    except ValueError:
        return str(stamp)


def _georef(identifier: Optional[str]) -> str:
    """The 16-character projection-and-datum code, e.g. ``TM          D518``."""
    if not identifier:
        return ""
    try:
        import warnings

        from .geodesy import describe_crs

        with warnings.catch_warnings():
            # PROJ warns that a PROJ string loses detail; only the projection
            # name and zone are read from it here.
            warnings.simplefilter("ignore", UserWarning)
            described = describe_crs(identifier)
    except Exception:
        return identifier[:16]

    label = (described.get("label") or "").upper()
    proj4 = described.get("proj4") or ""
    if "UTM" in label or "+proj=utm" in proj4:
        zone = next((p.split("=")[1] for p in proj4.split() if p.startswith("+zone=")), "")
        projection = f"UTM {zone}{' S' if '+south' in proj4 else ''}".strip()
    elif "+proj=tmerc" in proj4 or " LO" in f" {label}":
        projection = "TM"
    elif described.get("isGeographic"):
        projection = "LONG/LAT"
    else:
        projection = (described.get("label") or identifier)[:8]

    datum = (described.get("datum") or "").split(" ")[0]
    if not (datum.startswith("D") and datum[1:].isdigit()):
        datum = ""
    return f"{projection:<12s}{datum:>4s}"


def project_document(state: dict, project_file: str = "",
                     image_ids: Optional[list[str]] = None) -> dict:
    """Everything the project report shows, from a Fiducia project."""
    camera = CameraModel.from_dict(state.get("camera") or {})
    model = state.get("model") or {}
    gcps = {g["id"]: g for g in state.get("gcps", [])}
    orthos = {o.get("imageId"): o for o in state.get("orthos", [])}
    dem_path = (state.get("dem") or {}).get("referencePath")
    georef = _georef((state.get("projection") or {}).get("gcpSource")
                     or (state.get("projection") or {}).get("output"))

    # K0..K3 multiply r, r^3, r^5, r^7 -- the odd terms of an eight-term series.
    radial = [0.0] * 8
    radial[1], radial[3], radial[5], radial[7] = camera.k0, camera.k1, camera.k2, camera.k3

    fiducials = [(label, *camera.fiducials_mm[slot])
                 for slot, label in FIDUCIAL_ORDER if slot in (camera.fiducials_mm or {})]

    doc = {
        "filename": project_file or f"{state.get('name', 'Project')}.fidu",
        "description": state.get("description", ""),
        "camera": {
            "focalMm": camera.focal_mm,
            "ppoXMm": camera.ppo_x_mm,
            "ppoYMm": camera.ppo_y_mm,
            "radial": radial,
            "decentering": [camera.p1, camera.p2, 0.0, 0.0],
            "fiducialType": FIDUCIAL_TYPE.get(camera.fiducial_position, "EdgeCorner"),
            "fiducials": fiducials if camera.kind == "film" else [],
            "pixelSizeMm": camera.pixel_pitch_mm if camera.kind == "digital" else None,
            "columns": camera.columns,
            "rows": camera.rows,
            "scale": camera.image_scale or None,
            "earthRadiusM": (camera.earth_radius_m or 6371000.0)
            if ((state.get("model") or {}).get("corrections") or {}).get("curvature") else None,
        },
        "images": [],
    }

    image_sigma_mm = model.get("imageSigmaMm")
    control_sigma = model.get("controlSigmaM")
    observations = state.get("observations", [])

    for image in state.get("images", []):
        if image_ids is not None and image["id"] not in image_ids:
            continue
        fit = FiducialFit.from_dict(image["fiducialFit"]) if image.get("fiducialFit") else None
        pixel_mm = camera.pixel_scale_mm(fit) if (fit or camera.kind == "digital") else 0.0

        def pixel_sigma(default: float) -> float:
            if image_sigma_mm and pixel_mm:
                return image_sigma_mm / pixel_mm
            return default

        mine = [o for o in observations if o.get("imageId") == image["id"]]
        gcp_rows, tp_rows = [], []
        counts = {"gcp": 0, "check": 0, "tie": 0}
        for o in mine:
            gcp = gcps.get(o["pointId"])
            if gcp is not None and all(gcp.get(k) is not None for k in ("x", "y", "z")):
                kind = "check" if gcp.get("isCheckPoint") else "gcp"
                counts[kind] += 1
                sigma = gcp.get("sigma") or control_sigma or 0.01
                gcp_rows.append({
                    "id": str(o["pointId"]),
                    "status": "Check" if kind == "check" else "Active",
                    "z": float(gcp["z"]), "zSigma": float(sigma),
                    "col": float(o["col"]), "row": float(o["row"]),
                    "pixelSigma": pixel_sigma(0.1),
                    "georef": georef,
                    "x": float(gcp["x"]), "y": float(gcp["y"]), "xySigma": float(sigma),
                })
            elif o.get("kind") == "tie" or gcp is None:
                counts["tie"] += 1
                tp_rows.append({"id": str(o["pointId"]), "col": float(o["col"]),
                                "row": float(o["row"]), "pixelSigma": pixel_sigma(0.5)})
        gcp_rows.sort(key=lambda g: g["id"])

        ortho = orthos.get(image["id"])
        clip = image.get("clipRegion")
        eo = image.get("exterior")

        doc["images"].append({
            "name": image.get("name", image["id"]),
            "gcpCount": counts["gcp"], "tpCount": counts["tie"], "checkCount": counts["check"],
            "dateAdded": _date(image.get("addedAt")),
            "dateUpdated": _date(image.get("updatedAt") or state.get("modified")),
            "path": image.get("path", ""),
            "channels": list(range(1, int(image.get("bandCount") or 0) + 1)),
            "width": int(image.get("width") or 0),
            "height": int(image.get("height") or 0),
            "ortho": {
                "path": ortho["path"],
                "west": ortho["bounds"][0], "south": ortho["bounds"][1],
                "east": ortho["bounds"][2], "north": ortho["bounds"][3],
            } if ortho and ortho.get("bounds") else None,
            "dem": {"path": dem_path, "channel": 1},
            "clip": [int(round(clip[0])), int(round(clip[1])),
                     int(round(clip[2] - clip[0])), int(round(clip[3] - clip[1]))]
            if clip else None,
            "fiducialType": FIDUCIAL_TYPE.get(camera.fiducial_position, "EdgeCorner"),
            "fiducials": [(label, *image["fiducials"][slot]) for slot, label in FIDUCIAL_ORDER
                          if slot in (image.get("fiducials") or {})],
            "calibrationEdge": image.get("calibrationEdge", "Left"),
            "exterior": [eo[0], eo[1], eo[2], *(math.degrees(a) for a in eo[3:6])] if eo else None,
            "modelStatus": "Model up-to-date" if eo else "Model not computed",
            "gcps": gcp_rows,
            "tiePoints": tp_rows,
        })

    return doc


def _ray(camera: CameraModel, image: dict, col: float, row: float):
    """Perspective centre and ground-frame direction of the ray through a pixel."""
    fit = FiducialFit.from_dict(image["fiducialFit"]) if image.get("fiducialFit") else None
    film = camera.pixel_to_film([[col, row]], fit)[0]
    eo = np.asarray(image["exterior"], dtype=float)
    rot = rotation_matrix(eo[3], eo[4], eo[5])
    direction = rot.T @ np.array([film[0], film[1], -camera.focal_mm])
    return eo[:3], direction / np.linalg.norm(direction)


def _intersect(rays: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Least-squares point closest to every ray."""
    a = np.zeros((3, 3))
    b = np.zeros(3)
    for origin, direction in rays:
        projector = np.eye(3) - np.outer(direction, direction)
        a += projector
        b += projector @ origin
    return np.linalg.solve(a, b)


def residual_document(state: dict) -> dict:
    """Ground residuals for every measurement, in the classic columns.

    Definitions, since the layout itself does not state them:

    * **Control and check points** -- *Comp* is the point's coordinate as the
      adjustment left it: for control, the surveyed coordinate after its
      (weighted) correction; for a check point, which the adjustment does not
      constrain, the intersection of its rays. Res X and Res Y are Comp minus
      surveyed; Res Z is surveyed minus Comp. That mixed sign is not a slip:
      it is the classic report convention, established against a real report --
      Fiducia's corrections correlate 0.99 with the reference solution's in X
      and Y and are exactly opposite in sign in Z (scratch/test_reference_block.py). The
      row is the same on every image of the point, because it describes the
      point. Where no adjusted coordinate exists (a single-photo resection)
      the rays are intersected instead.
    * **Tie points** -- *Ground* is the adjusted coordinate. *Comp* is where
      this image's ray meets the horizontal plane at that elevation, so Res X
      and Res Y are this measurement's miss on the ground. Res Z is the
      vertical part of the shortest offset from the point to the ray.
    * **DS** columns are standardised residuals: each residual divided by the
      measurement's standard deviation carried to the ground at that point
      (image sigma x distance / focal length). Values above about 3 deserve a
      look.
    """
    camera = CameraModel.from_dict(state.get("camera") or {})
    model = state.get("model") or {}
    images = {i["id"]: i for i in state.get("images", []) if i.get("exterior")}
    gcps = {g["id"]: g for g in state.get("gcps", [])
            if all(g.get(k) is not None for k in ("x", "y", "z"))}
    adjusted = {k: np.asarray(v, dtype=float) for k, v in (model.get("objectPoints") or {}).items()}
    image_sigma_mm = float(model.get("imageSigmaMm") or 0.010)

    by_point: dict[str, list[tuple[dict, dict]]] = {}
    for o in state.get("observations", []):
        image = images.get(o.get("imageId"))
        if image is not None:
            by_point.setdefault(o["pointId"], []).append((o, image))

    rows: list[dict] = []
    for point_id, measured in by_point.items():
        rays = []
        for o, image in measured:
            try:
                rays.append(_ray(camera, image, float(o["col"]), float(o["row"])))
            except (ValueError, KeyError, np.linalg.LinAlgError):
                rays.append(None)
        usable = [r for r in rays if r is not None]
        if not usable:
            continue

        gcp = gcps.get(point_id)
        if gcp is not None:
            kind = "check" if gcp.get("isCheckPoint") else "gcp"
            ground = np.array([gcp["x"], gcp["y"], gcp["z"]], dtype=float)
            if point_id in adjusted:
                comp = adjusted[point_id]
            elif len(usable) >= 2:
                comp = _intersect(usable)
            else:
                origin, direction = usable[0]
                comp = origin + direction * (ground[2] - origin[2]) / direction[2]
        else:
            kind = "tie"
            if point_id in adjusted:
                ground = adjusted[point_id]
            elif len(usable) >= 2:
                ground = _intersect(usable)
            else:
                continue
            comp = None

        for (o, image), ray in zip(measured, rays):
            if ray is None:
                continue
            origin, direction = ray
            if comp is None:
                on_plane = origin + direction * (ground[2] - origin[2]) / direction[2]
                closest = origin + direction * float((ground - origin) @ direction)
                res = np.array([on_plane[0] - ground[0], on_plane[1] - ground[1],
                                closest[2] - ground[2]])
                comp_xy = on_plane[:2]
            else:
                res = comp - ground
                res[2] = -res[2]          # the classic sign for control Z
                comp_xy = comp[:2]

            # Rounded to what is printed, so a residual of -1e-12 reads 0.000
            # rather than -0.000.
            res = np.round(res, 6) + 0.0
            distance = float(np.linalg.norm(ground - origin))
            sigma_ground = image_sigma_mm * distance / max(camera.focal_mm, 1e-9)
            res_xy = float(math.hypot(res[0], res[1]))
            rows.append({
                "pointId": str(point_id),
                "kind": kind,
                "active": True,
                "imageName": image.get("name", image["id"]),
                "resXY": res_xy, "resX": float(res[0]), "resY": float(res[1]),
                "resZ": float(res[2]),
                "groundX": float(ground[0]), "groundY": float(ground[1]),
                "groundZ": float(ground[2]),
                "compX": float(comp_xy[0]), "compY": float(comp_xy[1]),
                "dsXY": res_xy / sigma_ground, "dsX": float(res[0]) / sigma_ground,
                "dsY": float(res[1]) / sigma_ground, "dsZ": float(res[2]) / sigma_ground,
            })

    names = [i.get("name", i["id"]) for i in state.get("images", []) if i["id"] in images]
    # Equal residuals list in image order, so the output is repeatable.
    order = {name: index for index, name in enumerate(names)}
    rows.sort(key=lambda r: order.get(r["imageName"], 0))
    return {"units": "Ground units", "listing": "All points All images",
            "rows": rows, "images": names}


def project_report(state: dict, project_file: str = "",
                   image_ids: Optional[list[str]] = None) -> str:
    return format_project_report(project_document(state, project_file, image_ids))


def residual_report(state: dict) -> str:
    return format_residual_report(residual_document(state))


def write(text: str, path: str | Path) -> str:
    """Write a report exactly as laid out: CRLF endings, ANSI code page."""
    target = Path(path)
    target.write_bytes(text.encode(ENCODING, errors="replace"))
    return str(target)
