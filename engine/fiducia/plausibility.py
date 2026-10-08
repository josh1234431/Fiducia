"""Plausibility checks -- is this setup, and this solution, physically possible?

A least-squares solver always returns numbers. Whether those numbers describe
a camera that flew over the ground is a separate question, and one the solver
cannot answer. This module answers it, in three places:

* **Before solving** -- the camera description against the images it
  describes. A digital camera whose chip is described 20010 x 13080 cannot
  have produced a 13080 x 20010 file without being turned; principal-point
  offsets are fractions of a millimetre, so 68 mm in that field is a chip
  dimension typed into the wrong box. Either one moves the principal point by
  thousands of pixels, and the adjustment then fits the error rather than the
  photograph -- with nothing in its output to say so.
* **After solving** -- a perspective centre below the ground, or residuals far
  beyond measurement precision, is not a solution however well it converged.
* **While typing** -- control coordinates entered the wrong way round.

Every check returns the problem in words and, where the fix is unambiguous,
the fix itself, so the interface can offer it as one click.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

__all__ = [
    "camera_problems",
    "judge_solution",
    "crs_differences",
    "PIXELS_FAIL",
    "PIXELS_WARN",
]

# In pixels of the image, because that is what a measurement is made in.
# A careful measurement on a sharp feature is good to a pixel or better; ten
# pixels of residual means the model and the data disagree about geometry;
# past twenty-five no hand measurement explains it.
#
# These judge the solution, but they cannot on their own catch a wrong camera
# description: a misplaced principal point is partly absorbed by the
# orientation, and can leave residuals that look merely mediocre. That is why
# camera_problems() blocks the solve outright rather than leaving it to this.
PIXELS_WARN = 3.0
PIXELS_FAIL = 25.0

# Principal-point offsets in a calibration certificate are microns to a few
# tenths of a millimetre. A millimetre is already implausible.
OFFSET_LIMIT_MM = 1.0


def _close(a: float, b: float, tolerance: float = 0.002) -> bool:
    return b != 0 and abs(a - b) / abs(b) < tolerance


def _turn_offset(x: float, y: float, degrees: int) -> tuple[float, float]:
    """A principal point offset after the image is turned clockwise.

    Image coordinates run x to the right and y up. Turning the image 90
    degrees clockwise carries the top edge to the right: (x, y) -> (y, -x).
    """
    if degrees == 90:
        return y, -x
    if degrees == 270:
        return -y, x
    if degrees == 180:
        return -x, -y
    return x, y


def turned_camera_fixes(camera: dict, width: int, height: int) -> tuple[list[dict], str]:
    """Camera patches for images delivered turned a quarter, and a note.

    The size swaps either way. The principal point offset turns with the
    image, and 90 and 270 degrees cannot be told apart from the size alone,
    so when the answer depends on it both are offered. The certificate's own
    rotation table is used where one was read; otherwise the offset is
    turned by the geometry above.
    """
    size = {"columns": width, "rows": height}
    x = float(camera.get("ppoXMm") or 0.0)
    y = float(camera.get("ppoYMm") or 0.0)
    table = camera.get("ppoRotations") or {}

    options = {}
    for degrees in (90, 270):
        entry = table.get(str(degrees))
        if entry is not None and len(entry) == 2:
            options[degrees] = (float(entry[0]), float(entry[1]), "certificate")
        elif x or y:
            tx, ty = _turn_offset(x, y, degrees)
            options[degrees] = (tx, ty, "turned")

    if not options:
        # Nothing but the size depends on the direction of the turn.
        return [{"fix": size, "label": f"Describe the camera as {width} x {height}"}], ""

    values = {(round(v[0], 6), round(v[1], 6)) for v in options.values()}
    if len(values) == 1:
        ox, oy, source = next(iter(options.values()))
        fix = {**size, "ppoXMm": ox, "ppoYMm": oy}
        note = (f" The principal point becomes {ox:.3f}, {oy:.3f} mm"
                + (", from the certificate's rotation table." if source == "certificate"
                   else ", turned with the image."))
        return [{"fix": fix, "label": f"Describe the camera as {width} x {height}"}], note

    fixes = []
    for degrees, (ox, oy, source) in sorted(options.items()):
        fixes.append({
            "fix": {**size, "ppoXMm": ox, "ppoYMm": oy},
            "label": f"Turned {degrees}° clockwise (principal point {ox:.3f}, {oy:.3f})",
        })
    from_table = all(source == "certificate" for _, _, source in options.values())
    note = (" The principal point also turns with the image, and the direction of "
            "the turn decides where. Choose the rotation the images were delivered "
            "with" + (", as listed in the certificate's rotation table." if from_table
                      else "; these values are the offset turned by geometry, so check "
                           "them against the certificate."))
    return fixes, note


def camera_problems(camera: dict, images: list[dict]) -> list[dict]:
    """Inconsistencies between a camera description and its images.

    Each problem is ``{"code", "severity", "message", "fix"?}``; ``severity``
    is ``"blocker"`` or ``"warning"``, and ``fix`` is a camera patch that
    resolves it.
    """
    problems: list[dict] = []
    kind = camera.get("kind") or "film"
    pitch = float(camera.get("pixelPitchMm") or 0)
    columns = int(camera.get("columns") or 0)
    rows = int(camera.get("rows") or 0)

    if kind == "digital" and columns and rows:
        sizes = {(int(i.get("width") or 0), int(i.get("height") or 0))
                 for i in images if i.get("online") and i.get("width")}
        for width, height in sorted(sizes):
            if (width, height) == (columns, rows):
                continue
            names = [i.get("name", "?") for i in images
                     if (i.get("width"), i.get("height")) == (width, height)]
            label = ", ".join(names[:3]) + ("..." if len(names) > 3 else "")
            if (width, height) == (rows, columns):
                fixes, note = turned_camera_fixes(camera, width, height)
                problems.append({
                    "code": "chip_turned",
                    "severity": "blocker",
                    "message": (
                        f"{label} is {width} x {height} pixels, but the camera is "
                        f"described as {columns} x {rows}. The image is the sensor "
                        "turned 90 degrees, and the camera has to describe the file as "
                        "delivered, or every measurement lands thousands of pixels "
                        "from where the model expects it." + note
                    ),
                    # The first fix also as fix/fixLabel, for older interfaces.
                    "fix": fixes[0]["fix"],
                    "fixLabel": fixes[0]["label"],
                    "fixes": fixes,
                })
            else:
                problems.append({
                    "code": "chip_size",
                    "severity": "blocker",
                    "message": (
                        f"{label} is {width} x {height} pixels, but the camera is "
                        f"described as {columns} x {rows}. If the image was cropped or "
                        "resampled, the camera description no longer applies to it."
                    ),
                })

    chip_dimensions = []
    if pitch and columns:
        chip_dimensions.append((columns * pitch, f"the chip's width ({columns} x {pitch * 1000:g} um)"))
        chip_dimensions.append((columns * pitch / 2, "half the chip's width"))
    if pitch and rows:
        chip_dimensions.append((rows * pitch, f"the chip's height ({rows} x {pitch * 1000:g} um)"))
        chip_dimensions.append((rows * pitch / 2, "half the chip's height"))

    offset_fields = [
        ("longTrackOffsetMm", "Long-track offset"),
        ("crossTrackOffsetMm", "Cross-track offset"),
        ("ppoXMm", "Principal point x"),
        ("ppoYMm", "Principal point y"),
    ]
    bad_fields = {}
    for key, label in offset_fields:
        value = float(camera.get(key) or 0)
        if abs(value) <= OFFSET_LIMIT_MM:
            continue
        match = next((what for size, what in chip_dimensions if _close(abs(value), size)), None)
        reason = (f"that is exactly {match}, not an offset" if match
                  else "principal-point offsets are fractions of a millimetre")
        problems.append({
            "code": f"offset_{key}",
            "severity": "blocker",
            "message": (
                f"{label} is {value:g} mm; {reason}. It moves the principal point "
                f"{abs(value) / pitch:,.0f} pixels off the centre of the image."
                if pitch else
                f"{label} is {value:g} mm; {reason}."
            ),
        })
        bad_fields[key] = 0.0
    if bad_fields:
        # One fix for all of them, attached to the first.
        first = next(p for p in problems if p["code"].startswith("offset_"))
        first["fix"] = bad_fields
        first["fixLabel"] = "Set " + ("these offsets" if len(bad_fields) > 1 else "it") + " to zero"

    return problems


def judge_solution(
    eo: np.ndarray,
    ground: np.ndarray,
    rms_mm: float,
    pixel_mm: float,
    focal_mm: float,
    label: str = "the photo",
) -> dict:
    """Whether a solved orientation is physically possible, and how good it is.

    Returns ``{"ok", "quality", "rmsPx", "rmsGroundM", "problems"}`` where
    quality is ``good``, ``poor`` or ``failed``.
    """
    problems: list[str] = []
    eo = np.asarray(eo, dtype=float)
    ground = np.atleast_2d(np.asarray(ground, dtype=float))
    pixel_mm = pixel_mm or 0.0

    rms_px = rms_mm / pixel_mm if pixel_mm else float("nan")
    height = float(eo[2] - np.max(ground[:, 2])) if ground.size else float("nan")
    ground_per_mm = (eo[2] - float(np.mean(ground[:, 2]))) / focal_mm if focal_mm else float("nan")
    rms_ground = rms_mm * ground_per_mm if math.isfinite(ground_per_mm) else float("nan")

    failed = False
    if not math.isfinite(height) or height <= 0:
        failed = True
        problems.append(
            f"The solution puts the camera for {label} {abs(height):,.0f} m "
            f"{'below' if height <= 0 else 'above'} the highest control point"
            + (" -- underground, which is not a solution." if height <= 0 else ".")
        )
    elif math.isfinite(rms_px) and rms_px > PIXELS_FAIL:
        failed = True
        problems.append(
            f"Residuals on {label} average {rms_px:,.0f} pixels "
            f"({rms_ground:,.1f} m on the ground). Nothing can be measured that "
            "badly by hand; the camera description or the control coordinates do "
            "not match this image."
        )
    elif math.isfinite(rms_px) and rms_px > PIXELS_WARN:
        problems.append(
            f"Residuals on {label} average {rms_px:.1f} pixels ({rms_ground:.2f} m). "
            "A careful measurement is good to about a pixel, so the model and the "
            "control disagree by more than measurement error: check the control "
            "coordinates and their source, and the camera constants."
        )

    quality = "failed" if failed else ("poor" if problems else "good")
    return {
        "ok": not failed,
        "quality": quality,
        "rmsPx": rms_px if math.isfinite(rms_px) else None,
        "rmsGroundM": rms_ground if math.isfinite(rms_ground) else None,
        "flyingHeightM": height if math.isfinite(height) else None,
        "problems": problems,
    }


def _describe(identifier_or_wkt: str):
    from pyproj import CRS

    from .geodesy import resolve_crs

    try:
        return resolve_crs(identifier_or_wkt)
    except Exception:  # noqa: BLE001
        return CRS.from_user_input(identifier_or_wkt)


# Southern Africa, generously: where Lo-zone coordinates can plausibly land.
_SA_LAT = (-35.5, -21.5)
_SA_LON = (15.0, 33.5)


def _zone_geometry(identifier: str) -> Optional[tuple[float, float]]:
    """(central meridian, half-width in degrees) of a transverse Mercator zone."""
    try:
        params = _describe(identifier).to_dict()
    except Exception:  # noqa: BLE001
        return None
    if params.get("proj") == "utm":
        zone = int(params.get("zone", 0))
        return (zone * 6 - 183, 3.0) if zone else None
    if params.get("proj") == "tmerc" and "lon_0" in params:
        return float(params["lon_0"]), 1.0
    return None


def _placed(identifier: str, xs, ys):
    """The control points as (lon, lat) medians when read in this system."""
    import numpy as np

    from .geodesy import transform_points

    try:
        lon, lat = transform_points(identifier, "EPSG:4326", np.asarray(xs, float), np.asarray(ys, float))
    except Exception:  # noqa: BLE001
        return None
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    good = np.isfinite(lon) & np.isfinite(lat)
    if not good.any():
        return None
    return float(np.median(lon[good])), float(np.median(lat[good]))


def _fits(identifier: str, xs, ys) -> bool:
    geometry = _zone_geometry(identifier)
    placed = _placed(identifier, xs, ys)
    if geometry is None or placed is None:
        return False
    (meridian, half), (lon, lat) = geometry, placed
    in_africa = _SA_LAT[0] <= lat <= _SA_LAT[1] and _SA_LON[0] <= lon <= _SA_LON[1]
    return in_africa and abs(lon - meridian) <= half


def control_crs_problems(gcps: list[dict], identifier: str,
                         hint: Optional[str] = None) -> list[dict]:
    """Whether the control coordinates make sense in the chosen system.

    Read in the wrong Lo zone, control lands hundreds of kilometres from the
    zone's meridian; read in the wrong orientation (south-oriented Y/X against
    E/N), the signs put it in the wrong hemisphere or ocean. Both give a
    solution that fails for no visible reason, so both are caught here, with
    the system the points do fit where one can be found.
    """
    from .geodesy import PRESETS

    points = [g for g in gcps if isinstance(g.get("x"), (int, float))
              and isinstance(g.get("y"), (int, float)) and (g["x"] or g["y"])]

    # Metres read as degrees: nothing downstream can work, so it is a blocker.
    # The adjustment itself would still "solve" -- it only sees numbers --
    # and the failure would surface later as an empty DEM or a blank ortho.
    if points and identifier:
        from .geodesy import PRESETS, resolve_crs

        try:
            geographic = resolve_crs(identifier).is_geographic
        except ValueError:
            geographic = False
        if geographic and any(abs(g["x"]) > 180 or abs(g["y"]) > 90 for g in points):
            xs = [g["x"] for g in points]
            ys = [g["y"] for g in points]
            # Lo coordinates are relative to their own meridian, so they fit
            # every Lo zone; the output projection's zone is the likely one.
            hinted = _zone_geometry(hint)[0] if hint and _zone_geometry(hint) else None
            keys = sorted((key for key in PRESETS
                           if (key.startswith("ZALO") or key.startswith("UTM"))
                           and not key.endswith(("_WGS84", "_CAPE"))),
                          key=lambda key: (_zone_geometry(key) or (None,))[0] != hinted)
            fit = next((key for key in keys if _fits(key, xs, ys)), None)
            message = ("The control projection is in degrees of latitude and longitude, but "
                       "the control coordinates are in metres. Choose the projected system "
                       "they were measured in")
            message += f": they fit {PRESETS[fit].label}." if fit else "."
            return [{"code": "control_crs", "severity": "blocker", "message": message,
                     "suggestion": fit}]

    geometry = _zone_geometry(identifier) if identifier else None
    if not points or geometry is None:
        return []
    xs = [g["x"] for g in points]
    ys = [g["y"] for g in points]
    placed = _placed(identifier, xs, ys)
    if placed is None:
        return []

    meridian, half = geometry
    lon, lat = placed
    south_african = identifier.startswith("ZALO")
    label = PRESETS[identifier].label if identifier in PRESETS else identifier
    off_zone = abs(lon - meridian) > half + 0.5
    off_map = south_african and not (_SA_LAT[0] <= lat <= _SA_LAT[1]
                                     and _SA_LON[0] <= lon <= _SA_LON[1])
    if not (off_zone or off_map):
        return []

    if off_map and not off_zone:
        message = (f"Read in {label}, the control points fall at "
                   f"{abs(lat):.1f}°{'S' if lat < 0 else 'N'}, {abs(lon):.1f}°"
                   f"{'E' if lon >= 0 else 'W'}, outside South Africa. The signs of the "
                   "coordinates suggest the other orientation.")
    else:
        km = abs(lon - meridian) * 111.32 * max(0.1, abs(math.cos(math.radians(lat))))
        message = (f"The control points lie about {km:,.0f} km from the {meridian:g}° "
                   f"central meridian of {label}, outside the zone.")

    # Which system the points do fit. Lo coordinates are relative to their
    # own meridian, so they "fit" every Lo zone of the right orientation; the
    # likeliest intended system keeps this one's meridian and changes only the
    # orientation, so that is tried first.
    def meridian_of(key: str) -> Optional[float]:
        found = _zone_geometry(key)
        return found[0] if found else None

    candidates = sorted(
        (key for key in PRESETS if key != identifier
         and (key.startswith("ZALO") or key.startswith("UTM"))),
        key=lambda key: (meridian_of(key) != meridian, not key.startswith("ZALO"),
                         key.endswith(("_WGS84", "_CAPE"))),
    )
    fit = next((key for key in candidates if _fits(key, xs, ys)), None)
    if fit:
        message += f" They fit {PRESETS[fit].label}; check the GCP projection."
    else:
        message += " Check the GCP projection and the coordinates' order and signs."
    return [{"code": "control_crs", "severity": "warning", "message": message,
             "suggestion": fit}]


def crs_differences(project_crs: str, data_crs: str) -> list[str]:
    """Plain-language differences between two projected systems.

    Only the differences that move coordinates by more than a survey
    tolerance: the central meridian, which way the axes run, and the datum
    family. Naming differences are ignored.
    """
    if not project_crs or not data_crs:
        return []
    try:
        a, b = _describe(project_crs), _describe(data_crs)
    except Exception:  # noqa: BLE001
        return []

    out: list[str] = []

    def meridian(crs) -> Optional[float]:
        try:
            params = crs.to_dict()
            return float(params.get("lon_0")) if "lon_0" in params else None
        except Exception:  # noqa: BLE001
            return None

    ma, mb = meridian(a), meridian(b)
    if ma is not None and mb is not None and abs(ma - mb) > 1e-6:
        out.append(f"central meridian {ma:g} deg in the project, {mb:g} deg in the data")

    def directions(crs) -> tuple:
        return tuple(axis.direction for axis in crs.axis_info[:2])

    da, db = directions(a), directions(b)
    if set(da) != set(db) and a.is_projected and b.is_projected:
        out.append(f"axes run {'/'.join(da)} in the project, {'/'.join(db)} in the data")
    return out
