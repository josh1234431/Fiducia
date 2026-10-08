"""Coordinate reference systems and the projection presets that matter here.

The headline feature is first-class support for the South African Gauss
Conform (LO) zones. Legacy packages ship these as "user projections" that still
carry the Cape datum, which has been wrong for every project since the 1999
switch to Hartebeesthoek94, so the earth model has to be corrected by hand
for every project. Here the correct datum is the
default and the legacy Cape variant is a separate, clearly-labelled entry.

LO zones are south-oriented: the y axis increases west and the x axis
increases south. Getting that wrong mirrors the output, so the axis order is
pinned explicitly in the proj string rather than left to a datum lookup.

Much South African data is nevertheless stored the other way: the same
Gauss Conform zone as an ordinary north-oriented transverse Mercator, so
coordinates read as an easting and a northing, both negative west of the
meridian and south of the equator. ArcGIS commonly presents Lo data like
this, and so do many DEMs. The numbers are the same distances with the signs
flipped, so each zone is offered both ways and labelled plainly; picking the
one that matches the data is what keeps a reference DEM, the control and the
output grid in agreement.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError

__all__ = [
    "ProjectionPreset",
    "PRESETS",
    "resolve_crs",
    "describe_crs",
    "make_transformer",
    "list_presets",
]


@dataclass(frozen=True)
class ProjectionPreset:
    key: str
    label: str
    group: str
    definition: str
    datum_label: str
    notes: str = ""


def _lo_zone_tm(meridian: int, datum: str) -> str:
    """The same Lo zone, north-oriented: negative eastings and northings."""
    return (
        f"+proj=tmerc +lat_0=0 +lon_0={meridian} +k=1 +x_0=0 +y_0=0 "
        f"+units=m {datum} +no_defs"
    )


def _lo_zone(meridian: int, datum: str) -> str:
    """South-oriented transverse Mercator for a SA LO zone.

    ``+axis=wsu`` is what makes it south-oriented; without it the coordinates
    come out negated and every GCP appears mirrored.
    """
    return (
        f"+proj=tmerc +lat_0=0 +lon_0={meridian} +k=1 +x_0=0 +y_0=0 "
        f"+axis=wsu +units=m {datum} +no_defs"
    )


_HART = "+ellps=WGS84 +towgs84=0,0,0,0,0,0,0"
_CAPE = "+ellps=clrk80 +towgs84=-136,-108,-292,0,0,0,0"
_WGS = "+datum=WGS84"

_LO_MERIDIANS = (15, 17, 19, 21, 23, 25, 27, 29, 31, 33)

PRESETS: dict[str, ProjectionPreset] = {}


def _register(preset: ProjectionPreset) -> None:
    PRESETS[preset.key] = preset


for _m in _LO_MERIDIANS:
    _register(
        ProjectionPreset(
            key=f"ZALO{_m}",
            label=f"ZA Lo{_m} (Hartebeesthoek94)",
            group="South Africa - Gauss Conform",
            definition=_lo_zone(_m, _HART),
            datum_label="D518 Hartebeesthoek94",
            notes="Correct datum for all South African work since 1999.",
        )
    )
    _register(
        ProjectionPreset(
            key=f"ZALO{_m}_EN",
            label=f"ZA Lo{_m} north-oriented, E/N (Hartebeesthoek94)",
            group="South Africa - Gauss Conform",
            definition=_lo_zone_tm(_m, _HART),
            datum_label="D518 Hartebeesthoek94",
            notes="Easting and northing, negative west of the meridian and south of "
                  "the equator. How ArcGIS and many DEMs store Lo data.",
        )
    )
    _register(
        ProjectionPreset(
            key=f"ZALO{_m}_WGS84",
            label=f"ZA Lo{_m} (WGS84)",
            group="South Africa - Gauss Conform",
            definition=_lo_zone(_m, _WGS),
            datum_label="D000 WGS84",
            notes="Hartebeesthoek94 is WGS84-aligned; use where data is tagged WGS84.",
        )
    )
    _register(
        ProjectionPreset(
            key=f"ZALO{_m}_CAPE",
            label=f"ZA Lo{_m} (Cape datum - legacy)",
            group="South Africa - Gauss Conform",
            definition=_lo_zone(_m, _CAPE),
            datum_label="D000 Cape",
            notes="Legacy only. Offsets roughly 300 m from Hartebeesthoek94.",
        )
    )

for _zone in range(32, 39):
    _register(
        ProjectionPreset(
            key=f"UTM{_zone}S",
            label=f"UTM Zone {_zone}S (WGS84)",
            group="UTM",
            definition=f"+proj=utm +zone={_zone} +south +datum=WGS84 +units=m +no_defs",
            datum_label="D000 WGS84",
        )
    )
    _register(
        ProjectionPreset(
            key=f"UTM{_zone}N",
            label=f"UTM Zone {_zone}N (WGS84)",
            group="UTM",
            definition=f"+proj=utm +zone={_zone} +datum=WGS84 +units=m +no_defs",
            datum_label="D000 WGS84",
        )
    )

_register(
    ProjectionPreset(
        key="EPSG:4326",
        label="WGS84 Geographic (lat/lon)",
        group="Geographic",
        definition="EPSG:4326",
        datum_label="D000 WGS84",
        notes="Degrees, not metres. Pixel spacing must be given in degrees.",
    )
)
_register(
    ProjectionPreset(
        key="EPSG:3857",
        label="Web Mercator",
        group="Global",
        definition="EPSG:3857",
        datum_label="D000 WGS84",
        notes="Display only -- not suitable as an orthorectification output.",
    )
)


def list_presets() -> list[dict]:
    """Presets grouped for the projection picker."""
    return [
        {
            "key": p.key,
            "label": p.label,
            "group": p.group,
            "datum": p.datum_label,
            "notes": p.notes,
        }
        for p in PRESETS.values()
    ]


@lru_cache(maxsize=256)
def resolve_crs(identifier: str) -> CRS:
    """Turn a preset key, EPSG code, proj string or WKT into a CRS.

    Raises ``ValueError`` with a readable message rather than leaking pyproj's
    internal exception text into the UI.
    """
    if not identifier:
        raise ValueError("No coordinate system specified")

    identifier = identifier.strip()
    preset = PRESETS.get(identifier)
    if preset is not None:
        return CRS.from_user_input(preset.definition)

    try:
        return CRS.from_user_input(identifier)
    except CRSError as exc:
        raise ValueError(f"Unrecognised coordinate system {identifier!r}: {exc}") from exc


@lru_cache(maxsize=64)
def file_crs(identifier: str) -> CRS:
    """The CRS to write into an output file, recognisable by other software.

    The presets are proj strings, which PROJ handles perfectly but which GDAL
    can only record as an anonymous "unknown" system. ArcGIS reads those
    GeoTIFF keys back as "Unknown Coordinate System". So a preset with an
    official EPSG equivalent is written as that code, and the north-oriented
    Lo zones, which have none, are written as a named transverse Mercator on
    the Hartebeesthoek94 datum. The coordinates are identical either way;
    only the label changes.
    """
    from pyproj.crs import ProjectedCRS
    from pyproj.crs.coordinate_operation import TransverseMercatorConversion

    key = (identifier or "").strip()
    for meridian in _LO_MERIDIANS:
        if key == f"ZALO{meridian}":
            return CRS.from_epsg(2046 + (meridian - 15) // 2)   # Hartebeesthoek94 / Lo15..Lo33
        if key == f"ZALO{meridian}_CAPE":
            return CRS.from_epsg(22275 + (meridian - 15))       # Cape / Lo15..Lo33
        if key == f"ZALO{meridian}_EN":
            return ProjectedCRS(
                name=f"Hartebeesthoek94 / Lo{meridian} (north-oriented)",
                geodetic_crs=CRS.from_epsg(4148),
                conversion=TransverseMercatorConversion(
                    latitude_natural_origin=0,
                    longitude_natural_origin=meridian,
                    false_easting=0,
                    false_northing=0,
                    scale_factor_natural_origin=1,
                ),
            )
    if key.startswith("UTM") and key[3:-1].isdigit() and key[-1] in "NS":
        zone = int(key[3:-1])
        return CRS.from_epsg((32700 if key[-1] == "S" else 32600) + zone)
    return resolve_crs(identifier)


def describe_crs(identifier: str) -> dict:
    """Human-readable summary used by the projection inspector."""
    crs = resolve_crs(identifier)
    preset = PRESETS.get(identifier)
    axis_info = [
        {"name": a.name, "abbrev": a.abbrev, "direction": a.direction, "unit": a.unit_name}
        for a in crs.axis_info
    ]
    # The code the output files are written with, which for the Lo zones is
    # known from the preset rather than recoverable from the proj string.
    try:
        epsg = file_crs(identifier).to_epsg()
    except Exception:
        epsg = crs.to_epsg()
    return {
        "identifier": identifier,
        "label": preset.label if preset else (crs.name or identifier),
        "datum": preset.datum_label if preset else (crs.datum.name if crs.datum else "Unknown"),
        "notes": preset.notes if preset else "",
        "isProjected": bool(crs.is_projected),
        "isGeographic": bool(crs.is_geographic),
        "unit": axis_info[0]["unit"] if axis_info else "unknown",
        "axes": axis_info,
        "epsg": epsg,
        # No EPSG code: a definition Fiducia carries itself (the north-oriented
        # E/N Lo zones, Lo zones on WGS84). Shown so nobody mistakes it for one.
        "custom": epsg is None,
        "definition": preset.definition if preset else (crs.to_proj4() or ""),
        "proj4": crs.to_proj4() if crs.is_projected or crs.is_geographic else "",
        "wkt": crs.to_wkt(pretty=True),
    }


@lru_cache(maxsize=128)
def make_transformer(source: str, target: str) -> Optional[Transformer]:
    """Cached transformer between two systems; ``None`` when they match.

    Returning ``None`` for the identity case lets callers skip the transform
    entirely, which matters when the caller is resampling millions of pixels.
    """
    src = resolve_crs(source)
    dst = resolve_crs(target)
    if src.equals(dst):
        return None
    return Transformer.from_crs(src, dst, always_xy=True)


def transform_points(source: str, target: str, xs, ys):
    """Transform coordinate arrays, short-circuiting the identity case."""
    transformer = make_transformer(source, target)
    if transformer is None:
        return xs, ys
    return transformer.transform(xs, ys)
