"""Output formats for rasters Fiducia produces.

Orthophotos are always rendered to a GeoTIFF first: it is the one format
GDAL writes efficiently tile by tile from many processes. When another format
is wanted, that working file is converted once, here, and removed.

Lossy compression and a nodata value do not mix: JPEG smears the background
value into the image edge, leaving a dark fringe that GIS software then
partly hides. JPEG output therefore carries an internal transparency mask,
stored losslessly, instead of a nodata value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import rasterio
from rasterio.shutil import copy as gdal_copy

FORMATS: dict[str, dict] = {
    "GTiff": {"label": "GeoTIFF", "extension": ".tif", "compressions": ("DEFLATE", "LZW", "JPEG", "NONE")},
    "COG": {"label": "Cloud-Optimised GeoTIFF", "extension": ".tif", "compressions": ("DEFLATE", "LZW", "JPEG", "NONE")},
    "PCIDSK": {"label": "PCIDSK", "extension": ".pix", "compressions": ()},
    "HFA": {"label": "ERDAS Imagine", "extension": ".img", "compressions": ()},
    # Lossless only: JPEG 2000 codes an alpha channel lossily along with the
    # colour, which blurred the image edge over several pixels on real
    # imagery (2.5% of pixels on the wrong side). JPEG-compressed GeoTIFF or
    # COG covers the small-file case with an exact mask.
    "JP2OpenJPEG": {"label": "JPEG 2000", "extension": ".jp2", "compressions": ("LOSSLESS",)},
}

LOSSY = {"JPEG"}


def provenance(author: str = "", project: str = "", source: str = "") -> dict[str, str]:
    """Who made a raster, with what, when and from what.

    The TIFFTAG_ keys become the standard TIFF Artist, Software and DateTime
    tags, which ArcGIS and most GIS software display. The plain keys carry
    the same information in formats without TIFF tags.
    """
    from datetime import datetime

    from . import __version__

    now = datetime.now()
    software = f"Fiducia {__version__}"
    tags = {
        "SOFTWARE": software,
        "PROCESSING_DATE": now.isoformat(timespec="seconds"),
        "TIFFTAG_SOFTWARE": software,
        "TIFFTAG_DATETIME": now.strftime("%Y:%m:%d %H:%M:%S"),
    }
    if author.strip():
        tags.update(AUTHOR=author.strip(), TIFFTAG_ARTIST=author.strip())
    if project.strip():
        tags["PROJECT"] = project.strip()
    if source.strip():
        tags["SOURCE_IMAGE"] = source.strip()
    return tags


def write_provenance(path: str | Path, tags: dict[str, str]) -> None:
    with rasterio.open(str(path), "r+") as dataset:
        dataset.update_tags(**tags)


def list_formats() -> list[dict]:
    return [
        {"id": key, "label": value["label"], "extension": value["extension"],
         "compressions": list(value["compressions"])}
        for key, value in FORMATS.items()
    ]


def normalise(fmt: Optional[str], compress: Optional[str]) -> tuple[str, str]:
    """A known format and a compression that format supports."""
    fmt = fmt if fmt in FORMATS else "GTiff"
    options = FORMATS[fmt]["compressions"]
    compress = (compress or "").upper()
    if not options:
        return fmt, ""
    return fmt, compress if compress in options else options[0]


def extension_for(fmt: str) -> str:
    return FORMATS.get(fmt, FORMATS["GTiff"])["extension"]


def writes_directly(fmt: str, compress: str) -> bool:
    """Whether the renderer can write the final file itself, with no conversion."""
    return fmt == "GTiff" and compress not in LOSSY


def convert(
    source: str | Path,
    target: str | Path,
    fmt: str,
    compress: str = "",
    quality: int = 90,
    progress: Optional[Callable[[str], None]] = None,
) -> list[str]:
    """Convert a working GeoTIFF to the requested format. Returns warnings."""
    fmt, compress = normalise(fmt, compress)
    source, target = str(source), str(target)
    quality = int(min(100, max(10, quality)))
    warnings: list[str] = []
    Path(target).unlink(missing_ok=True)

    with rasterio.open(source) as src:
        eight_bit = all(dtype == "uint8" for dtype in src.dtypes)
    if compress in LOSSY and not eight_bit:
        warnings.append("JPEG compression needs 8-bit imagery; the output was compressed "
                        "losslessly instead.")
        compress = "DEFLATE" if fmt in ("GTiff", "COG") else "LOSSLESS"

    say = progress or (lambda _message: None)

    if fmt == "GTiff":
        say("Writing JPEG-compressed GeoTIFF")
        _masked_copy(source, target, compress="JPEG", quality=quality)
        _overviews(target, warnings)
    elif fmt == "COG":
        say("Writing Cloud-Optimised GeoTIFF")
        options = {"COMPRESS": compress, "BIGTIFF": "IF_SAFER", "NUM_THREADS": "ALL_CPUS",
                   "OVERVIEW_RESAMPLING": "AVERAGE", "RESAMPLING": "AVERAGE"}
        if compress == "JPEG":
            # The COG driver carries a source mask into the output, so build
            # one first from the nodata value, then compress.
            masked = target + ".masked.tif"
            try:
                _masked_copy(source, masked, compress="DEFLATE")
                gdal_copy(masked, target, driver="COG", QUALITY=str(quality), **options)
            finally:
                Path(masked).unlink(missing_ok=True)
        else:
            if compress in ("DEFLATE", "LZW"):
                options["PREDICTOR"] = "YES"
            gdal_copy(source, target, driver="COG", **options)
    elif fmt == "PCIDSK":
        say("Writing PCIDSK")
        gdal_copy(source, target, driver="PCIDSK", INTERLEAVING="TILED", TILESIZE="256")
        _overviews(target, warnings)
    elif fmt == "HFA":
        say("Writing ERDAS Imagine")
        gdal_copy(source, target, driver="HFA", COMPRESSED="YES")
        # A copy leaves the metadata in a .aux.xml sidecar, which is lost when
        # the .img is shared on its own. Written through an open file, it is
        # stored inside the .img.
        with rasterio.open(source) as src:
            tags = src.tags()
        with rasterio.open(target, "r+") as dst:
            dst.update_tags(**tags)
        Path(target + ".aux.xml").unlink(missing_ok=True)
        _overviews(target, warnings)
    elif fmt == "JP2OpenJPEG":
        say("Writing JPEG 2000")
        gdal_copy(source, target, driver="JP2OpenJPEG", BLOCKXSIZE="1024", BLOCKYSIZE="1024",
                  NUM_THREADS="ALL_CPUS", REVERSIBLE="YES", QUALITY="100",
                  WRITE_METADATA="YES", MAIN_MD_DOMAIN_ONLY="YES")

    return warnings


def _masked_copy(source: str, target: str, compress: str, quality: int = 90) -> None:
    """Copy block by block, replacing the nodata value with an internal mask."""
    with rasterio.open(source) as src:
        profile = src.profile.copy()
        profile.update(
            driver="GTiff", tiled=True, blockxsize=256, blockysize=256,
            compress=compress, BIGTIFF="IF_SAFER", nodata=None,
        )
        if compress == "JPEG":
            profile.update(jpeg_quality=quality)
            if src.count == 3:
                profile.update(photometric="YCBCR")
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(target, "w", **profile) as dst:
                dst.update_tags(**src.tags())
                for _, window in src.block_windows(1):
                    dst.write(src.read(window=window), window=window)
                    dst.write_mask(src.dataset_mask(window=window).astype(np.uint8), window=window)


def _overviews(target: str, warnings: list[str]) -> None:
    from . import raster

    try:
        raster.build_overviews(target)
    except Exception:  # noqa: BLE001
        warnings.append("Output written, but its overview pyramid could not be built.")
