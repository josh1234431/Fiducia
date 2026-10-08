"""Finding where each photo lies on a reference image, with no clicks.

Automatic ground control needs a rough idea of where a photo lands before it
can search: either a solved orientation, or three points measured by hand on
every photo. On a four-photo block that is a dozen careful clicks before any
automation helps, and it is the slowest part of setting up a block.

A raw aerial photo and a rectified reference differ in scale, rotation and a
little perspective, which is exactly what scale- and rotation-invariant
features (SIFT) are built to see through. So:

1. The reference is read once, small (about 2 m pixels), as one image even
   when it comes as several tiles.
2. Each photo is read small too, and its features matched to the reference;
   a homography fitted by RANSAC places it.
3. A photo that does not match the reference well -- new building, seasonal
   change, sea -- is placed through an overlapping photo that did: photo to
   photo is the same sensor at the same scale, and matches far more easily.

The result per photo is a set of seed points (pixel -> ground) spread over the
frame, which is all automatic control needs to start. Terrain is not modelled
here; automatic control and the adjustment take it from there.
"""

from __future__ import annotations

import math
import os
import xml.sax.saxutils as xml
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

__all__ = ["Placement", "build_reference_vrt", "place_photos"]

# Ground resolution the reference is matched at, and the long side photos are
# read at. Small enough to be quick on a 260-megapixel photo, large enough to
# keep the features that survive both.
REFERENCE_RES_M = 2.0
PHOTO_LONG_SIDE = 2500
MIN_INLIERS = 25          # against the reference
MIN_CHAIN_INLIERS = 60    # photo to photo


@dataclass
class Placement:
    image_id: str
    name: str
    placed: bool
    method: str = ""                 # "reference", "via <photo>", or why it failed
    inliers: int = 0
    gsd_m: float = 0.0               # ground size of one full-resolution pixel
    footprint: list = field(default_factory=list)   # four corners, control CRS
    seeds: list = field(default_factory=list)       # (col, row, x, y), full-res pixels

    def to_dict(self) -> dict:
        return {
            "imageId": self.image_id, "name": self.name, "placed": self.placed,
            "method": self.method, "inliers": self.inliers, "gsdM": self.gsd_m,
            "footprint": self.footprint,
            "seeds": [list(s) for s in self.seeds],
        }


def build_reference_vrt(paths: Sequence[str], target: str) -> str:
    """One virtual image over several reference tiles, without copying them.

    The tiles must share a coordinate system, pixel size and band layout, as
    tiles cut from one orthomosaic do. Returns the VRT's path.
    """
    import rasterio

    infos = []
    for path in paths:
        with rasterio.open(path) as d:
            infos.append((path, d.bounds, d.res, d.crs, d.count, d.dtypes[0], d.width, d.height))
    _, _, res, crs, count, dtype, _, _ = infos[0]
    for path, _, r, c, n, t, _, _ in infos[1:]:
        if c != crs or n != count or t != dtype or abs(r[0] - res[0]) > 1e-6 * res[0]:
            raise ValueError(
                f"{os.path.basename(path)} does not match {os.path.basename(infos[0][0])} "
                "in coordinate system, pixel size or bands, so they cannot be used as one "
                "reference. Use tiles cut from the same orthomosaic."
            )
    west = min(i[1].left for i in infos)
    north = max(i[1].top for i in infos)
    east = max(i[1].right for i in infos)
    south = min(i[1].bottom for i in infos)
    width = int(round((east - west) / res[0]))
    height = int(round((north - south) / res[1]))
    gdal_type = {"uint8": "Byte", "uint16": "UInt16", "int16": "Int16", "float32": "Float32",
                 "uint32": "UInt32", "int32": "Int32", "float64": "Float64"}.get(dtype, "Byte")

    lines = [f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">',
             f"  <SRS>{xml.escape(crs.to_wkt())}</SRS>",
             f"  <GeoTransform>{west!r}, {res[0]!r}, 0.0, {north!r}, 0.0, {-res[1]!r}</GeoTransform>"]
    for band in range(1, count + 1):
        lines.append(f'  <VRTRasterBand dataType="{gdal_type}" band="{band}">')
        if count >= 3 and band <= 3:
            lines.append(f"    <ColorInterp>{('Red', 'Green', 'Blue')[band - 1]}</ColorInterp>")
        for path, bounds, _, _, _, _, w, h in infos:
            x_off = int(round((bounds.left - west) / res[0]))
            y_off = int(round((north - bounds.top) / res[1]))
            lines += [
                "    <SimpleSource>",
                f'      <SourceFilename relativeToVRT="0">{xml.escape(os.path.abspath(path))}</SourceFilename>',
                f"      <SourceBand>{band}</SourceBand>",
                f'      <SrcRect xOff="0" yOff="0" xSize="{w}" ySize="{h}" />',
                f'      <DstRect xOff="{x_off}" yOff="{y_off}" xSize="{w}" ySize="{h}" />',
                "    </SimpleSource>",
            ]
        lines.append("  </VRTRasterBand>")
    lines.append("</VRTDataset>")

    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return target


def _grey(dataset, out_shape):
    import cv2
    from rasterio.enums import Resampling

    if dataset.count >= 3:
        rgb = dataset.read([1, 2, 3], out_shape=(3, *out_shape), resampling=Resampling.average)
        return cv2.cvtColor(np.ascontiguousarray(np.transpose(rgb, (1, 2, 0))), cv2.COLOR_RGB2GRAY)
    band = dataset.read(1, out_shape=out_shape, resampling=Resampling.average)
    return cv2.normalize(band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def place_photos(
    images: Sequence[dict],
    reference_path: str,
    control_crs: Optional[str] = None,
    progress: Optional[Callable[[float, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> list[Placement]:
    """Place each photo on the reference; see the module notes."""
    import cv2
    import rasterio

    def say(fraction, message):
        if progress:
            progress(fraction, message)

    def cancelled():
        if should_cancel and should_cancel():
            raise InterruptedError("Cancelled")

    say(0.02, "Reading the reference")
    with rasterio.open(reference_path) as reference:
        step = REFERENCE_RES_M / abs(reference.transform.a)
        shape = (max(1, int(reference.height / max(step, 1))), max(1, int(reference.width / max(step, 1))))
        ref = _grey(reference, shape)
        ref_transform = reference.transform * reference.transform.scale(
            reference.width / shape[1], reference.height / shape[0])
        ref_crs = reference.crs

    to_control = None
    if control_crs and ref_crs is not None:
        from pyproj import Transformer

        from .geodesy import resolve_crs

        target = resolve_crs(control_crs)
        if not target.equals(ref_crs):
            to_control = Transformer.from_crs(ref_crs, target, always_xy=True).transform

    sift = cv2.SIFT_create(nfeatures=20000)
    ref_kp, ref_desc = sift.detectAndCompute(ref, None)
    if ref_desc is None or len(ref_kp) < 50:
        raise ValueError("The reference image has too little detail to match photos against.")
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))

    def matches(desc_a, kp_a, desc_b, kp_b, ratio=0.75):
        pairs = matcher.knnMatch(desc_a, desc_b, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < ratio * n.distance]
        return (np.float32([kp_a[m.queryIdx].pt for m in good]),
                np.float32([kp_b[m.trainIdx].pt for m in good]))

    # Features of every photo, read small.
    photos = []
    for index, image in enumerate(images):
        cancelled()
        say(0.05 + 0.45 * index / max(1, len(images)), f"Reading {image.get('name')}")
        with rasterio.open(image["path"]) as dataset:
            factor = max(1.0, max(dataset.width, dataset.height) / PHOTO_LONG_SIDE)
            small = _grey(dataset, (int(dataset.height / factor), int(dataset.width / factor)))
        kp, desc = sift.detectAndCompute(small, None)
        photos.append({"image": image, "factor": factor, "shape": small.shape, "kp": kp, "desc": desc,
                       "H": None, "method": "", "inliers": 0})

    def plausible(H, shape):
        """A homography that keeps the frame a convex, non-degenerate shape."""
        if H is None or not np.isfinite(H).all():
            return False
        h, w = shape
        corners = cv2.perspectiveTransform(
            np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2), H).reshape(-1, 2)
        return cv2.isContourConvex(corners.astype(np.float32)) and cv2.contourArea(corners) > 100

    # 1. Each photo against the reference.
    for index, photo in enumerate(photos):
        cancelled()
        say(0.5 + 0.3 * index / max(1, len(photos)), f"Placing {photo['image'].get('name')}")
        if photo["desc"] is None:
            continue
        src, dst = matches(photo["desc"], photo["kp"], ref_desc, ref_kp)
        if len(src) < 8:
            continue
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 8.0)
        count = int(mask.sum()) if mask is not None else 0
        if count >= MIN_INLIERS and plausible(H, photo["shape"]):
            photo.update(H=H, method="reference", inliers=count)

    # 2. The rest through an overlapping photo that is already placed.
    progress_made = True
    while progress_made:
        progress_made = False
        for photo in (p for p in photos if p["H"] is None and p["desc"] is not None):
            best = None
            for other in (p for p in photos if p["H"] is not None):
                src, dst = matches(photo["desc"], photo["kp"], other["desc"], other["kp"])
                if len(src) < 8:
                    continue
                H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
                count = int(mask.sum()) if mask is not None else 0
                if count >= MIN_CHAIN_INLIERS and (best is None or count > best[1]):
                    best = (other["H"] @ H, count, other["image"].get("name"))
            if best and plausible(best[0], photo["shape"]):
                photo.update(H=best[0], method=f"via {best[2]}", inliers=best[1])
                progress_made = True

    # Results, in full-resolution pixels and control coordinates. Feature
    # positions have whole numbers at pixel centres; a raster transform has
    # them at pixel edges, hence the half pixel.
    def to_ground(points):
        xs, ys = ref_transform * (points[:, 0] + 0.5, points[:, 1] + 0.5)
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        if to_control:
            xs, ys = to_control(xs, ys)
        return np.column_stack([xs, ys])

    out = []
    for photo in photos:
        image = photo["image"]
        if photo["H"] is None:
            out.append(Placement(image["id"], image.get("name", ""), False,
                                 method="no match against the reference or an overlapping photo"))
            continue
        h, w = photo["shape"]
        f = photo["factor"]
        corners_small = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        grid_small = np.float32([[w * u, h * v] for v in (0.15, 0.5, 0.85) for u in (0.15, 0.5, 0.85)])
        mapped = cv2.perspectiveTransform(
            np.vstack([corners_small, grid_small]).reshape(-1, 1, 2), photo["H"]).reshape(-1, 2)
        ground = to_ground(mapped)
        corners, grid = ground[:4], ground[4:]
        # Ground size of a full-resolution pixel, from the frame's mean side.
        side = np.mean([np.hypot(*(corners[(i + 1) % 4] - corners[i])) /
                        (w if i % 2 == 0 else h) for i in range(4)]) / f
        # A reduced pixel's centre is (i + 0.5) f - 0.5 in full-resolution pixels.
        seeds = [(float((gx + 0.5) * f - 0.5), float((gy + 0.5) * f - 0.5), float(x), float(y))
                 for (gx, gy), (x, y) in zip(grid_small, grid)]
        out.append(Placement(image["id"], image.get("name", ""), True, photo["method"],
                             photo["inliers"], float(side),
                             [[float(x), float(y)] for x, y in corners], seeds))
    say(1.0, "Placed")
    return out
