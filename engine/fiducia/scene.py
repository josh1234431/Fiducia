"""Photos without positions: from a phone or camera walk-around to a 3D model.

Structure from motion (``sfm``) places the photos and the points they share
in a frame of its own. Around it this module does what a set of everyday
photos needs:

* photos are made upright (a phone stores a portrait photo on its side, with
  a tag saying which way up it goes) and HEIC photos are converted, so every
  later step sees what the person saw;
* photos are grouped by camera -- make, model, lens and sensor -- because two
  phones, or one phone's wide and zoom lenses, each have a lens of their own;
* each camera's focal length comes from the photos' 35 mm equivalent, the
  one figure every phone records;
* the model is turned upright, from which way down the photos were held, and
  given a scale: from a known distance the person measures, or, roughly, from
  the phones' own GPS when the photos were taken far enough apart.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .drone import FULL_FRAME_DIAGONAL_MM, PhotoMeta, pixel_pitch_mm, read_metadata

__all__ = ["HEIC_EXTENSIONS", "heic_supported", "photo_info", "make_upright", "camera_groups",
           "level", "gps_scale", "apply_transform", "sparse_colours", "write_cloud", "read_cloud"]

HEIC_EXTENSIONS = {".heic", ".heif", ".hif"}
# Files a camera or phone writes, which may be stored on their side.
PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png"} | HEIC_EXTENSIONS
ORIENTATION_TAG = 0x0112
# Without a focal length on record, a phone's main camera is close to 26 mm
# equivalent; the reconstruction then refines it freely.
GUESSED_FOCAL_35MM = 26.0


def heic_supported() -> bool:
    """Whether HEIC photos can be read: they need the optional pillow-heif."""
    try:
        import pillow_heif  # noqa: F401
    except ImportError:
        return False
    return True


def _register_heic() -> None:
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except ImportError:
        pass


def _is_heic(path: str) -> bool:
    return Path(path).suffix.lower() in HEIC_EXTENSIONS


def is_photo(path: str) -> bool:
    """A camera's or phone's photo, as opposed to a scan or an image product."""
    return Path(path).suffix.lower() in PHOTO_EXTENSIONS


def photo_info(path: str) -> dict:
    """What a photo says about its camera, read from the file as it came."""
    _register_heic()
    from PIL import Image

    if _is_heic(path) and not heic_supported():
        raise ValueError("HEIC photos need the HEIC reader (pillow-heif), which is not installed")
    meta = read_metadata(path)
    with Image.open(path) as image:
        orientation = int(image.getexif().get(ORIENTATION_TAG, 1) or 1)
    # Orientations 5 to 8 turn the photo through a quarter: its sides swap.
    width, height = (meta.height, meta.width) if orientation in (5, 6, 7, 8) else (meta.width, meta.height)
    return {
        "make": meta.make, "model": meta.model,
        "focalMm": meta.focal_mm, "focal35mm": meta.focal_35mm,
        "focalPlaneXRes": meta.focal_plane_x_res, "focalPlaneUnitMm": meta.focal_plane_unit_mm,
        "orientation": orientation,
        "width": width, "height": height,       # upright
        "latitude": meta.latitude, "longitude": meta.longitude, "altitude": meta.altitude,
        "taken": meta.taken,
    }


def make_upright(path: str, folder: str) -> str:
    """The photo as it should be seen, as a file every reader understands.

    A JPEG or TIFF already the right way up is used where it is. A photo
    stored on its side, or a HEIC, is written upright as a high-quality JPEG
    into ``folder``, its camera details kept, and that copy's path returned.
    """
    _register_heic()
    from PIL import Image, ImageOps

    heic = _is_heic(path)
    if heic and not heic_supported():
        raise ValueError("HEIC photos need the HEIC reader (pillow-heif), which is not installed")
    with Image.open(path) as image:
        orientation = int(image.getexif().get(ORIENTATION_TAG, 1) or 1)
        if not heic and orientation == 1:
            return path
        stat = Path(path).stat()
        tag = hashlib.sha1(f"{Path(path).resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()[:8]
        target = Path(folder) / f"{Path(path).stem}_{tag}.jpg"
        if target.exists():
            return str(target)
        exif = image.getexif()
        upright = ImageOps.exif_transpose(image)
        if upright.mode not in ("RGB", "L"):
            upright = upright.convert("RGB")
        exif[ORIENTATION_TAG] = 1
        Path(folder).mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".part")
        upright.save(partial, "JPEG", quality=95, subsampling=0, exif=exif.tobytes())
        partial.replace(target)
    return str(target)


def focal_px(info: dict) -> tuple[float, bool, str]:
    """A photo's focal length in pixels of its upright frame, whether it is
    on record or a guess, and where it came from."""
    width, height = int(info["width"]), int(info["height"])
    long_side, short_side = max(width, height), min(width, height)
    meta = PhotoMeta(path="", width=long_side, height=short_side,
                     focal_mm=info.get("focalMm"), focal_35mm=info.get("focal35mm"),
                     focal_plane_x_res=info.get("focalPlaneXRes"),
                     focal_plane_unit_mm=info.get("focalPlaneUnitMm"))
    pitch, source = pixel_pitch_mm(meta)
    if pitch and meta.focal_mm:
        return meta.focal_mm / pitch, True, source
    if meta.focal_35mm:
        # Against the diagonal of the whole 4:3 sensor for a widescreen crop,
        # as pixel_pitch_mm does.
        diagonal = math.hypot(long_side, 0.75 * long_side) if short_side / long_side < 0.7 \
            else math.hypot(long_side, short_side)
        return meta.focal_35mm / FULL_FRAME_DIAGONAL_MM * diagonal, True, "35 mm equivalent focal length"
    diagonal = math.hypot(long_side, short_side)
    return GUESSED_FOCAL_35MM / FULL_FRAME_DIAGONAL_MM * diagonal, False, "a typical phone lens (no focal length on record)"


def camera_groups(infos: Sequence[dict]) -> tuple[list[int], list[dict]]:
    """Each photo's camera, and the cameras: one per make, model, lens and sensor.

    A portrait photo and a landscape one from the same lens are one camera.
    """
    keys: dict = {}
    camera_of, groups = [], []
    for info in infos:
        focal, known, source = focal_px(info)
        sides = tuple(sorted((int(info["width"]), int(info["height"]))))
        key = (info.get("make") or "", info.get("model") or "", round(info.get("focalMm") or 0.0, 2),
               round(info.get("focal35mm") or 0.0, 1), sides)
        if key not in keys:
            keys[key] = len(groups)
            name = " ".join(p for p in (info.get("make"), info.get("model")) if p) or "Unknown camera"
            if info.get("focal35mm"):
                name += f", {info['focal35mm']:.0f} mm equivalent"
            elif info.get("focalMm"):
                name += f", {info['focalMm']:.1f} mm"
            groups.append({"name": name, "focalPx": focal, "focalKnown": known, "focalSource": source,
                           "size": [sides[1], sides[0]], "photos": 0})
        camera_of.append(keys[key])
        groups[keys[key]]["photos"] += 1
    return camera_of, groups


# -- the model's frame ----------------------------------------------------------------------


def _rotation(rvec) -> np.ndarray:
    import cv2

    return cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))[0]


def level(poses: dict, points: np.ndarray) -> dict:
    """A transform that stands the model upright, centred, its unit unchanged.

    People hold a camera level more or less, so the photos' downward axis,
    averaged, is gravity. Photos all looking the same way down (a drone, a
    camera on a pole) are levelled from that instead. The horizontal axes
    follow the spread of the points, and the origin is at their middle, on
    the lowest of them.
    """
    rotations = [_rotation(p["rvec"]) for p in poses.values()]
    rights = np.array([R[0] for R in rotations])              # the image's x axis, in the world
    down = np.mean([R[1] for R in rotations], axis=0)        # ... and its y axis
    forward = np.mean([R[2] for R in rotations], axis=0)
    centres = np.array([p["centre"] for p in poses.values()])
    # A photo tilted up or down still has a level x axis; only turning the
    # phone on its side tilts it. So up is the direction most nearly square
    # to every photo's x axis, which tilt cannot bias as it does the average
    # of the y axes -- when the photos face different ways.
    values, vectors = np.linalg.eigh(rights.T @ rights)
    if np.linalg.norm(forward) > 0.7 and np.linalg.norm(down) < 0.5:
        up, source = -forward / np.linalg.norm(forward), "the photos looking down"
    elif np.linalg.norm(down) > 0.5 and values[1] > 0.05 * len(rights):
        up = vectors[:, 0] if vectors[:, 0] @ -down >= 0 else -vectors[:, 0]
        source = "how the photos were held"
    elif np.linalg.norm(down) > 0.5:
        # All facing one way: the y axes, less their share along the x axes.
        mean_right = rights.mean(axis=0)
        mean_right /= max(np.linalg.norm(mean_right), 1e-12)
        up = -down - (-down @ mean_right) * mean_right
        up, source = up / np.linalg.norm(up), "how the photos were held"
    else:
        # The cameras spread over a surface; its normal, on the side the
        # photos were held upright.
        _, _, vt = np.linalg.svd(centres - centres.mean(axis=0))
        up = vt[2] if vt[2] @ -down >= 0 else -vt[2]
        source = "the spread of the cameras"
    z = up / np.linalg.norm(up)
    cloud = points if len(points) >= 3 else centres
    flat = cloud - np.outer(cloud @ z, z)
    flat = flat - flat.mean(axis=0)
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    x = vt[0] - (vt[0] @ z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rotation = np.array([x, y, z])
    turned = cloud @ rotation.T
    shift = -np.array([np.median(turned[:, 0]), np.median(turned[:, 1]), np.percentile(turned[:, 2], 2)])
    return {"scale": 1.0, "rotation": rotation.tolist(), "shift": shift.tolist(),
            "upFrom": source, "scaleSource": "none", "units": "model"}


def apply_transform(transform: dict, points: np.ndarray) -> np.ndarray:
    """Model-frame points into the scene: scaled, turned and shifted."""
    R = np.asarray(transform["rotation"], float)
    return float(transform["scale"]) * (np.asarray(points, float) @ R.T) + np.asarray(transform["shift"], float)


def gps_scale(transform: dict, centres: dict, infos: dict, min_spread_m: float = 10.0) -> Optional[dict]:
    """The levelled model placed on the map from the photos' own GPS, roughly.

    A phone's position is good to a few metres, so this needs photos spread
    over more than ``min_spread_m``; only the scale, heading and position are
    taken from it, never the tilt, which the photos give far better. Returns
    the new transform (in a UTM zone), or None.
    """
    from pyproj import Transformer

    from .drone import suggest_crs

    with_gps = [k for k in centres if infos.get(k, {}).get("latitude") is not None
                and infos[k].get("longitude") is not None]
    if len(with_gps) < 4:
        return None
    first = infos[with_gps[0]]
    crs = suggest_crs(first["latitude"], first["longitude"])
    to_map = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    ground = np.array([to_map.transform(infos[k]["longitude"], infos[k]["latitude"]) for k in with_gps])
    spread = float(np.sqrt(((ground - ground.mean(axis=0)) ** 2).sum(axis=1).mean()))
    if spread < min_spread_m:
        return None
    model = apply_transform(transform, np.array([centres[k] for k in with_gps]))
    # A similarity in the plane: scale, heading and shift.
    a, b = model[:, :2] - model[:, :2].mean(axis=0), ground - ground.mean(axis=0)
    za, zb = a[:, 0] + 1j * a[:, 1], b[:, 0] + 1j * b[:, 1]
    factor = (np.conj(za) @ zb) / max((np.abs(za) ** 2).sum(), 1e-12)
    scale, heading = float(abs(factor)), float(np.angle(factor))
    residual = float(np.sqrt(np.mean(np.abs(zb - za * factor) ** 2)))
    if residual > 0.5 * spread:
        return None
    turn = np.array([[math.cos(heading), -math.sin(heading), 0], [math.sin(heading), math.cos(heading), 0],
                     [0, 0, 1]])
    altitudes = [infos[k]["altitude"] for k in with_gps if infos[k].get("altitude") is not None]
    rotation = turn @ np.asarray(transform["rotation"], float)
    total = scale * float(transform["scale"])
    placed = total * (np.array([centres[k] for k in with_gps]) @ rotation.T)
    shift = np.zeros(3)
    shift[:2] = ground.mean(axis=0) - placed[:, :2].mean(axis=0)
    shift[2] = (float(np.median(altitudes)) - float(np.median(placed[:, 2]))) if altitudes else \
        -float(np.median(placed[:, 2]))
    return {**transform, "scale": total, "rotation": rotation.tolist(), "shift": shift.tolist(),
            "scaleSource": "gps", "units": "m", "crs": crs,
            "gpsResidualM": round(residual, 2), "gpsSpreadM": round(spread, 1), "gpsPhotos": len(with_gps)}


# -- colour and files --------------------------------------------------------------------------


def _colour_image(path: str, max_px: int):
    """A reduced colour copy and its reduction factor."""
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGB")
        factor = max(1.0, max(image.size) / max_px)
        if factor > 1.0:
            image = image.resize((max(1, int(image.width / factor)), max(1, int(image.height / factor))),
                                 Image.BILINEAR)
        return np.asarray(image), factor


def sparse_colours(paths: Sequence[str], observations: Sequence[list], max_px: int = 1200) -> np.ndarray:
    """Each point's colour, from the first photo that sees it."""
    colours = np.full((len(observations), 3), 160, np.uint8)
    by_photo: dict = {}
    for n, obs in enumerate(observations):
        if obs:
            i, col, row = obs[0]
            by_photo.setdefault(int(i), []).append((n, col, row))
    for i, members in by_photo.items():
        try:
            image, factor = _colour_image(paths[i], max_px)
        except Exception:  # noqa: BLE001 -- a photo that cannot be read leaves its points grey
            continue
        idx = np.array([m[0] for m in members])
        cols = np.clip(((np.array([m[1] for m in members]) + 0.5) / factor - 0.5).round().astype(int), 0, image.shape[1] - 1)
        rows = np.clip(((np.array([m[2] for m in members]) + 0.5) / factor - 0.5).round().astype(int), 0, image.shape[0] - 1)
        colours[idx] = image[rows, cols, :3]
    return colours


def write_cloud(path: str, points: np.ndarray, colours: np.ndarray, crs: Optional[str] = None) -> dict:
    """Points with colour to a file: PLY, or LAS/LAZ for point cloud software."""
    suffix = Path(path).suffix.lower()
    points = np.asarray(points, float)
    colours = np.asarray(colours, np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    partial = str(path) + ".part"
    if suffix == ".ply":
        record = np.zeros(len(points), dtype=[("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
                                              ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        record["x"], record["y"], record["z"] = points.T
        record["red"], record["green"], record["blue"] = colours.T
        header = ("ply\nformat binary_little_endian 1.0\ncomment Fiducia\n"
                  f"element vertex {len(points)}\n"
                  "property double x\nproperty double y\nproperty double z\n"
                  "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        with open(partial, "wb") as handle:
            handle.write(header.encode("ascii"))
            handle.write(record.tobytes())
    elif suffix in (".las", ".laz"):
        import laspy

        header = laspy.LasHeader(point_format=2, version="1.2")
        span = float(np.ptp(points, axis=0).max()) if len(points) else 1.0
        # A thousandth of a millimetre for a small object, a millimetre for a site.
        resolution = 1e-4 if span < 50 else 1e-3
        header.scales = np.array([resolution] * 3)
        header.offsets = np.floor(points.min(axis=0)) if len(points) else np.zeros(3)
        if crs:
            try:
                from pyproj import CRS

                header.add_crs(CRS.from_user_input(crs))
            except Exception:  # noqa: BLE001 -- a cloud without its CRS is still a cloud
                pass
        cloud = laspy.LasData(header)
        cloud.x, cloud.y, cloud.z = points[:, 0], points[:, 1], points[:, 2]
        # LAS colour is 16-bit.
        cloud.red, cloud.green, cloud.blue = (colours.astype(np.uint16) * 257).T
        cloud.write(partial, do_compress=suffix == ".laz")
    else:
        raise ValueError(f"Point clouds are written as .ply, .las or .laz, not {suffix or 'no extension'}")
    Path(partial).replace(path)
    return {"path": str(path), "points": int(len(points))}


def read_cloud(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Points and colours saved by the reconstruction (an .npz of the model frame)."""
    with np.load(path) as data:
        return data["points"], data["colours"]
