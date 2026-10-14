"""The alignment as a COLMAP model, for Gaussian splatting and other tools.

Gaussian splatting (Kerbl et al., 2023) and most tools built on it --
Postshot, Nerfstudio, Brush, gsplat -- start from the cameras and sparse
points of a structure-from-motion run, read in COLMAP's layout:

    images/             the photos, undistorted
    sparse/0/cameras.txt   each photo's pinhole camera
    sparse/0/images.txt    each photo's pose and the points it sees
    sparse/0/points3D.txt  the tie points, coloured, with their tracks

Writing Fiducia's own alignment there saves running COLMAP again. The
photos are undistorted (splatting expects a plain pinhole camera), reduced
to a size the training can hold, and placed in the model's upright,
scaled frame, z up.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Optional

import numpy as np

__all__ = ["export_colmap"]


def _quaternion(R: np.ndarray) -> np.ndarray:
    """A rotation matrix as COLMAP's unit quaternion (w, x, y, z)."""
    t = np.trace(R)
    if t > 0:
        s = 2.0 * math.sqrt(t + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def export_colmap(folder: str, photos: list, poses: dict, lenses: dict, transform: dict,
                  points: np.ndarray, colours: np.ndarray, observations: list,
                  max_px: Optional[int] = 3200,
                  progress: Optional[Callable[[float, str], None]] = None,
                  should_cancel: Optional[Callable[[], bool]] = None) -> dict:
    """Write the model into ``folder`` in COLMAP's text layout.

    ``photos`` are each placed photo's ``path``, ``name``, ``width`` and
    ``height`` by index; ``poses`` (rvec, t: world to camera) and ``lenses``
    (focal, cx, cy, k1, k2, full-size pixels) by the same index; ``points``,
    ``colours`` and ``observations`` (index, column, row) the sparse model,
    all in the reconstruction's frame. ``transform`` places it in the scene.
    """
    import cv2
    from PIL import Image

    from .scene import _rotation, apply_transform

    say = progress or (lambda fraction, message: None)
    cancelled = should_cancel or (lambda: False)
    root = Path(folder)
    image_dir = root / "images"
    model_dir = root / "sparse" / "0"
    image_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    s = float(transform["scale"])
    Rw = np.asarray(transform["rotation"], float)
    shift = np.asarray(transform["shift"], float)

    order = sorted(poses)
    names, cameras, frames, maps = {}, [], [], {}
    used = set()
    for n, i in enumerate(order):
        if cancelled():
            raise InterruptedError("Cancelled")
        photo, lens = photos[i], lenses[i]
        width, height = int(photo["width"]), int(photo["height"])
        K = np.array([[lens["focal"], 0, lens["cx"]], [0, lens["focal"], lens["cy"]], [0, 0, 1.0]])
        D = np.array([lens.get("k1", 0.0), lens.get("k2", 0.0), 0.0, 0.0])
        # The largest undistorted view with no empty corners, then reduced.
        newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (width, height), 0, (width, height))
        factor = min(1.0, (max_px or max(width, height)) / max(width, height))
        out_w, out_h = max(1, round(width * factor)), max(1, round(height * factor))
        P = newK.copy()
        P[0, 0] *= factor
        P[1, 1] *= factor
        P[0, 2] = (newK[0, 2] + 0.5) * factor - 0.5
        P[1, 2] = (newK[1, 2] + 0.5) * factor - 0.5
        # Undistorted at full size, then reduced by area, so fine texture does not alias.
        map_x, map_y = cv2.initUndistortRectifyMap(K, D, np.eye(3), newK, (width, height), cv2.CV_32FC1)
        with Image.open(photo["path"]) as image:
            pixels = np.asarray(image.convert("RGB"))
        undistorted = cv2.remap(pixels, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        del map_x, map_y, pixels
        if factor < 1:
            undistorted = cv2.resize(undistorted, (out_w, out_h), interpolation=cv2.INTER_AREA)
        stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in photo.get("name") or f"photo_{n + 1}")
        name = f"{stem}.jpg"
        while name in used:
            name = f"{stem}_{n + 1}.jpg"
        used.add(name)
        Image.fromarray(undistorted).save(image_dir / name, quality=95)
        names[i] = name
        cameras.append((n + 1, out_w, out_h, P[0, 0], P[1, 1], P[0, 2], P[1, 2]))
        # The pose in the scene's frame: X_cam = R_c Rw^T (X_scene - shift) / s + t_c,
        # with the camera's coordinates scaled by s alike.
        Rc = _rotation(poses[i]["rvec"])
        R = Rc @ Rw.T
        t = s * np.asarray(poses[i]["t"], float) - R @ shift
        frames.append((n + 1, R, t))
        maps[i] = (K, D, P)
        say(0.9 * (n + 1) / len(order), f"Undistorted {n + 1} of {len(order)} photos")

    # Each observation, undistorted into its photo's new pinhole camera.
    index = {i: n for n, i in enumerate(order)}
    placed = apply_transform(transform, np.asarray(points, float))
    per_image: dict = {i: [] for i in order}
    tracks = []
    for p, obs in enumerate(observations):
        track = []
        for i, col, row in obs:
            if i not in maps:
                continue
            K, D, P = maps[i]
            uv = cv2.undistortPoints(np.array([[[col, row]]], float), K, D, P=P).reshape(2)
            per_image[i].append((uv[0], uv[1], p + 1))
            track.append((index[i] + 1, len(per_image[i]) - 1))
        tracks.append(track)

    say(0.95, "Writing the model")
    with open(model_dir / "cameras.txt", "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n"
                "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
                f"# Number of cameras: {len(cameras)}\n")
        for cid, w, h, fx, fy, cx, cy in cameras:
            f.write(f"{cid} PINHOLE {w} {h} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")
    with open(model_dir / "images.txt", "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n"
                "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
                "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
                f"# Number of images: {len(frames)}\n")
        for (iid, R, t), i in zip(frames, order):
            q = _quaternion(R)
            f.write(f"{iid} {q[0]:.12f} {q[1]:.12f} {q[2]:.12f} {q[3]:.12f} "
                    f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {iid} {names[i]}\n")
            f.write(" ".join(f"{x:.3f} {y:.3f} {pid}" for x, y, pid in per_image[i]) + "\n")
    with open(model_dir / "points3D.txt", "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n"
                "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
                f"# Number of points: {len(placed)}\n")
        for p, (xyz, rgb, track) in enumerate(zip(placed, np.asarray(colours, np.uint8), tracks)):
            pairs = " ".join(f"{iid} {k}" for iid, k in track)
            f.write(f"{p + 1} {xyz[0]:.9f} {xyz[1]:.9f} {xyz[2]:.9f} {rgb[0]} {rgb[1]} {rgb[2]} 0 {pairs}\n")
    say(1.0, f"{len(frames)} photos and {len(placed):,} points written")
    return {"folder": str(root), "photos": len(frames), "points": int(len(placed)),
            "maxPx": max_px}
