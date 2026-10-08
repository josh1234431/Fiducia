"""Atmospheric refraction and earth curvature.

The collinearity equations assume straight rays and flat ground. Neither holds
exactly, and both errors are radial from the principal point:

* **Refraction** bends each ray through a denser atmosphere nearer the
  ground, so the camera sees every point slightly further from nadir than the
  straight chord: image points are displaced *outward*. For a horizontally
  layered atmosphere the refraction angle is delta = K tan(beta), beta the
  off-nadir angle, with K the height-averaged excess refractive index between
  the ground and the camera minus the excess at the camera. Here K is computed
  from the International Standard Atmosphere by that integral directly,
  rather than from the empirical fit K = 2410 H/(H^2 - 6H + 250) - ... found
  in textbooks, which differs from it by up to ~20% depending on the heights
  (checked by ray tracing; see scratch/test_corrections.py). In the image,
  r = f tan(beta) gives dr = K (r + r^3 / f^2).

* **Earth curvature**: control is given in a map projection with heights
  above a curved datum. A point at horizontal distance d from nadir lies
  d^2 / 2R below the plane the flat collinearity model assumes, so it appears
  closer to nadir: image points are displaced *inward* by
  dr = r^3 (H - h) / (2 R f^2), with H - h the flying height above ground.

``to_flat`` removes both from measured image coordinates, giving the
coordinates a straight ray over flat ground would have produced -- what the
adjustment needs. ``from_flat`` is its inverse, for the orthorectifier.
Both are approximations for near-vertical photography, as in all standard
treatments; they are radial about the principal point, not the nadir.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional

import numpy as np

__all__ = ["refraction_constant", "radial_displacement", "to_flat", "from_flat",
           "EARTH_RADIUS_M"]

EARTH_RADIUS_M = 6371000.0


def _refractivity_excess(z_m: np.ndarray) -> np.ndarray:
    """n - 1 in the International Standard Atmosphere (troposphere).

    N = 77.6 P / T (P in hPa, T in K): the dry term of the Smith-Weintraub
    formula, which carries all but a few percent of optical refractivity.
    """
    z = np.clip(np.asarray(z_m, dtype=float), -500.0, 11000.0)
    temperature = 288.15 - 0.0065 * z
    pressure = 1013.25 * (temperature / 288.15) ** 5.25588
    return 77.6 * pressure / temperature * 1e-6


@lru_cache(maxsize=4096)
def _refraction_constant_cached(camera_m: float, ground_m: float) -> float:
    if camera_m <= ground_m + 1.0:
        return 0.0
    heights = np.linspace(ground_m, camera_m, 401)
    mean_excess = float(np.trapz(_refractivity_excess(heights), heights)) / (camera_m - ground_m)
    return mean_excess - float(_refractivity_excess(camera_m))


def refraction_constant(camera_height_m: float, ground_height_m: float) -> float:
    """K in radians: the refraction angle is K tan(beta).

    Heights above sea level, in metres. Rounded to the metre for caching.
    """
    return _refraction_constant_cached(round(float(camera_height_m)), round(float(ground_height_m)))


def radial_displacement(r_mm: np.ndarray, focal_mm: float, camera_height_m: float,
                        ground_height_m, refraction: bool, curvature: bool,
                        earth_radius_m: float = EARTH_RADIUS_M) -> np.ndarray:
    """Outward radial displacement of image points (mm) caused by the two
    effects, at image radius ``r_mm``. Negative means inward."""
    r = np.asarray(r_mm, dtype=float)
    ground = np.asarray(ground_height_m, dtype=float)
    shift = np.zeros(np.broadcast(r, ground).shape)
    if refraction:
        if ground.ndim == 0:
            k = refraction_constant(camera_height_m, float(ground))
        else:
            # Per-point ground heights: K varies slowly, so evaluate it on the
            # distinct metre-rounded heights only.
            rounded = np.round(ground).astype(np.int64)
            unique, inverse = np.unique(rounded, return_inverse=True)
            k = np.array([refraction_constant(camera_height_m, float(u)) for u in unique])[inverse]
            k = k.reshape(ground.shape)
        shift = shift + k * (r + r ** 3 / focal_mm ** 2)
    if curvature:
        above_ground = np.maximum(camera_height_m - ground, 0.0)
        shift = shift - r ** 3 * above_ground / (2.0 * earth_radius_m * focal_mm ** 2)
    return shift


def to_flat(film_mm: np.ndarray, focal_mm: float, camera_height_m: float,
            ground_height_m, refraction: bool, curvature: bool,
            earth_radius_m: float = EARTH_RADIUS_M) -> np.ndarray:
    """Measured image coordinates -> those of a straight ray over flat ground."""
    film = np.atleast_2d(np.asarray(film_mm, dtype=float))
    if not (refraction or curvature):
        return film.copy()
    r = np.hypot(film[:, 0], film[:, 1])
    shift = radial_displacement(r, focal_mm, camera_height_m, ground_height_m,
                                refraction, curvature, earth_radius_m)
    scale = np.where(r > 1e-12, 1.0 - shift / np.maximum(r, 1e-12), 1.0)
    return film * scale[:, None]


def from_flat(film_mm: np.ndarray, focal_mm: float, camera_height_m: float,
              ground_height_m, refraction: bool, curvature: bool,
              earth_radius_m: float = EARTH_RADIUS_M, iterations: int = 6) -> np.ndarray:
    """Inverse of ``to_flat``: where a flat-model image point was really
    recorded. The displacement is a tiny perturbation, so a few fixed-point
    steps converge far below a micrometre."""
    target = np.atleast_2d(np.asarray(film_mm, dtype=float))
    if not (refraction or curvature):
        return target.copy()
    guess = target.copy()
    for _ in range(iterations):
        error = to_flat(guess, focal_mm, camera_height_m, ground_height_m,
                        refraction, curvature, earth_radius_m) - target
        guess = guess - error
        if np.nanmax(np.abs(error)) < 1e-9:
            break
    return guess
