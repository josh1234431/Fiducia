"""Least-squares matching recovers known subpixel geometry.

A textured patch is resampled through a known shift and affine, with a
brightness change and noise, and matched back. The recovered position must
be within a few hundredths of a pixel, and the reported standard deviation
must be an honest measure of the actual error.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))

from fiducia.tiepoints import _Window, least_squares_match  # noqa: E402

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  - {detail}" if detail else ""))


class Array:
    """Enough of a raster dataset for _Window."""

    def __init__(self, data):
        self.data = data
        self.width, self.height = data.shape[1], data.shape[0]

    def read(self, band, window):
        r0, c0 = int(window.row_off), int(window.col_off)
        return self.data[r0:r0 + int(window.height), c0:c0 + int(window.width)]


rng = np.random.default_rng(0)
texture = gaussian_filter(rng.normal(0, 1, (400, 400)), 2.0) * 60 + 120

errors, sigmas = [], []
for trial in range(40):
    shift = rng.uniform(-0.5, 0.5, 2)
    affine = np.eye(2) + rng.normal(0, 0.03, (2, 2))
    gain, offset = rng.uniform(0.8, 1.2), rng.uniform(-15, 15)
    # right(x) = gain * left(A^-1 (x - centre - shift) + centre) + offset
    rows, cols = np.mgrid[0:400, 0:400].astype(float)
    centre = np.array([200.0, 200.0])
    inverse = np.linalg.inv(affine)
    dc, dr = cols - centre[0] - shift[0], rows - centre[1] - shift[1]
    src_c = inverse[0, 0] * dc + inverse[0, 1] * dr + centre[0]
    src_r = inverse[1, 0] * dc + inverse[1, 1] * dr + centre[1]
    right = gain * map_coordinates(texture, [src_r, src_c], order=3) + offset
    right += rng.normal(0, 2.0, right.shape)
    left = texture + rng.normal(0, 2.0, texture.shape)

    half = 15
    template = left[200 - half:200 + half + 1, 200 - half:200 + half + 1].astype(np.float32)
    window = _Window(Array(right.astype(np.float32)), 1, 150, 150, 251, 251)
    start = (200.0 + np.round(shift[0]), 200.0 + np.round(shift[1]))   # whole-pixel start
    fit = least_squares_match(template, window, start, affine)
    if fit is None:
        errors.append(np.nan)
        continue
    col, row, sigma, correlation = fit
    errors.append(np.hypot(col - (200 + shift[0]), row - (200 + shift[1])))
    sigmas.append(sigma)

errors = np.array(errors)
ok = np.isfinite(errors)
check("converges on every trial", ok.all(), f"{ok.sum()} of {len(errors)}")
rms = float(np.sqrt(np.mean(errors[ok] ** 2)))
check("position error below 0.05 px RMS", rms < 0.05, f"{rms:.4f} px")
ratio = rms / float(np.sqrt(np.mean(np.square(sigmas))))
check("reported sigma honest to a factor of 2", 0.5 < ratio < 2.0, f"actual / reported = {ratio:.2f}")

print(f"\n{'=' * 64}\n  TOTAL {len(PASS)} passed, {len(FAIL)} failed\n{'=' * 64}")
if FAIL:
    print("  FAILED:", ", ".join(FAIL))
    sys.exit(1)
