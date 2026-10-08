"""Automatic fiducial mark detection.

Measure the eight fiducials on one photograph by hand and every other frame
from the same camera and scanner follows automatically. On a block of thirty
scanned aerials that is two hundred and forty clicks the operator does not
make, and the machine places them more repeatably than a hand does.

Two detection modes:

``template``
    Chips cut from a photo the operator has already measured, matched into the
    remaining frames. This is the accurate path and the one to prefer -- the
    template is the actual fiducial of the actual camera, scanned on the actual
    scanner, so correlation is near-perfect.

``synthetic``
    A generated cross, for the very first photo when there is nothing to copy
    from. Less certain, so results come back with scores and the interface
    presents them as proposals rather than measurements.

Both paths finish the same way: fit the interior orientation, re-predict every
mark from that fit, and search again in a tight window. One refinement pass
converts a rough first guess into a sub-pixel result, and the residual of the
fit is the honest report on whether it worked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from .camera import FIDUCIAL_SLOTS, CameraModel, fit_fiducial_transform

__all__ = [
    "FiducialCandidate",
    "DetectionOptions",
    "DetectionResult",
    "extract_templates",
    "detect_fiducials",
]


@dataclass
class FiducialCandidate:
    slot: str
    col: float
    row: float
    score: float
    residual_px: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "slot": self.slot,
            "col": float(self.col),
            "row": float(self.row),
            "score": float(self.score),
            "residualPx": None if self.residual_px is None else float(self.residual_px),
        }


@dataclass
class DetectionOptions:
    mode: str = "template"          # template | synthetic
    chip_px: int = 96               # template size cut from the measured photo
    search_px: int = 0              # 0 = derive from image size
    min_score: float = 0.45
    refine: bool = True
    max_residual_px: float = 4.0    # marks worse than this are dropped on refit


@dataclass
class DetectionResult:
    candidates: list[FiducialCandidate]
    fit: Optional[dict]
    rms_px: Optional[float]
    accepted: int
    rejected: list[dict] = field(default_factory=list)
    mode: str = "template"
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "fit": self.fit,
            "rmsPx": None if self.rms_px is None else float(self.rms_px),
            "accepted": self.accepted,
            "rejected": self.rejected,
            "mode": self.mode,
            "message": self.message,
        }


def _read_window(dataset, col: float, row: float, half: int) -> tuple[np.ndarray, int, int]:
    """Read a full-resolution square around a point, clamped to the image."""
    from rasterio.windows import Window

    c0 = int(round(col)) - half
    r0 = int(round(row)) - half
    size = half * 2 + 1

    band = 1 if dataset.count < 3 else 2
    data = dataset.read(
        band, window=Window(c0, r0, size, size), boundless=True, fill_value=0
    ).astype(np.float32)
    return data, c0, r0


def _normalise(patch: np.ndarray) -> np.ndarray:
    finite = patch[np.isfinite(patch)]
    if finite.size == 0:
        return np.zeros_like(patch, dtype=np.uint8)
    lo, hi = np.percentile(finite, [1, 99])
    span = max(float(hi - lo), 1e-6)
    return np.clip((patch - lo) / span * 255.0, 0, 255).astype(np.uint8)


def extract_templates(
    image_path: str,
    measured: dict[str, Sequence[float]],
    chip_px: int = 96,
) -> dict[str, list]:
    """Cut template chips from a photo whose fiducials are already measured."""
    import rasterio

    half = max(12, chip_px // 2)
    chips: dict[str, list] = {}

    with rasterio.open(image_path) as dataset:
        for slot, position in measured.items():
            if slot not in FIDUCIAL_SLOTS:
                continue
            patch, _, _ = _read_window(dataset, float(position[0]), float(position[1]), half)
            if patch.size == 0 or float(patch.std()) < 3.0:
                continue
            chips[slot] = _normalise(patch).tolist()

    return chips


def _synthetic_cross(size: int) -> np.ndarray:
    """A generated fiducial: a thin cross, mid-grey ground, bright arms.

    Matched with a contrast-invariant score and tried at both polarities, so
    it finds a dark cross on light film as readily as the reverse.
    """
    size = size | 1
    chip = np.full((size, size), 128, dtype=np.uint8)
    centre = size // 2
    arm = max(2, size // 10)
    reach = int(size * 0.40)

    chip[centre - arm: centre + arm + 1, centre - reach: centre + reach + 1] = 245
    chip[centre - reach: centre + reach + 1, centre - arm: centre + arm + 1] = 245
    return chip


def _predict_from_geometry(
    camera: CameraModel, width: int, height: int
) -> dict[str, tuple[float, float]]:
    """Rough pixel positions for each calibrated mark, with no fit available.

    Assumes the fiducial bounding box occupies most of the scan and that the
    scan is roughly square to the film -- true of any normal scan, and only
    needs to be close enough to put the mark inside the search window.
    """
    calibrated = camera.fiducials_mm
    if len(calibrated) < 3:
        return {}

    xs = [v[0] for v in calibrated.values()]
    ys = [v[1] for v in calibrated.values()]
    span_x = max(max(xs) - min(xs), 1e-6)
    span_y = max(max(ys) - min(ys), 1e-6)
    centre_x = (max(xs) + min(xs)) / 2.0
    centre_y = (max(ys) + min(ys)) / 2.0

    # The marks sit a little inside the scanned area.
    scale_x = (width * 0.90) / span_x
    scale_y = (height * 0.90) / span_y

    return {
        slot: (
            width / 2.0 + (mm[0] - centre_x) * scale_x,
            height / 2.0 - (mm[1] - centre_y) * scale_y,
        )
        for slot, mm in calibrated.items()
    }


def _predict_from_fit(fit, calibrated: dict) -> dict[str, tuple[float, float]]:
    """Exact pixel positions implied by a solved interior orientation."""
    bx, by = np.asarray(fit.bx), np.asarray(fit.by)
    return {
        slot: (
            float(bx[0] + bx[1] * mm[0] + bx[2] * mm[1]),
            float(by[0] + by[1] * mm[0] + by[2] * mm[1]),
        )
        for slot, mm in calibrated.items()
    }


def _match_one(
    dataset,
    template: np.ndarray,
    predicted: tuple[float, float],
    search_px: int,
) -> Optional[tuple[float, float, float]]:
    """Correlate one template into one search window. Returns (col, row, score)."""
    import cv2

    half_template = template.shape[0] // 2
    half_search = half_template + max(8, search_px)

    window, c0, r0 = _read_window(dataset, predicted[0], predicted[1], half_search)
    if window.shape[0] < template.shape[0] or window.shape[1] < template.shape[1]:
        return None

    haystack = _normalise(window)
    if float(haystack.std()) < 2.0:
        return None

    best = None
    # Try both polarities: film emulsion can be positive or negative, and a
    # fiducial is dark-on-light as often as the reverse.
    for needle in (template, 255 - template):
        response = cv2.matchTemplate(haystack, needle, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(response)
        if best is None or score > best[0]:
            best = (score, location, response)

    score, (mx, my), response = best

    # Parabolic subpixel refinement on the correlation surface.
    dx = dy = 0.0
    if 0 < mx < response.shape[1] - 1:
        a, b, c = response[my, mx - 1], response[my, mx], response[my, mx + 1]
        denom = a - 2 * b + c
        if abs(denom) > 1e-9:
            dx = 0.5 * (a - c) / denom
    if 0 < my < response.shape[0] - 1:
        a, b, c = response[my - 1, mx], response[my, mx], response[my + 1, mx]
        denom = a - 2 * b + c
        if abs(denom) > 1e-9:
            dy = 0.5 * (a - c) / denom

    return (
        c0 + mx + dx + half_template,
        r0 + my + dy + half_template,
        float(score),
    )


def detect_fiducials(
    image_path: str,
    camera: CameraModel,
    templates: Optional[dict[str, list]] = None,
    options: Optional[DetectionOptions] = None,
    progress: Optional[Callable[[float, str], None]] = None,
) -> DetectionResult:
    """Find every calibrated fiducial mark on one photograph."""
    import rasterio

    options = options or DetectionOptions()
    calibrated = camera.fiducials_mm

    if len(calibrated) < 3:
        return DetectionResult(
            [], None, None, 0, mode=options.mode,
            message="The camera has fewer than three calibrated fiducials. "
                    "Enter them in the Camera step first.",
        )

    mode = "template" if templates else "synthetic"

    with rasterio.open(image_path) as dataset:
        width, height = dataset.width, dataset.height
        search_px = options.search_px or max(40, int(min(width, height) * 0.045))

        # Prepare one template per slot.
        chips: dict[str, np.ndarray] = {}
        if templates:
            for slot, chip in templates.items():
                array = np.asarray(chip, dtype=np.uint8)
                if array.ndim == 2 and array.shape[0] >= 9:
                    chips[slot] = array
        if not chips:
            synthetic = _synthetic_cross(max(31, options.chip_px // 2 | 1))
            chips = {slot: synthetic for slot in calibrated}

        predicted = _predict_from_geometry(camera, width, height)
        if not predicted:
            return DetectionResult([], None, None, 0, mode=mode,
                                   message="Could not predict where the marks should be")

        # -- pass one: coarse search from the geometric prediction ---------
        found: dict[str, FiducialCandidate] = {}
        slots = [s for s in FIDUCIAL_SLOTS if s in calibrated and s in chips]

        for index, slot in enumerate(slots):
            if progress:
                progress(0.1 + 0.45 * index / max(len(slots), 1), f"Searching {slot}")
            hit = _match_one(dataset, chips[slot], predicted[slot], search_px)
            if hit and hit[2] >= options.min_score:
                found[slot] = FiducialCandidate(slot, hit[0], hit[1], hit[2])

        if len(found) < 3:
            return DetectionResult(
                list(found.values()), None, None, len(found), mode=mode,
                message=f"Only {len(found)} of {len(slots)} marks matched confidently. "
                        "Measure this photo by hand, then use it as the template "
                        "for the rest of the block.",
            )

        # -- pass two: refit, re-predict, search a tight window -------------
        fit = None
        rejected: list[dict] = []

        for attempt in range(2 if options.refine else 1):
            try:
                fit = fit_fiducial_transform(
                    {s: (c.col, c.row) for s, c in found.items()},
                    {s: tuple(calibrated[s]) for s in found},
                )
            except ValueError as exc:
                return DetectionResult(
                    list(found.values()), None, None, len(found), mode=mode,
                    message=f"Interior orientation could not be fitted: {exc}",
                )

            for slot, residual in zip(fit.used, fit.residuals_px):
                if slot in found:
                    found[slot].residual_px = float(residual)

            if attempt == 0 and options.refine:
                if progress:
                    progress(0.65, "Refining against the fitted orientation")
                tight = _predict_from_fit(fit, calibrated)
                for slot in slots:
                    hit = _match_one(dataset, chips[slot], tight[slot], max(12, search_px // 4))
                    if hit and hit[2] >= options.min_score:
                        found[slot] = FiducialCandidate(slot, hit[0], hit[1], hit[2])

        # -- drop marks the fit says are wrong ------------------------------
        if fit is not None and len(found) > 3:
            for slot, residual in zip(list(fit.used), list(fit.residuals_px)):
                if residual > options.max_residual_px and len(found) > 3:
                    rejected.append({
                        "slot": slot,
                        "residualPx": float(residual),
                        "reason": f"Residual of {residual:.1f} px after fitting; "
                                  "most likely matched the wrong feature.",
                    })
                    found.pop(slot, None)

            if rejected:
                try:
                    fit = fit_fiducial_transform(
                        {s: (c.col, c.row) for s, c in found.items()},
                        {s: tuple(calibrated[s]) for s in found},
                    )
                    for slot, residual in zip(fit.used, fit.residuals_px):
                        if slot in found:
                            found[slot].residual_px = float(residual)
                except ValueError:
                    pass

    if progress:
        progress(1.0, f"Placed {len(found)} marks")

    ordered = [found[s] for s in FIDUCIAL_SLOTS if s in found]
    rms = fit.rms_px if fit else None

    if rms is not None and rms <= 2.0:
        message = f"{len(ordered)} marks placed, RMS {rms:.2f} px."
    elif rms is not None:
        message = (
            f"{len(ordered)} marks placed, but RMS is {rms:.2f} px. "
            "Check each mark before accepting."
        )
    else:
        message = f"{len(ordered)} marks placed."

    return DetectionResult(
        candidates=ordered,
        fit=fit.to_dict() if fit else None,
        rms_px=rms,
        accepted=len(ordered),
        rejected=rejected,
        mode=mode,
        message=message,
    )
