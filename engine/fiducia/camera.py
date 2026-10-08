"""Interior orientation: the pixel <-> focal-plane relationship.

Two camera types are supported, following the standard aerial math model:

``film``
    Scanned analogue photography. The mapping from scan pixels to calibrated
    film millimetres is recovered by least squares from measured fiducial
    marks, then lens distortion and principal-point offset are removed.

``digital``
    Frame digital / UAV sensors. There are no fiducials -- the mapping is fixed
    by the chip's pixel pitch and the principal point, so interior orientation
    is exact by construction and needs no measurement.

Distortion is the odd-power radial polynomial conventionally written R1/R3/R5/R7
(labelled K0..K3), optionally with Conrady-Brown decentering terms P1/P2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional

import numpy as np

__all__ = [
    "FiducialFit",
    "CameraModel",
    "fit_fiducial_transform",
    "fit_radial_from_table",
]

FIDUCIAL_SLOTS = (
    "top_left",
    "top_middle",
    "top_right",
    "right_middle",
    "bottom_right",
    "bottom_middle",
    "bottom_left",
    "left_middle",
)


@dataclass
class FiducialFit:
    """Result of fitting scan pixels to calibrated film millimetres."""

    # Forward coefficients: film_mm = A @ [1, col, row]
    ax: np.ndarray
    ay: np.ndarray
    # Inverse coefficients: pixel = B @ [1, x_mm, y_mm]
    bx: np.ndarray
    by: np.ndarray
    residuals_px: np.ndarray          # per-fiducial radial residual, in pixels
    rms_px: float
    max_px: float
    used: list[str]
    transform_kind: Literal["affine", "similarity"]

    def to_dict(self) -> dict:
        return {
            "ax": self.ax.tolist(),
            "ay": self.ay.tolist(),
            "bx": self.bx.tolist(),
            "by": self.by.tolist(),
            "residualsPx": self.residuals_px.tolist(),
            "rmsPx": float(self.rms_px),
            "maxPx": float(self.max_px),
            "used": list(self.used),
            "transformKind": self.transform_kind,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FiducialFit":
        return cls(
            ax=np.asarray(d["ax"], dtype=float),
            ay=np.asarray(d["ay"], dtype=float),
            bx=np.asarray(d["bx"], dtype=float),
            by=np.asarray(d["by"], dtype=float),
            residuals_px=np.asarray(d.get("residualsPx", []), dtype=float),
            rms_px=float(d.get("rmsPx", 0.0)),
            max_px=float(d.get("maxPx", 0.0)),
            used=list(d.get("used", [])),
            transform_kind=d.get("transformKind", "affine"),
        )


FIDUCIAL_TRANSFORMS = ("auto", "conformal", "affine")


def fit_fiducial_transform(
    measured: dict[str, tuple[float, float]],
    calibrated: dict[str, tuple[float, float]],
    kind: str = "auto",
) -> FiducialFit:
    """Least-squares fit of scan pixel coordinates to calibrated film mm.

    ``measured`` maps a fiducial slot name to the pixel (column, row) the
    operator clicked; ``calibrated`` maps the same slot to the (x, y)
    millimetres printed on the calibration certificate.

    With four or more common fiducials a full six-parameter affine is solved,
    which absorbs the differential scale and non-orthogonality a flatbed
    scanner introduces. With three, an affine is still exactly determined but
    has no redundancy, so a four-parameter similarity is used instead -- it
    cannot absorb scanner skew but it also cannot silently hide a mis-clicked
    mark in a zero residual.

    ``kind`` chooses explicitly: ``conformal`` (four parameters: shift,
    rotation, one scale) for a scanner known to be square and orthogonal,
    ``affine`` (six: also differential scale and skew), or ``auto`` as above.
    """
    slots = [s for s in FIDUCIAL_SLOTS if s in measured and s in calibrated]
    if len(slots) < 3:
        raise ValueError(
            f"Interior orientation needs at least 3 fiducial marks, got {len(slots)}"
        )

    px = np.array([measured[s] for s in slots], dtype=float)
    mm = np.array([calibrated[s] for s in slots], dtype=float)
    n = len(slots)
    if kind == "affine" and n < 4:
        raise ValueError("An affine interior orientation needs at least 4 fiducial marks")

    if (kind == "affine") or (kind == "auto" and n >= 4):
        kind = "affine"
        design = np.column_stack([np.ones(n), px[:, 0], px[:, 1]])
        ax, *_ = np.linalg.lstsq(design, mm[:, 0], rcond=None)
        ay, *_ = np.linalg.lstsq(design, mm[:, 1], rcond=None)
    else:
        kind = "similarity"
        # [x; y] = [a -b; b a][col; row] + [c; d]
        rows = np.zeros((2 * n, 4))
        rhs = np.zeros(2 * n)
        rows[0::2, 0] = px[:, 0]
        rows[0::2, 1] = -px[:, 1]
        rows[0::2, 2] = 1.0
        rows[1::2, 0] = px[:, 1]
        rows[1::2, 1] = px[:, 0]
        rows[1::2, 3] = 1.0
        rhs[0::2] = mm[:, 0]
        rhs[1::2] = mm[:, 1]
        sol, *_ = np.linalg.lstsq(rows, rhs, rcond=None)
        a, b, c, d = sol
        ax = np.array([c, a, -b])
        ay = np.array([d, b, a])

    predicted_mm = np.column_stack(
        [
            ax[0] + ax[1] * px[:, 0] + ax[2] * px[:, 1],
            ay[0] + ay[1] * px[:, 0] + ay[2] * px[:, 1],
        ]
    )

    # Invert the forward map so residuals can be reported in pixels, which is
    # the unit the operator is actually clicking in.
    fwd = np.array([[ax[1], ax[2]], [ay[1], ay[2]]])
    det = np.linalg.det(fwd)
    if abs(det) < 1e-12:
        raise ValueError("Fiducial marks are collinear or coincident")
    inv = np.linalg.inv(fwd)
    offset = -inv @ np.array([ax[0], ay[0]])
    bx = np.array([offset[0], inv[0, 0], inv[0, 1]])
    by = np.array([offset[1], inv[1, 0], inv[1, 1]])

    back_px = np.column_stack(
        [
            bx[0] + bx[1] * predicted_mm[:, 0] + bx[2] * predicted_mm[:, 1],
            by[0] + by[1] * predicted_mm[:, 0] + by[2] * predicted_mm[:, 1],
        ]
    )
    # predicted_mm round-trips exactly; residual is measured vs the mm the
    # certificate says, expressed back in pixels through the same inverse.
    target_px = np.column_stack(
        [
            bx[0] + bx[1] * mm[:, 0] + bx[2] * mm[:, 1],
            by[0] + by[1] * mm[:, 0] + by[2] * mm[:, 1],
        ]
    )
    diff = px - target_px
    residuals = np.hypot(diff[:, 0], diff[:, 1])

    return FiducialFit(
        ax=ax,
        ay=ay,
        bx=bx,
        by=by,
        residuals_px=residuals,
        rms_px=float(np.sqrt(np.mean(residuals**2))),
        max_px=float(np.max(residuals)),
        used=slots,
        transform_kind=kind,
    )


def fit_radial_from_table(
    radii_mm: Iterable[float],
    distortion_um: Iterable[float],
    distance_units: str = "mm",
    distortion_units: str = "um",
) -> dict:
    """Fit K0..K3 to a calibration certificate's radial distortion table.

    Certificates report mean radial distortion at a series of radial distances
    rather than the polynomial coefficients themselves, so the coefficients are
    computed from the table; the model fitted is

        dr = K0*r + K1*r^3 + K2*r^5 + K3*r^7

    with ``r`` in millimetres and ``dr`` in millimetres. Certificates almost
    always tabulate the distortion in micrometres, hence the unit conversion.

    Returns the coefficients plus fit diagnostics, so the UI can show the
    operator how well the polynomial actually reproduces their table -- a large
    residual here usually means a transcription error, caught in seconds
    instead of surfacing later as an inexplicable GCP residual.
    """
    r = np.asarray(list(radii_mm), dtype=float)
    d = np.asarray(list(distortion_um), dtype=float)
    if r.size != d.size:
        raise ValueError("Radial distance and distortion tables differ in length")
    if r.size < 2:
        raise ValueError("Need at least two table entries to fit a distortion curve")

    if distance_units == "um":
        r = r / 1000.0
    elif distance_units == "cm":
        r = r * 10.0
    if distortion_units == "um":
        d = d / 1000.0
    elif distortion_units == "nm":
        d = d / 1e6

    keep = r > 1e-9
    r, d = r[keep], d[keep]

    # Fit as many odd powers as the table can support without going singular.
    n_terms = int(min(4, max(1, r.size - 1)))
    design = np.column_stack([r ** (2 * i + 1) for i in range(n_terms)])
    coeffs, *_ = np.linalg.lstsq(design, d, rcond=None)

    k = np.zeros(4)
    k[:n_terms] = coeffs
    fitted = design @ coeffs
    residual_um = (d - fitted) * 1000.0

    return {
        "k0": float(k[0]),
        "k1": float(k[1]),
        "k2": float(k[2]),
        "k3": float(k[3]),
        "termsFitted": n_terms,
        "rmsUm": float(np.sqrt(np.mean(residual_um**2))),
        "maxUm": float(np.max(np.abs(residual_um))) if residual_um.size else 0.0,
        "samples": [
            {"radiusMm": float(rr), "observedUm": float(dd * 1000.0),
             "fittedUm": float(ff * 1000.0), "residualUm": float(res)}
            for rr, dd, ff, res in zip(r, d, fitted, residual_um)
        ],
    }


@dataclass
class CameraModel:
    """A complete interior orientation.

    Distances are millimetres throughout, matching calibration certificates.
    """

    kind: Literal["film", "digital"] = "film"
    name: str = ""
    focal_mm: float = 0.0

    # Principal point offset from the fiducial centre (film) or chip centre
    # (digital). A certificate's FL/PPA/PPS form is reduced to a single PPO on
    # input, since PPO = PPA + PPS.
    ppo_x_mm: float = 0.0
    ppo_y_mm: float = 0.0

    # Odd-power radial polynomial: dr = k0*r + k1*r^3 + k2*r^5 + k3*r^7
    k0: float = 0.0
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    # Conrady-Brown decentering. Rarely reported for aerial cameras; usually 0.
    p1: float = 0.0
    p2: float = 0.0

    # Film only.
    fiducials_mm: dict = field(default_factory=dict)
    fiducial_position: Literal["edge", "corner", "edge_and_corner"] = "edge_and_corner"
    fiducial_transform: str = "auto"    # FIDUCIAL_TRANSFORMS
    image_scale: float = 0.0

    # Digital only.
    pixel_pitch_mm: float = 0.0         # chip size / pixel spacing
    columns: int = 0
    rows: int = 0
    long_track_offset_mm: float = 0.0   # along the flying direction
    cross_track_offset_mm: float = 0.0
    # The certificate's principal point for each delivered rotation, clockwise
    # degrees ("0", "90", "180", "270") -> (x, y) mm. Kept so turning the
    # camera to match rotated images takes the offset from the certificate.
    ppo_rotations: dict = field(default_factory=dict)

    apply_atmospheric: bool = False
    apply_earth_curvature: bool = False
    earth_radius_m: float = 6371000.0

    # Corrections found by self-calibration in the bundle adjustment, on top
    # of the certificate: {"set", "radiusMm", "values": {name: value},
    # "sigmas": {...}}. See bundle.apply_additional_parameters. Applied in
    # pixel_to_film and film_to_pixel, so every product uses the camera the
    # block was solved with.
    adjustment: Optional[dict] = None

    def without_adjustment(self) -> "CameraModel":
        from dataclasses import replace

        return replace(self, adjustment=None)

    def _adjustment_delta(self, film: np.ndarray) -> np.ndarray:
        from .bundle import apply_additional_parameters

        adj = self.adjustment or {}
        return apply_additional_parameters(film, adj.get("values") or {}, self.focal_mm,
                                           float(adj.get("radiusMm") or 0.0))

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "name": self.name,
            "focalMm": self.focal_mm,
            "ppoXMm": self.ppo_x_mm,
            "ppoYMm": self.ppo_y_mm,
            "k0": self.k0, "k1": self.k1, "k2": self.k2, "k3": self.k3,
            "p1": self.p1, "p2": self.p2,
            "fiducialsMm": {k: list(v) for k, v in self.fiducials_mm.items()},
            "fiducialPosition": self.fiducial_position,
            "fiducialTransform": self.fiducial_transform,
            "imageScale": self.image_scale,
            "pixelPitchMm": self.pixel_pitch_mm,
            "columns": self.columns,
            "rows": self.rows,
            "longTrackOffsetMm": self.long_track_offset_mm,
            "crossTrackOffsetMm": self.cross_track_offset_mm,
            "ppoRotations": {k: list(v) for k, v in self.ppo_rotations.items()},
            "applyAtmospheric": self.apply_atmospheric,
            "applyEarthCurvature": self.apply_earth_curvature,
            "earthRadiusM": self.earth_radius_m,
            "adjustment": self.adjustment,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CameraModel":
        return cls(
            kind=d.get("kind", "film"),
            name=d.get("name", ""),
            focal_mm=float(d.get("focalMm") or 0.0),
            ppo_x_mm=float(d.get("ppoXMm") or 0.0),
            ppo_y_mm=float(d.get("ppoYMm") or 0.0),
            k0=float(d.get("k0") or 0.0),
            k1=float(d.get("k1") or 0.0),
            k2=float(d.get("k2") or 0.0),
            k3=float(d.get("k3") or 0.0),
            p1=float(d.get("p1") or 0.0),
            p2=float(d.get("p2") or 0.0),
            fiducials_mm={k: tuple(v) for k, v in (d.get("fiducialsMm") or {}).items()},
            fiducial_position=d.get("fiducialPosition", "edge_and_corner"),
            fiducial_transform=(d.get("fiducialTransform")
                                if d.get("fiducialTransform") in FIDUCIAL_TRANSFORMS else "auto"),
            image_scale=float(d.get("imageScale") or 0.0),
            pixel_pitch_mm=float(d.get("pixelPitchMm") or 0.0),
            columns=int(d.get("columns") or 0),
            rows=int(d.get("rows") or 0),
            long_track_offset_mm=float(d.get("longTrackOffsetMm") or 0.0),
            cross_track_offset_mm=float(d.get("crossTrackOffsetMm") or 0.0),
            ppo_rotations={
                str(k): (float(v[0]), float(v[1]))
                for k, v in (d.get("ppoRotations") or {}).items()
                if str(k) in ("0", "90", "180", "270") and v is not None and len(v) == 2
            },
            apply_atmospheric=bool(d.get("applyAtmospheric", False)),
            apply_earth_curvature=bool(d.get("applyEarthCurvature", False)),
            earth_radius_m=float(d.get("earthRadiusM") or 6371000.0),
            adjustment=d.get("adjustment") if isinstance(d.get("adjustment"), dict)
            and (d["adjustment"].get("values") or {}) else None,
        )

    # -- distortion ------------------------------------------------------

    def _radial_delta(self, r: np.ndarray) -> np.ndarray:
        r2 = r * r
        return r * (self.k0 + r2 * (self.k1 + r2 * (self.k2 + r2 * self.k3)))

    def remove_distortion(self, xy: np.ndarray) -> np.ndarray:
        """Observed principal-point-relative mm -> ideal (distortion-free) mm.

        This is the direction used when reducing a measured image point for
        adjustment.
        """
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        x, y = xy[:, 0], xy[:, 1]
        r = np.hypot(x, y)
        safe = np.where(r < 1e-12, 1.0, r)
        scale = self._radial_delta(r) / safe

        dx = x * scale
        dy = y * scale
        if self.p1 or self.p2:
            r2 = r * r
            dx = dx + self.p1 * (r2 + 2 * x * x) + 2 * self.p2 * x * y
            dy = dy + self.p2 * (r2 + 2 * y * y) + 2 * self.p1 * x * y

        return np.column_stack([x - dx, y - dy])

    def apply_distortion(self, xy: np.ndarray, iterations: int = 8) -> np.ndarray:
        """Ideal mm -> observed mm, by fixed-point inversion of the above.

        This is the direction the ortho resampler needs: it has an ideal
        focal-plane position from collinearity and must find where that point
        actually landed on the film. The radial polynomial has no closed-form
        inverse, but it is a very small perturbation, so a handful of
        fixed-point iterations converge to well below a micrometre.
        """
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        if not (self.k0 or self.k1 or self.k2 or self.k3 or self.p1 or self.p2):
            return xy.copy()

        guess = xy.copy()
        for _ in range(iterations):
            corrected = self.remove_distortion(guess)
            error = corrected - xy
            guess = guess - error
            if np.nanmax(np.abs(error)) < 1e-9:
                break
        return guess

    # -- interior orientation -------------------------------------------

    def pixel_to_film(self, pixels: np.ndarray, fit: Optional[FiducialFit] = None) -> np.ndarray:
        """Image (column, row) -> ideal focal-plane mm, principal point at origin.

        For film this needs the per-image :class:`FiducialFit`, because every
        scan of every photo has its own pixel-to-film relationship.
        """
        pixels = np.atleast_2d(np.asarray(pixels, dtype=float))

        if self.kind == "film":
            if fit is None:
                raise ValueError("Film cameras need a fiducial fit for interior orientation")
            col, row = pixels[:, 0], pixels[:, 1]
            mm = np.column_stack(
                [
                    fit.ax[0] + fit.ax[1] * col + fit.ax[2] * row,
                    fit.ay[0] + fit.ay[1] * col + fit.ay[2] * row,
                ]
            )
        else:
            if self.pixel_pitch_mm <= 0:
                raise ValueError("Digital cameras need a non-zero pixel pitch")
            cx = (self.columns - 1) / 2.0
            cy = (self.rows - 1) / 2.0
            mm = np.column_stack(
                [
                    (pixels[:, 0] - cx) * self.pixel_pitch_mm,
                    # Row index grows downwards; film y grows upwards.
                    -(pixels[:, 1] - cy) * self.pixel_pitch_mm,
                ]
            )
            mm[:, 0] -= self.cross_track_offset_mm
            mm[:, 1] -= self.long_track_offset_mm

        mm[:, 0] -= self.ppo_x_mm
        mm[:, 1] -= self.ppo_y_mm
        ideal = self.remove_distortion(mm)
        if self.adjustment:
            # measured = ideal + delta(ideal): solved by fixed point, the
            # correction being a small, smooth perturbation.
            target = ideal.copy()
            for _ in range(6):
                ideal = target - self._adjustment_delta(ideal)
        return ideal

    def film_to_pixel(self, film: np.ndarray, fit: Optional[FiducialFit] = None) -> np.ndarray:
        """Ideal focal-plane mm -> image (column, row). Inverse of the above."""
        film = np.atleast_2d(np.asarray(film, dtype=float))
        if self.adjustment:
            film = film + self._adjustment_delta(film)
        observed = self.apply_distortion(film)
        observed = observed + np.array([self.ppo_x_mm, self.ppo_y_mm])

        if self.kind == "film":
            if fit is None:
                raise ValueError("Film cameras need a fiducial fit for interior orientation")
            x, y = observed[:, 0], observed[:, 1]
            return np.column_stack(
                [
                    fit.bx[0] + fit.bx[1] * x + fit.bx[2] * y,
                    fit.by[0] + fit.by[1] * x + fit.by[2] * y,
                ]
            )

        observed[:, 0] += self.cross_track_offset_mm
        observed[:, 1] += self.long_track_offset_mm
        cx = (self.columns - 1) / 2.0
        cy = (self.rows - 1) / 2.0
        return np.column_stack(
            [
                observed[:, 0] / self.pixel_pitch_mm + cx,
                -observed[:, 1] / self.pixel_pitch_mm + cy,
            ]
        )

    def pixel_scale_mm(self, fit: Optional[FiducialFit] = None) -> float:
        """Approximate millimetres per pixel, for converting residual units."""
        if self.kind == "digital":
            return self.pixel_pitch_mm or 1.0
        if fit is None:
            return 1.0
        return float(np.sqrt(abs(fit.ax[1] * fit.ay[2] - fit.ax[2] * fit.ay[1])))

    def ground_sample_distance(self, fit: Optional[FiducialFit] = None) -> float:
        """Nominal metres on the ground per image pixel, from the photo scale."""
        if not self.image_scale:
            return 0.0
        return self.pixel_scale_mm(fit) / 1000.0 * self.image_scale
