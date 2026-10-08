"""Satellite sensor models.

Optical satellite scenes ship with Rational Polynomial Coefficients rather than
a physical camera model: two ratios of cubic polynomials that map ground
latitude, longitude and height to image line and sample. This module evaluates
them, inverts them, and -- the part that actually matters operationally --
refines them against ground control.

Vendor RPCs are typically good to somewhere between 5 and 50 metres absolute.
A handful of GCPs and an affine refinement in image space pulls that to
sub-pixel without needing a rigorous orbital model, which is why this approach
has largely displaced Toutin-style physical models for production work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

__all__ = ["RpcModel", "load_rpc", "refine_rpc", "RpcRefinement"]


def _cubic_terms(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """The 20 RPC basis terms, in the standard RPC00B order."""
    return np.stack(
        [
            np.ones_like(x), x, y, z,
            x * y, x * z, y * z,
            x * x, y * y, z * z,
            x * y * z,
            x ** 3, x * y * y, x * z * z,
            x * x * y, y ** 3, y * z * z,
            x * x * z, y * y * z, z ** 3,
        ]
    )


@dataclass
class RpcModel:
    """A rational polynomial sensor model with optional affine refinement."""

    line_num: Sequence[float]
    line_den: Sequence[float]
    sample_num: Sequence[float]
    sample_den: Sequence[float]

    line_offset: float = 0.0
    line_scale: float = 1.0
    sample_offset: float = 0.0
    sample_scale: float = 1.0
    lat_offset: float = 0.0
    lat_scale: float = 1.0
    lon_offset: float = 0.0
    lon_scale: float = 1.0
    height_offset: float = 0.0
    height_scale: float = 1.0

    # Affine refinement in image space: applied after the RPC evaluation.
    # [a0, a1, a2] for sample, [b0, b1, b2] for line.
    adjust_sample: Sequence[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    adjust_line: Sequence[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def to_dict(self) -> dict:
        return {
            "lineNum": list(self.line_num),
            "lineDen": list(self.line_den),
            "sampleNum": list(self.sample_num),
            "sampleDen": list(self.sample_den),
            "lineOffset": self.line_offset, "lineScale": self.line_scale,
            "sampleOffset": self.sample_offset, "sampleScale": self.sample_scale,
            "latOffset": self.lat_offset, "latScale": self.lat_scale,
            "lonOffset": self.lon_offset, "lonScale": self.lon_scale,
            "heightOffset": self.height_offset, "heightScale": self.height_scale,
            "adjustSample": list(self.adjust_sample),
            "adjustLine": list(self.adjust_line),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RpcModel":
        return cls(
            line_num=d["lineNum"], line_den=d["lineDen"],
            sample_num=d["sampleNum"], sample_den=d["sampleDen"],
            line_offset=d.get("lineOffset", 0.0), line_scale=d.get("lineScale", 1.0),
            sample_offset=d.get("sampleOffset", 0.0), sample_scale=d.get("sampleScale", 1.0),
            lat_offset=d.get("latOffset", 0.0), lat_scale=d.get("latScale", 1.0),
            lon_offset=d.get("lonOffset", 0.0), lon_scale=d.get("lonScale", 1.0),
            height_offset=d.get("heightOffset", 0.0), height_scale=d.get("heightScale", 1.0),
            adjust_sample=d.get("adjustSample", [0.0, 0.0, 0.0]),
            adjust_line=d.get("adjustLine", [0.0, 0.0, 0.0]),
        )

    def forward(self, lon, lat, height) -> tuple[np.ndarray, np.ndarray]:
        """Ground (lon, lat, height) -> image (sample, line), i.e. (col, row)."""
        lon = np.atleast_1d(np.asarray(lon, dtype=float))
        lat = np.atleast_1d(np.asarray(lat, dtype=float))
        height = np.atleast_1d(np.asarray(height, dtype=float))

        x = (lon - self.lon_offset) / self.lon_scale
        y = (lat - self.lat_offset) / self.lat_scale
        z = (height - self.height_offset) / self.height_scale

        terms = _cubic_terms(x, y, z)
        line_n = np.asarray(self.line_num) @ terms
        line_d = np.asarray(self.line_den) @ terms
        sample_n = np.asarray(self.sample_num) @ terms
        sample_d = np.asarray(self.sample_den) @ terms

        with np.errstate(divide="ignore", invalid="ignore"):
            line = line_n / line_d * self.line_scale + self.line_offset
            sample = sample_n / sample_d * self.sample_scale + self.sample_offset

        a0, a1, a2 = self.adjust_sample
        b0, b1, b2 = self.adjust_line
        adjusted_sample = sample + a0 + a1 * sample + a2 * line
        adjusted_line = line + b0 + b1 * sample + b2 * line

        return adjusted_sample, adjusted_line

    def inverse(self, sample, line, height, iterations: int = 12) -> tuple[np.ndarray, np.ndarray]:
        """Image (sample, line) + height -> ground (lon, lat).

        RPCs have no analytic inverse, so this is Newton iteration on the
        forward model with a numerical 2x2 Jacobian. It converges in a handful
        of steps because the mapping is very nearly affine over a scene.
        """
        sample = np.atleast_1d(np.asarray(sample, dtype=float))
        line = np.atleast_1d(np.asarray(line, dtype=float))
        height = np.atleast_1d(np.asarray(height, dtype=float))

        lon = np.full(sample.shape, self.lon_offset, dtype=float)
        lat = np.full(sample.shape, self.lat_offset, dtype=float)

        step_lon = self.lon_scale * 1e-6
        step_lat = self.lat_scale * 1e-6

        for _ in range(iterations):
            s0, l0 = self.forward(lon, lat, height)
            ds = sample - s0
            dl = line - l0
            if np.nanmax(np.abs(ds)) < 1e-6 and np.nanmax(np.abs(dl)) < 1e-6:
                break

            s_lon, l_lon = self.forward(lon + step_lon, lat, height)
            s_lat, l_lat = self.forward(lon, lat + step_lat, height)

            j11 = (s_lon - s0) / step_lon
            j12 = (s_lat - s0) / step_lat
            j21 = (l_lon - l0) / step_lon
            j22 = (l_lat - l0) / step_lat

            det = j11 * j22 - j12 * j21
            det = np.where(np.abs(det) < 1e-15, np.nan, det)

            lon = lon + (ds * j22 - dl * j12) / det
            lat = lat + (dl * j11 - ds * j21) / det

        return lon, lat


def load_rpc(path: str) -> Optional[RpcModel]:
    """Read RPCs from a GeoTIFF's metadata or a sidecar _RPC.TXT.

    Vendors disagree on where RPCs live, so both routes are tried. Returns
    ``None`` when the scene genuinely has none, rather than raising -- the
    caller decides whether that is an error.
    """
    import rasterio

    try:
        with rasterio.open(path) as dataset:
            tags = dataset.tags(ns="RPC")
    except Exception:
        tags = {}

    if not tags:
        from pathlib import Path

        base = Path(path)
        for candidate in (
            base.with_suffix("").with_name(base.stem + "_RPC.TXT"),
            base.with_suffix(".RPB"),
            base.with_suffix(".rpc"),
        ):
            if candidate.exists():
                tags = _parse_rpc_text(candidate.read_text(encoding="utf-8", errors="ignore"))
                break

    if not tags:
        return None

    def coefficients(key: str) -> list[float]:
        raw = tags.get(key, "")
        if isinstance(raw, str):
            values = [float(v) for v in raw.replace(",", " ").split()]
        else:
            values = [float(v) for v in raw]
        if len(values) != 20:
            raise ValueError(f"RPC field {key} has {len(values)} coefficients, expected 20")
        return values

    def scalar(key: str, default: float = 0.0) -> float:
        try:
            return float(tags.get(key, default))
        except (TypeError, ValueError):
            return default

    return RpcModel(
        line_num=coefficients("LINE_NUM_COEFF"),
        line_den=coefficients("LINE_DEN_COEFF"),
        sample_num=coefficients("SAMP_NUM_COEFF"),
        sample_den=coefficients("SAMP_DEN_COEFF"),
        line_offset=scalar("LINE_OFF"), line_scale=scalar("LINE_SCALE", 1.0),
        sample_offset=scalar("SAMP_OFF"), sample_scale=scalar("SAMP_SCALE", 1.0),
        lat_offset=scalar("LAT_OFF"), lat_scale=scalar("LAT_SCALE", 1.0),
        lon_offset=scalar("LONG_OFF"), lon_scale=scalar("LONG_SCALE", 1.0),
        height_offset=scalar("HEIGHT_OFF"), height_scale=scalar("HEIGHT_SCALE", 1.0),
    )


def _parse_rpc_text(text: str) -> dict:
    """Parse an _RPC.TXT / .RPB sidecar into the GDAL tag vocabulary."""
    tags: dict[str, str] = {}
    collecting: Optional[str] = None
    values: list[str] = []

    alias = {
        "LINENUMCOEF": "LINE_NUM_COEFF", "LINEDENCOEF": "LINE_DEN_COEFF",
        "SAMPNUMCOEF": "SAMP_NUM_COEFF", "SAMPDENCOEF": "SAMP_DEN_COEFF",
        "LINEOFFSET": "LINE_OFF", "SAMPOFFSET": "SAMP_OFF",
        "LATOFFSET": "LAT_OFF", "LONGOFFSET": "LONG_OFF", "HEIGHTOFFSET": "HEIGHT_OFF",
        "LINESCALE": "LINE_SCALE", "SAMPSCALE": "SAMP_SCALE",
        "LATSCALE": "LAT_SCALE", "LONGSCALE": "LONG_SCALE", "HEIGHTSCALE": "HEIGHT_SCALE",
    }

    for raw_line in text.splitlines():
        line = raw_line.strip().rstrip(";")
        if not line:
            continue

        if collecting:
            cleaned = line.strip("(), ")
            if cleaned:
                values.extend(cleaned.split(","))
            if ")" in line:
                tags[collecting] = " ".join(v.strip() for v in values if v.strip())
                collecting, values = None, []
            continue

        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().upper().replace("_", "").replace(" ", "")
        key = alias.get(key, key)
        value = value.strip()

        if value.startswith("("):
            collecting = key
            values = [v for v in value.strip("()").split(",") if v.strip()]
            if ")" in value:
                tags[key] = " ".join(v.strip() for v in values)
                collecting, values = None, []
        else:
            tags[key] = value

    return tags


@dataclass
class RpcRefinement:
    model: RpcModel
    rms_before_px: float
    rms_after_px: float
    residuals_px: list
    point_count: int
    order: str

    def to_dict(self) -> dict:
        return {
            "model": self.model.to_dict(),
            "rmsBeforePx": self.rms_before_px,
            "rmsAfterPx": self.rms_after_px,
            "residualsPx": self.residuals_px,
            "pointCount": self.point_count,
            "order": self.order,
        }


def refine_rpc(
    model: RpcModel,
    observations: Sequence[tuple[float, float, float, float, float]],
    order: str = "auto",
) -> RpcRefinement:
    """Fit an image-space correction to the vendor RPC using ground control.

    ``observations`` are ``(sample, line, lon, lat, height)`` tuples.

    With one or two points only a shift can be solved; with three or more a
    full affine is available. ``order="auto"`` picks the richest model the
    control will support, which avoids the classic failure of fitting six
    parameters to three points and reporting a flattering zero residual.
    """
    if not observations:
        raise ValueError("RPC refinement needs at least one ground control point")

    data = np.asarray(observations, dtype=float)
    sample_obs, line_obs = data[:, 0], data[:, 1]
    lon, lat, height = data[:, 2], data[:, 3], data[:, 4]

    base = RpcModel.from_dict({**model.to_dict(),
                               "adjustSample": [0.0, 0.0, 0.0],
                               "adjustLine": [0.0, 0.0, 0.0]})
    sample_pred, line_pred = base.forward(lon, lat, height)

    residual_sample = sample_obs - sample_pred
    residual_line = line_obs - line_pred
    rms_before = float(np.sqrt(np.mean(residual_sample**2 + residual_line**2)))

    n = len(data)
    if order == "auto":
        chosen = "affine" if n >= 4 else ("shift" if n < 3 else "shift")
    else:
        chosen = order

    if chosen == "affine" and n >= 4:
        design = np.column_stack([np.ones(n), sample_pred, line_pred])
        adjust_sample, *_ = np.linalg.lstsq(design, residual_sample, rcond=None)
        adjust_line, *_ = np.linalg.lstsq(design, residual_line, rcond=None)
    else:
        chosen = "shift"
        adjust_sample = np.array([float(np.mean(residual_sample)), 0.0, 0.0])
        adjust_line = np.array([float(np.mean(residual_line)), 0.0, 0.0])

    refined = RpcModel.from_dict({
        **model.to_dict(),
        "adjustSample": adjust_sample.tolist(),
        "adjustLine": adjust_line.tolist(),
    })

    sample_after, line_after = refined.forward(lon, lat, height)
    dx = sample_obs - sample_after
    dy = line_obs - line_after
    rms_after = float(np.sqrt(np.mean(dx**2 + dy**2)))

    return RpcRefinement(
        model=refined,
        rms_before_px=rms_before,
        rms_after_px=rms_after,
        residuals_px=[
            {"dx": float(a), "dy": float(b), "magnitude": float(np.hypot(a, b))}
            for a, b in zip(dx, dy)
        ],
        point_count=n,
        order=chosen,
    )
