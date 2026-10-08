"""Project and residual reports.

These are deliverables, not debug output -- they are filed with the job and
handed to whoever checks the work. Both are produced as plain text
that reads the way a photogrammetric report has read for decades, and as
structured data the interface renders as a sortable table.

The residual report carries the one control that matters for cleaning a block:
every point can be listed in pixels or in ground units, with control and check
points separated, so a blunder is obvious at a glance rather than needing to be
hunted through a scrolling list.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

import numpy as np

from .camera import CameraModel, FiducialFit
from .geodesy import describe_crs

__all__ = ["project_report", "residual_report", "residual_table", "accuracy_statement"]


def accuracy_statement(residuals: dict, check_points: set) -> dict:
    """Positional accuracy from independent check points, in the standard forms.

    ``residuals`` maps point id -> (dX, dY, dZ), surveyed minus adjusted.

    * RMSE per axis, horizontal (RMSE_r = sqrt(RMSE_x^2 + RMSE_y^2)) and
      vertical: the ASPRS Positional Accuracy Standards (2023) report these.
    * NSSDA 95% confidence (FGDC-STD-007.3-1998): horizontal 1.7308 RMSE_r
      when RMSE_x and RMSE_y are similar, otherwise the 2.4477 x mean(RMSE_x,
      RMSE_y) approximation (valid for ratios 0.6 to 1.0); vertical
      1.9600 RMSE_z.
    * Mean error per axis, which exposes a systematic offset that an RMSE
      alone hides.

    Only check points count: control points are part of the solution and
    measure its fit, not its accuracy. Fewer than 20 check points is below
    what the standards recommend, which is stated.
    """
    rows = np.array([[np.nan if v is None else v for v in residuals[p]]
                     for p in check_points if p in residuals], dtype=float).reshape(-1, 3)
    out: dict = {"checkPoints": int(len(rows))}
    if len(rows) == 0:
        return out
    # A check point seen on one photo checks position only (dZ is NaN).
    vertical = np.isfinite(rows[:, 2])
    out["checkPointsVertical"] = int(vertical.sum())
    rmse = np.array([np.sqrt(np.mean(rows[:, 0] ** 2)), np.sqrt(np.mean(rows[:, 1] ** 2)),
                     np.sqrt(np.mean(rows[vertical, 2] ** 2)) if vertical.any() else np.nan])
    mean = np.array([rows[:, 0].mean(), rows[:, 1].mean(),
                     rows[vertical, 2].mean() if vertical.any() else np.nan])
    rmse_r = float(np.hypot(rmse[0], rmse[1]))
    low, high = sorted((float(rmse[0]), float(rmse[1])))
    ratio = low / high if high > 0 else 1.0
    if ratio >= 0.6:
        # Equal to 1.7308 RMSE_r when RMSE_x = RMSE_y.
        horizontal_95 = 2.4477 * 0.5 * (rmse[0] + rmse[1])
    else:
        horizontal_95 = None   # outside the range the NSSDA approximation covers
    out.update({
        "rmseX": float(rmse[0]), "rmseY": float(rmse[1]),
        "rmseZ": float(rmse[2]) if vertical.any() else None,
        "rmseHorizontal": rmse_r,
        "rmseVertical": float(rmse[2]) if vertical.any() else None,
        "meanX": float(mean[0]), "meanY": float(mean[1]),
        "meanZ": float(mean[2]) if vertical.any() else None,
        "maxHorizontal": float(np.max(np.hypot(rows[:, 0], rows[:, 1]))),
        "maxVertical": float(np.max(np.abs(rows[vertical, 2]))) if vertical.any() else None,
        "nssdaHorizontal95": float(horizontal_95) if horizontal_95 is not None else None,
        "nssdaVertical95": float(1.96 * rmse[2]) if vertical.any() else None,
        "enoughCheckPoints": bool(len(rows) >= 20),
    })
    return out

_RULE = "=" * 78
_THIN = "-" * 78


def _fmt(value, spec: str = "12.4f", blank: str = "n/a") -> str:
    if value is None:
        return blank.rjust(int(spec.split(".")[0].strip("+-")) if "." in spec else 10)
    try:
        if isinstance(value, float) and not math.isfinite(value):
            return blank
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def _adjustment_details(model: dict) -> list[str]:
    """Weights, self-calibration, data snooping and accuracy for the report."""
    lines: list[str] = []
    settings = model.get("settings") or {}
    if model.get("method") == "bundle":
        lines += [
            "  A-priori standard deviations",
            f"    Control image measurements : {_fmt((model.get('imageSigmaMm') or 0) * 1000, '8.2f')} um",
            f"    Tie point measurements     : {_fmt((model.get('tieSigmaMm') or 0) * 1000, '8.2f')} um",
            f"    Control, horizontal        : {_fmt(model.get('controlSigmaM'), '8.3f')} m",
            f"    Control, vertical          : {_fmt(model.get('controlSigmaZM'), '8.3f')} m",
        ]
        if model.get("exteriorObservationResiduals"):
            lines.append(
                f"    Camera positions (GNSS)    : {_fmt(settings.get('eoSigmaXY'), '8.3f')} m horizontal, "
                f"{_fmt(settings.get('eoSigmaZ'), '6.3f')} m vertical")
            if model.get("gnssShiftM"):
                shift = model["gnssShiftM"]
                lines.append(f"    GNSS shift solved          : {shift[0]:.3f} {shift[1]:.3f} {shift[2]:.3f} m")
        lines.append(f"  Estimation             : "
                     f"{'robust (soft-L1, then Cauchy)' if settings.get('robust', True) else 'least squares'}")
        if model.get("meanRedundancy") is not None:
            lines.append(f"  Mean redundancy number : {model['meanRedundancy']:.3f}")
        lines.append("")

        calibration = model.get("selfCalibration")
        if calibration and calibration.get("values"):
            lines += [
                f"  Self-calibration ({calibration.get('set')}), radius {calibration.get('radiusMm', 0):.2f} mm",
                f"    {'Parameter':<10s} {'Value':>14s} {'Std dev':>14s} {'Significant':>12s}",
            ]
            for name, value in calibration["values"].items():
                sigma = (calibration.get("sigmas") or {}).get(name)
                significant = (abs(value) > 3 * sigma) if sigma else None
                lines.append(f"    {name:<10s} {_fmt(value, '14.6f')} {_fmt(sigma, '14.6f')} "
                             f"{'yes' if significant else 'no' if significant is not None else '-':>12s}")
            lines.append("")

        rejected = model.get("rejectedTies") or {}
        single = model.get("singleRayPoints") or []
        if rejected or single:
            lines.append(f"  Tie points rejected by data snooping : {len(rejected)}")
            lines.append(f"  Points on one photo only (unused)    : {len(single)}")
            lines.append("")

    accuracy = model.get("accuracy") or {}
    if accuracy.get("checkPoints"):
        lines += [
            _THIN,
            "  ACCURACY ASSESSMENT (independent check points)",
            _THIN,
            f"  Check points           : {accuracy['checkPoints']}"
            + ("" if accuracy.get("enoughCheckPoints") else "  (standards recommend at least 20)"),
            f"  RMSE X / Y / Z         : {_fmt(accuracy.get('rmseX'), '8.3f')} "
            f"{_fmt(accuracy.get('rmseY'), '8.3f')} {_fmt(accuracy.get('rmseZ'), '8.3f')} m",
            f"  Mean X / Y / Z         : {_fmt(accuracy.get('meanX'), '8.3f')} "
            f"{_fmt(accuracy.get('meanY'), '8.3f')} {_fmt(accuracy.get('meanZ'), '8.3f')} m",
            f"  RMSE horizontal (r)    : {_fmt(accuracy.get('rmseHorizontal'), '8.3f')} m   (ASPRS 2023)",
            f"  RMSE vertical          : {_fmt(accuracy.get('rmseVertical'), '8.3f')} m   (ASPRS 2023)",
            f"  Horizontal at 95%      : {_fmt(accuracy.get('nssdaHorizontal95'), '8.3f')} m   (NSSDA)",
            f"  Vertical at 95%        : {_fmt(accuracy.get('nssdaVertical95'), '8.3f')} m   (NSSDA)",
            "",
        ]
    return lines


def _header(title: str, project: dict) -> list[str]:
    return [
        _RULE,
        f"  FIDUCIA  |  {title}",
        _RULE,
        f"  Project          : {project.get('name', '')}",
        f"  Description      : {project.get('description', '') or '-'}",
        f"  Generated        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Project created  : {project.get('created', '')}",
        f"  Last modified    : {project.get('modified', '')}",
        "",
    ]


def _applied(project: dict, which: str) -> str:
    used = ((project.get("model") or {}).get("corrections") or {}).get(which)
    ticked = bool((project.get("camera") or {}).get(
        "applyAtmospheric" if which == "refraction" else "applyEarthCurvature"))
    if used:
        return "applied"
    if ticked:
        return "requested, but the model was solved without it -- recompute"
    return "not applied"


def _ground_scale(project: dict, image: Optional[dict]) -> float:
    """Millimetres of image per metre on the ground, for unit conversion.

    From the solved orientation where there is one: focal length over the
    height of the camera above the ground -- not above sea level, which on
    high ground (1700 m at Johannesburg) would understate every ground
    residual by that proportion. The nominal photo scale is only a fallback.
    """
    camera = CameraModel.from_dict(project.get("camera") or {})
    if camera.focal_mm > 0 and image and image.get("exterior"):
        heights = [float(g["z"]) for g in project.get("gcps", []) if g.get("z") is not None]
        ground = float(np.mean(heights)) if heights else 0.0
        above = float(image["exterior"][2]) - ground
        if above > 1.0:
            return camera.focal_mm / above
    if camera.kind == "film" and camera.image_scale:
        return 1000.0 / camera.image_scale
    return 1.0


def project_report(
    project: dict,
    include: Optional[dict] = None,
    image_ids: Optional[list[str]] = None,
) -> str:
    """The full project report.

    ``include`` toggles sections, mirroring the checkboxes of a classic
    project report. Everything is on by default: an incomplete export is easy
    to miss, and complete is the safer default for a deliverable.
    """
    include = {
        "projectInfo": True,
        "cameraInfo": True,
        "projection": True,
        "imageInfo": True,
        "fiducials": True,
        "exteriorOrientation": True,
        "gcps": True,
        "tiePoints": True,
        "modelStatistics": True,
        "outputs": True,
        **(include or {}),
    }

    images = project.get("images", [])
    if image_ids:
        images = [i for i in images if i["id"] in image_ids]

    lines = _header("PROJECT REPORT", project)

    if include["projectInfo"]:
        math_model = project.get("mathModel", {})
        lines += [
            _THIN,
            "  MATH MODEL",
            _THIN,
            f"  Model            : {math_model.get('kind', '-')}",
            f"  Exterior source  : {math_model.get('exteriorSource', '-')}",
            f"  Images in project: {len(project.get('images', []))}",
            f"  Images reported  : {len(images)}",
            "",
        ]

    if include["cameraInfo"] and project.get("camera"):
        camera = CameraModel.from_dict(project["camera"])
        lines += [_THIN, "  CAMERA CALIBRATION (INTERIOR ORIENTATION)", _THIN]
        lines += [
            f"  Camera name      : {camera.name or '-'}",
            f"  Type             : {'Scanned film' if camera.kind == 'film' else 'Digital / UAV'}",
            f"  Focal length     : {_fmt(camera.focal_mm, '10.4f')} mm",
            f"  Principal point  : x {_fmt(camera.ppo_x_mm, '10.6f')} mm   "
            f"y {_fmt(camera.ppo_y_mm, '10.6f')} mm",
        ]
        if camera.kind == "film":
            lines += [
                f"  Image scale      : 1 : {int(camera.image_scale) if camera.image_scale else '-'}",
                f"  Fiducial layout  : {camera.fiducial_position}",
            ]
        else:
            lines += [
                f"  Pixel pitch      : {_fmt(camera.pixel_pitch_mm, '10.6f')} mm",
                f"  Chip size        : {camera.columns} x {camera.rows} pixels",
                f"  Long-track PPO   : {_fmt(camera.long_track_offset_mm, '10.6f')} mm",
                f"  Cross-track PPO  : {_fmt(camera.cross_track_offset_mm, '10.6f')} mm",
            ]
        lines += [
            "",
            "  Radial lens distortion   dr = K0*r + K1*r^3 + K2*r^5 + K3*r^7",
            f"    K0 (R1) = {camera.k0: .6e}      K1 (R3) = {camera.k1: .6e}",
            f"    K2 (R5) = {camera.k2: .6e}      K3 (R7) = {camera.k3: .6e}",
            f"  Decentering  P1 = {camera.p1: .6e}   P2 = {camera.p2: .6e}",
            # What the last solution actually used, not merely what is ticked.
            "  Atmospheric refraction   : " + _applied(project, "refraction"),
            "  Earth curvature          : " + _applied(project, "curvature"),
            "",
        ]
        if camera.fiducials_mm:
            lines += ["  Calibrated fiducial marks (mm):"]
            for slot, (fx, fy) in camera.fiducials_mm.items():
                lines.append(f"    {slot:<16s} x {_fmt(fx, '10.4f')}   y {_fmt(fy, '10.4f')}")
            lines.append("")

    if include["projection"]:
        projection = project.get("projection", {})
        lines += [_THIN, "  PROJECTION", _THIN]
        for label, key in (("Output", "output"), ("GCP", "gcpSource")):
            identifier = projection.get(key)
            if not identifier:
                lines.append(f"  {label} projection : not set")
                continue
            try:
                described = describe_crs(identifier)
                lines += [
                    f"  {label} projection : {described['label']}",
                    f"    Datum          : {described['datum']}",
                    f"    Units          : {described['unit']}",
                    f"    EPSG           : {described['epsg'] or 'custom'}",
                ]
            except Exception as exc:
                lines.append(f"  {label} projection : {identifier} (unresolved: {exc})")
        lines += [
            f"  Output pixel spacing : {projection.get('pixelSpacingX')} x "
            f"{projection.get('pixelSpacingY')}",
            f"  Elevation reference  : {projection.get('elevationReference')}",
            "",
        ]

    if include["imageInfo"]:
        lines += [_THIN, "  IMAGES", _THIN]
        lines.append(
            f"  {'Name':<26s} {'Size':>13s} {'Bands':>6s} {'Status':>9s}  {'Fid RMS(px)':>11s}"
        )
        for image in images:
            size = f"{image.get('width', 0)} x {image.get('height', 0)}"
            fit = image.get("fiducialFit")
            rms = f"{fit['rmsPx']:.3f}" if fit else "-"
            status = "online" if image.get("online") else "OFFLINE"
            lines.append(
                f"  {image.get('name', '')[:26]:<26s} {size:>13s} "
                f"{image.get('bandCount', 0):>6d} {status:>9s}  {rms:>11s}"
            )
            lines.append(f"      path: {image.get('path', '')}")
        lines.append("")

    if include["fiducials"]:
        for image in images:
            fit = image.get("fiducialFit")
            if not fit:
                continue
            lines += [
                _THIN,
                f"  FIDUCIAL MEASUREMENT - {image.get('name')}",
                _THIN,
                f"  Transform          : {fit.get('transformKind', 'affine')}",
                f"  RMS residual       : {_fmt(fit.get('rmsPx'), '8.4f')} pixels",
                f"  Maximum residual   : {_fmt(fit.get('maxPx'), '8.4f')} pixels",
                "",
                f"  {'Fiducial':<16s} {'Col':>11s} {'Row':>11s} {'Residual(px)':>13s}",
            ]
            measured = image.get("fiducials", {})
            for slot, residual in zip(fit.get("used", []), fit.get("residualsPx", [])):
                col, row = measured.get(slot, (None, None))
                lines.append(
                    f"  {slot:<16s} {_fmt(col, '11.2f')} {_fmt(row, '11.2f')} "
                    f"{_fmt(residual, '13.4f')}"
                )
            lines.append("")

    if include["exteriorOrientation"]:
        lines += [_THIN, "  EXTERIOR ORIENTATION", _THIN]
        lines.append(
            f"  {'Image':<22s} {'X0':>14s} {'Y0':>14s} {'Z0':>12s} "
            f"{'omega':>9s} {'phi':>9s} {'kappa':>9s}"
        )
        for image in images:
            eo = image.get("exterior")
            if not eo:
                lines.append(f"  {image.get('name', '')[:22]:<22s}   not solved")
                continue
            lines.append(
                f"  {image.get('name', '')[:22]:<22s} {_fmt(eo[0], '14.3f')} "
                f"{_fmt(eo[1], '14.3f')} {_fmt(eo[2], '12.3f')} "
                f"{_fmt(math.degrees(eo[3]), '9.4f')} {_fmt(math.degrees(eo[4]), '9.4f')} "
                f"{_fmt(math.degrees(eo[5]), '9.4f')}"
            )
        lines += ["", "  Angles in degrees. Rotation order Rz(kappa) Ry(phi) Rx(omega).", ""]

    if include["gcps"]:
        gcps = project.get("gcps", [])
        control = [g for g in gcps if not g.get("isCheckPoint")]
        checks = [g for g in gcps if g.get("isCheckPoint")]
        lines += [
            _THIN,
            "  GROUND CONTROL POINTS",
            _THIN,
            f"  Active control points : {len(control)}",
            f"  Check points          : {len(checks)}",
            "",
            f"  {'ID':<10s} {'Type':<7s} {'X':>14s} {'Y':>14s} {'Z':>10s} {'Images':>7s}",
        ]
        observations = project.get("observations", [])
        for gcp in gcps:
            used = sum(1 for o in observations if o.get("pointId") == gcp.get("id"))
            kind = "check" if gcp.get("isCheckPoint") else "GCP"
            lines.append(
                f"  {str(gcp.get('id', ''))[:10]:<10s} {kind:<7s} "
                f"{_fmt(gcp.get('x'), '14.3f')} {_fmt(gcp.get('y'), '14.3f')} "
                f"{_fmt(gcp.get('z'), '10.3f')} {used:>7d}"
            )
        lines.append("")

    if include["tiePoints"]:
        tie_points = project.get("tiePoints", [])
        lines += [
            _THIN,
            "  TIE POINTS",
            _THIN,
            f"  Total tie points      : {len(tie_points)}",
        ]
        observations = [o for o in project.get("observations", []) if o.get("kind") == "tie"]
        per_image: dict[str, int] = {}
        for observation in observations:
            per_image[observation["imageId"]] = per_image.get(observation["imageId"], 0) + 1
        names = {i["id"]: i.get("name", i["id"]) for i in project.get("images", [])}
        for image_id, count in sorted(per_image.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {names.get(image_id, image_id):<26s} {count:>5d} measurements")
        lines.append("")

    if include["modelStatistics"] and project.get("model"):
        model = project["model"]
        lines += [
            _THIN,
            "  MODEL / BUNDLE ADJUSTMENT",
            _THIN,
            f"  Solved at            : {model.get('solvedAt', '-')}",
            f"  Method               : {model.get('method', '-')}",
            f"  Converged            : {'yes' if model.get('converged') else 'NO'}",
            f"  Iterations           : {model.get('iterations', '-')}",
            f"  Degrees of freedom   : {model.get('degreesOfFreedom', '-')}",
            f"  Sigma-0              : {_fmt(model.get('sigma0'), '10.6f')}",
            f"  RMS image residual   : {_fmt(model.get('rmsImageMm'), '10.6f')} mm",
            f"  RMS control residual : {_fmt(model.get('rmsControlM'), '10.4f')} m",
            f"  RMS check residual   : {_fmt(model.get('rmsCheckM'), '10.4f')} m",
            f"  Message              : {model.get('message', '')}",
            "",
        ]
        lines += _adjustment_details(model)

    if include["outputs"]:
        lines += [_THIN, "  OUTPUTS", _THIN]
        orthos = project.get("orthos", [])
        if orthos:
            for ortho in orthos:
                lines.append(
                    f"  ORTHO  {ortho.get('name', '')[:28]:<28s} "
                    f"{ortho.get('width', 0)} x {ortho.get('height', 0)} px"
                )
                lines.append(f"         {ortho.get('path', '')}")
        else:
            lines.append("  No orthoimages generated.")
        for entry in project.get("mosaics", []):
            lines.append(f"  MOSAIC {entry.get('name', '')[:28]:<28s} {entry.get('path', '')}")
        lines.append("")

    lines += [_RULE, "  End of project report", _RULE]
    return "\n".join(lines)


def residual_table(project: dict, units: str = "ground", show: str = "all") -> list[dict]:
    """Structured residuals for the interface's sortable table.

    ``units`` is ``"ground"`` (metres) or ``"pixels"``. ``show`` filters to
    ``"gcp"``, ``"check"``, ``"tie"`` or ``"all"``.
    """
    model = project.get("model") or {}
    residuals = model.get("residuals") or {}
    images = {i["id"]: i for i in project.get("images", [])}
    gcps = {g["id"]: g for g in project.get("gcps", [])}
    camera = CameraModel.from_dict(project.get("camera") or {})

    rows: list[dict] = []
    for key, value in residuals.items():
        image_id, point_id = key.split("|", 1) if "|" in key else (key, "")
        image = images.get(image_id)
        gcp = gcps.get(point_id)

        kind = "tie"
        if gcp is not None:
            kind = "check" if gcp.get("isCheckPoint") else "gcp"
        if show != "all" and show != kind:
            continue

        dx_mm, dy_mm = float(value[0]), float(value[1])

        fit = FiducialFit.from_dict(image["fiducialFit"]) if image and image.get("fiducialFit") else None
        pixel_mm = camera.pixel_scale_mm(fit) or 1.0
        ground_per_mm = 1.0 / _ground_scale(project, image) if _ground_scale(project, image) else 1.0

        if units == "pixels":
            dx, dy = dx_mm / pixel_mm, dy_mm / pixel_mm
        else:
            dx, dy = dx_mm * ground_per_mm, dy_mm * ground_per_mm

        rows.append(
            {
                "imageId": image_id,
                "imageName": image.get("name", image_id) if image else image_id,
                "pointId": point_id,
                "kind": kind,
                "dx": dx,
                "dy": dy,
                "magnitude": float(math.hypot(dx, dy)),
                "units": "px" if units == "pixels" else "m",
            }
        )

    rows.sort(key=lambda r: r["magnitude"], reverse=True)
    return rows


def residual_report(
    project: dict,
    units: str = "ground",
    show: str = "all",
    residual_type: str = "rms",
) -> str:
    """The residual report, formatted for submission."""
    rows = residual_table(project, units, show)
    model = project.get("model") or {}
    unit_label = "pixels" if units == "pixels" else "ground units (m)"

    lines = _header("RESIDUAL REPORT", project)
    lines += [
        _THIN,
        "  SETTINGS",
        _THIN,
        f"  Residual type    : {residual_type.upper()}",
        f"  Residual units   : {unit_label}",
        f"  Points shown     : {show}",
        f"  Model solved at  : {model.get('solvedAt', 'not solved')}",
        "",
    ]

    if not rows:
        lines += ["  No residuals available. Compute the model first.", "", _RULE]
        return "\n".join(lines)

    lines += [
        _THIN,
        "  POINT RESIDUALS",
        _THIN,
        f"  {'Point':<10s} {'Type':<6s} {'Image':<24s} "
        f"{'dX':>12s} {'dY':>12s} {'Residual':>12s}",
    ]

    for row in rows:
        lines.append(
            f"  {row['pointId'][:10]:<10s} {row['kind']:<6s} {row['imageName'][:24]:<24s} "
            f"{_fmt(row['dx'], '12.4f')} {_fmt(row['dy'], '12.4f')} "
            f"{_fmt(row['magnitude'], '12.4f')}"
        )

    lines.append("")

    def summarise(label: str, subset: list[dict]) -> list[str]:
        if not subset:
            return [f"  {label:<22s} no points"]
        magnitudes = np.array([r["magnitude"] for r in subset], dtype=float)
        return [
            f"  {label:<22s} n = {len(subset):<4d}  "
            f"RMS {np.sqrt(np.mean(magnitudes**2)):9.4f}  "
            f"mean {magnitudes.mean():9.4f}  max {magnitudes.max():9.4f}"
        ]

    lines += [_THIN, f"  SUMMARY ({unit_label})", _THIN]
    lines += summarise("Ground control", [r for r in rows if r["kind"] == "gcp"])
    lines += summarise("Check points", [r for r in rows if r["kind"] == "check"])
    lines += summarise("Tie points", [r for r in rows if r["kind"] == "tie"])
    lines += summarise("All points", rows)
    lines += ["", _RULE, "  End of residual report", _RULE]

    return "\n".join(lines)
