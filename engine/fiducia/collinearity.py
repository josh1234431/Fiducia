"""Collinearity condition and rigid-body rotation helpers.

Everything downstream -- space resection, bundle adjustment, ortho resampling,
epipolar geometry -- is expressed in terms of the functions in this module, so
the sign and axis conventions are fixed here once and never restated.

Convention
----------
Rotation is the photogrammetric ``R = Rz(kappa) @ Ry(phi) @ Rx(omega)``, applied
as a ground-to-camera transform:

    q = R @ (P_ground - P_perspective_centre)

Image (film/focal-plane) coordinates in millimetres follow from

    x = -f * q0 / q2
    y = -f * q1 / q2

with ``f`` the calibrated focal length, ``x`` increasing to the right and ``y``
increasing upwards, origin at the principal point. This matches the convention
used by the standard aerial math model, so exterior orientation angles
exported from a classic project report can be read straight in.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "rotation_matrix",
    "rotation_derivatives",
    "ground_to_camera",
    "project_points",
    "collinearity_jacobian",
    "camera_position_from_angles",
]


def rotation_matrix(omega: float, phi: float, kappa: float) -> np.ndarray:
    """Ground-to-camera rotation ``Rz(kappa) @ Ry(phi) @ Rx(omega)``.

    Angles are in radians. The result is orthonormal to machine precision.
    """
    so, co = np.sin(omega), np.cos(omega)
    sp, cp = np.sin(phi), np.cos(phi)
    sk, ck = np.sin(kappa), np.cos(kappa)

    return np.array(
        [
            [cp * ck, co * sk + so * sp * ck, so * sk - co * sp * ck],
            [-cp * sk, co * ck - so * sp * sk, so * ck + co * sp * sk],
            [sp, -so * cp, co * cp],
        ]
    )


def rotation_derivatives(omega: float, phi: float, kappa: float):
    """Analytic partials of :func:`rotation_matrix` w.r.t. each angle.

    Returned as ``(dR_domega, dR_dphi, dR_dkappa)``. Used by the bundle
    adjustment to build an exact Jacobian -- finite differences on the rotation
    block are the single largest source of slow convergence in a block
    adjustment, so it is worth carrying these by hand.
    """
    so, co = np.sin(omega), np.cos(omega)
    sp, cp = np.sin(phi), np.cos(phi)
    sk, ck = np.sin(kappa), np.cos(kappa)

    d_omega = np.array(
        [
            [0.0, -so * sk + co * sp * ck, co * sk + so * sp * ck],
            [0.0, -so * ck - co * sp * sk, co * ck - so * sp * sk],
            [0.0, -co * cp, -so * cp],
        ]
    )

    d_phi = np.array(
        [
            [-sp * ck, so * cp * ck, -co * cp * ck],
            [sp * sk, -so * cp * sk, co * cp * sk],
            [cp, so * sp, -co * sp],
        ]
    )

    d_kappa = np.array(
        [
            [-cp * sk, co * ck - so * sp * sk, so * ck + co * sp * sk],
            [-cp * ck, -co * sk - so * sp * ck, -so * sk + co * sp * ck],
            [0.0, 0.0, 0.0],
        ]
    )

    return d_omega, d_phi, d_kappa


def ground_to_camera(ground: np.ndarray, eo: np.ndarray) -> np.ndarray:
    """Rotate ground points into the camera frame.

    ``ground`` is ``(N, 3)`` of X/Y/Z in projected map units (metres).
    ``eo`` is ``(X0, Y0, Z0, omega, phi, kappa)``.
    """
    ground = np.atleast_2d(np.asarray(ground, dtype=float))
    rot = rotation_matrix(eo[3], eo[4], eo[5])
    return (ground - np.asarray(eo[:3], dtype=float)) @ rot.T


def project_points(ground: np.ndarray, eo: np.ndarray, focal: float) -> np.ndarray:
    """Project ground points to focal-plane millimetres.

    Points at or behind the perspective centre return NaN rather than raising,
    so a caller resampling a large tile can mask them out in bulk.
    """
    cam = ground_to_camera(ground, eo)
    denom = cam[:, 2]
    bad = np.abs(denom) < 1e-9
    denom = np.where(bad, np.nan, denom)

    out = np.empty((cam.shape[0], 2), dtype=float)
    out[:, 0] = -focal * cam[:, 0] / denom
    out[:, 1] = -focal * cam[:, 1] / denom
    out[bad] = np.nan
    return out


def collinearity_jacobian(ground: np.ndarray, eo: np.ndarray, focal: float):
    """Partials of focal-plane (x, y) w.r.t. exterior orientation and ground XYZ.

    Returns ``(d_eo, d_ground)`` shaped ``(N, 2, 6)`` and ``(N, 2, 3)``.
    """
    ground = np.atleast_2d(np.asarray(ground, dtype=float))
    n = ground.shape[0]

    rot = rotation_matrix(eo[3], eo[4], eo[5])
    delta = ground - np.asarray(eo[:3], dtype=float)
    cam = delta @ rot.T
    u, v, w = cam[:, 0], cam[:, 1], cam[:, 2]
    inv_w = 1.0 / w

    # d(x,y)/d(u,v,w) -- the projective divide, common to every column below.
    dx_du = -focal * inv_w
    dx_dw = focal * u * inv_w**2
    dy_dv = -focal * inv_w
    dy_dw = focal * v * inv_w**2

    d_eo = np.zeros((n, 2, 6))
    d_ground = np.zeros((n, 2, 3))

    # Ground XYZ and the perspective-centre translation share a Jacobian up to
    # sign: moving the camera +1 in X is the same as moving the point -1 in X.
    for axis in range(3):
        du, dv, dw = rot[0, axis], rot[1, axis], rot[2, axis]
        d_ground[:, 0, axis] = dx_du * du + dx_dw * dw
        d_ground[:, 1, axis] = dy_dv * dv + dy_dw * dw
        d_eo[:, 0, axis] = -d_ground[:, 0, axis]
        d_eo[:, 1, axis] = -d_ground[:, 1, axis]

    for i, d_rot in enumerate(rotation_derivatives(eo[3], eo[4], eo[5])):
        d_cam = delta @ d_rot.T
        du, dv, dw = d_cam[:, 0], d_cam[:, 1], d_cam[:, 2]
        d_eo[:, 0, 3 + i] = dx_du * du + dx_dw * dw
        d_eo[:, 1, 3 + i] = dy_dv * dv + dy_dw * dw

    return d_eo, d_ground


def camera_position_from_angles(
    ground_centroid: np.ndarray,
    mean_elevation: float,
    focal_mm: float,
    image_scale: float,
) -> np.ndarray:
    """Seed a perspective centre from nominal flying height.

    ``H = f * S`` is the textbook relation between focal length, photo scale and
    height above mean ground. Good enough to start Gauss-Newton; the adjustment
    moves it the rest of the way.
    """
    height = (focal_mm / 1000.0) * float(image_scale)
    return np.array(
        [ground_centroid[0], ground_centroid[1], mean_elevation + height],
        dtype=float,
    )
