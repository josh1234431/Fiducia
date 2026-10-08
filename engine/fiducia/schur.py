"""The normal equations of a bundle with its points eliminated.

A block's unknowns are a few per photo (orientation, and the shared
self-calibration) and three per point -- and there are hundreds of points
per photo. Each point is tied only to the photos that see it, so the
point-point part of the normal matrix is block diagonal, 3 x 3 per point, and
can be inverted point by point. Eliminating the points leaves the reduced
camera system

    S = U - W V^-1 W^T

of the size of the photo parameters alone. That is the Schur complement, and
it is what makes a drone block of hundreds of photos and tens of thousands of
points tractable: the expensive factorisation is of S, not of everything.

Unknowns are ordered photos first, then points (one contiguous run), then
anything shared (self-calibration, a GNSS shift).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["ReducedSystem"]

# Above this many photo-and-shared unknowns the reduced system is kept sparse.
DENSE_LIMIT = 6000


class ReducedSystem:
    """A factorised normal matrix with the points ``[p0, p1)`` eliminated."""

    def __init__(self, normal, p0: int, p1: int):
        from scipy.sparse import bsr_matrix, csr_matrix

        normal = csr_matrix(normal)
        n = normal.shape[0]
        self.n, self.p0, self.p1 = n, p0, p1
        self.m = (p1 - p0) // 3
        others = np.r_[np.arange(0, p0), np.arange(p1, n)]
        self.others = others

        # The 3 x 3 point blocks. Anything off the block diagonal would be a
        # point tied to another point, which a bundle never has.
        points = normal[p0:p1, p0:p1].tocoo()
        blocks = np.zeros((self.m, 3, 3))
        same = (points.row // 3) == (points.col // 3)
        np.add.at(blocks, (points.row[same] // 3, points.row[same] % 3, points.col[same] % 3),
                  points.data[same])
        # A point the photos barely constrain (rays nearly parallel) still
        # gets an inverse; a tiny ridge keeps it finite without moving it.
        ridge = 1e-12 * np.maximum(np.trace(blocks, axis1=1, axis2=2), 1e-300)
        blocks[:, [0, 1, 2], [0, 1, 2]] += ridge[:, None]
        self.v_inv_blocks = np.linalg.inv(blocks)
        self.v_inv = bsr_matrix((self.v_inv_blocks, np.arange(self.m), np.arange(self.m + 1)),
                                shape=(3 * self.m, 3 * self.m)).tocsr()

        # The photo rows and columns are the two runs either side of the
        # points; sliced as runs, which sparse matrices do quickly, rather
        # than picked out by index.
        from scipy.sparse import hstack, vstack

        head, tail = normal[:p0], normal[p1:]
        rows = vstack([head, tail]).tocsc()
        self.w = rows[:, p0:p1].tocsr()                    # photos x points
        u = hstack([rows[:, :p0], rows[:, p1:]]).tocsr()
        self.b = (self.w @ self.v_inv).tocsr()            # W V^-1
        s = (u - self.b @ self.w.T).tocsc()
        self.size = s.shape[0]
        self.dense = self.size <= DENSE_LIMIT
        if self.dense:
            from scipy.linalg import cho_factor, lu_factor

            s = s.toarray()
            s = 0.5 * (s + s.T)
            try:
                self._factor = ("cho", cho_factor(s, check_finite=False))
            except np.linalg.LinAlgError:
                self._factor = ("lu", lu_factor(s, check_finite=False))
        else:
            from scipy.sparse.linalg import splu

            self._factor = ("splu", splu(s))
        self._cov = None

    # -- solving ---------------------------------------------------------------

    def _solve_s(self, rhs: np.ndarray) -> np.ndarray:
        kind, factor = self._factor
        if kind == "cho":
            from scipy.linalg import cho_solve

            return cho_solve(factor, rhs, check_finite=False)
        if kind == "lu":
            from scipy.linalg import lu_solve

            return lu_solve(factor, rhs, check_finite=False)
        return factor.solve(rhs)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """x with N x = rhs."""
        rhs = np.asarray(rhs, dtype=float)
        r_c = rhs[self.others]
        r_p = rhs[self.p0:self.p1]
        x_c = self._solve_s(r_c - self.b @ r_p)
        x_p = self.v_inv @ (r_p - self.w.T @ x_c)
        out = np.empty(self.n)
        out[self.others] = x_c
        out[self.p0:self.p1] = x_p
        return out

    # -- covariance --------------------------------------------------------------

    def photo_covariance(self) -> Optional[np.ndarray]:
        """(N^-1) over the photo and shared unknowns: S^-1, dense.

        None when the reduced system is too large to hold densely; then only
        the per-photo blocks are available, through ``photo_block``.
        """
        if self._cov is None and self.dense:
            self._cov = self._solve_s(np.eye(self.size))
            self._cov = 0.5 * (self._cov + self._cov.T)
        return self._cov

    def reduced_index(self, index: int) -> int:
        """Position of an unknown (not a point) in the reduced system."""
        return index if index < self.p0 else index - (self.p1 - self.p0)

    def covariance_block(self, indices) -> np.ndarray:
        """(N^-1) restricted to some photo or shared unknowns."""
        idx = np.array([self.reduced_index(i) for i in indices])
        cov = self.photo_covariance()
        if cov is not None:
            return cov[np.ix_(idx, idx)]
        columns = np.zeros((self.size, len(idx)))
        columns[idx, np.arange(len(idx))] = 1.0
        return self._solve_s(columns)[idx]

    def point_variances(self, max_points: Optional[int] = None) -> Optional[np.ndarray]:
        """Variances (m, 3) of the points, from

            Cov_pp,i = V_i^-1 + (W V^-1)_i^T  S^-1  (W V^-1)_i

        evaluated over only the few photo unknowns each point touches.
        """
        if max_points is not None and self.m > max_points:
            return None
        cov = self.photo_covariance()
        if cov is None:
            return None
        # diag(B^T S^-1 B), a few thousand columns at a time: each column of B
        # touches only the photos that see its point, so B^T S^-1 is cheap.
        bt = self.b.T.tocsr()                               # points x photos
        extra = np.empty(3 * self.m)
        chunk = 6000
        for c0 in range(0, 3 * self.m, chunk):
            part = bt[c0:c0 + chunk]
            projected = np.asarray(part @ cov)              # (k, photos)
            extra[c0:c0 + chunk] = np.asarray(part.multiply(projected).sum(axis=1)).ravel()
        own = np.einsum("iaa->ia", self.v_inv_blocks)
        return own + extra.reshape(self.m, 3)
