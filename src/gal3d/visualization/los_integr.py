"""
Adaptive Simpson line-of-sight integrator for :class:`~gal3d.density.DensitySource`.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from gal3d.density import DensitySource


class LOSIntegrator:
    """
    Adaptive Simpson line-of-sight integrator for any
    :class:`~gal3d.density.DensitySource`.

    Integrates the 3-D density along the *z*-axis for every pixel on an
    *xy* grid, producing a 2-D surface-density map.  The integrator
    refines each pixel independently until the Richardson-extrapolated
    error is within tolerance, or until *nz_max* samples have been used.

    Parameters
    ----------
    source : DensitySource
        Any object that implements ``_evaluate_density(pos)``.
    x_range : (float, float)
        Minimum and maximum extent along the *x*-axis of the output grid.
    y_range : (float, float)
        Minimum and maximum extent along the *y*-axis of the output grid.
    resolution : int, optional
        Number of pixels along each axis.  Default ``200``.
    rotation_matrix : ndarray, shape (3, 3), optional
        Rotation applied to query positions *before* evaluating the density.
        ``None`` (default) means no rotation.
    z_range : (float, float), optional
        Integration limits along the line of sight.  If ``None`` a
        symmetric range ``[-max(Δx, Δy), +max(Δx, Δy)]`` is used.
    nz_min : int, optional
        Minimum number of *z* samples (odd) for the initial Simpson grid.
        Default ``33``.
    nz_max : int, optional
        Maximum number of *z* samples per pixel before the integrator
        accepts an unconverged result with a warning.  Default ``4097``.
    rtol : float, optional
        Relative tolerance for the adaptive integrator.  Default ``1e-4``.
    atol : float, optional
        Absolute tolerance for the adaptive integrator.  Default ``1e-8``.
    eval_batch_size : int, optional
        Maximum number of density evaluations per batch.  Default
        ``1_000_000``.
    """

    def __init__(
        self,
        source: DensitySource,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        resolution: int = 200,
        rotation_matrix: np.ndarray | None = None,
        z_range: tuple[float, float] | None = None,
        nz_min: int = 33,
        nz_max: int = 4097,
        rtol: float = 1e-4,
        atol: float = 1e-8,
        eval_batch_size: int = 1_000_000,
    ) -> None:
        self.source = source
        self.x_range = x_range
        self.y_range = y_range
        self.resolution = resolution
        self.rotation_matrix = self._validate_rotation_matrix(rotation_matrix)
        self.z_range = z_range if z_range is not None else self._default_z_range(
            x_range, y_range
        )
        self.nz_min = self._to_odd(nz_min)
        self.nz_max = max(self._to_odd(nz_max), self.nz_min)
        self.rtol = rtol
        self.atol = atol
        self.eval_batch_size = eval_batch_size
        self._validate()

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_odd(n: int) -> int:
        """Return the smallest odd integer >= max(3, n)."""
        n = max(3, int(n))
        return n if n % 2 == 1 else n + 1

    @staticmethod
    def _validate_rotation_matrix(
        mat: np.ndarray | None,
    ) -> np.ndarray | None:
        if mat is None:
            return None
        mat = np.asarray(mat, dtype=float)
        if mat.shape != (3, 3):
            raise ValueError("rotation_matrix must have shape (3, 3).")
        return mat

    @staticmethod
    def _default_z_range(
        x_range: tuple[float, float],
        y_range: tuple[float, float],
    ) -> tuple[float, float]:
        half = max(x_range[1] - x_range[0], y_range[1] - y_range[0])
        return (-half, half)

    def _validate(self) -> None:
        if self.resolution <= 0:
            raise ValueError("resolution must be > 0.")
        if self.z_range[1] <= self.z_range[0]:
            raise ValueError("z_range must satisfy z_max > z_min.")
        if self.rtol < 0 or self.atol < 0:
            raise ValueError("rtol and atol must be >= 0.")

    # ------------------------------------------------------------------
    # Density evaluation
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate the source density at arbitrary (x, y, z) triplets.

        Applies ``rotation_matrix`` when set, and processes points in
        batches of ``eval_batch_size`` to limit peak memory usage.

        Parameters
        ----------
        x, y, z : ndarray, shape (n,)
            Coordinates of evaluation points.

        Returns
        -------
        ndarray, shape (n,)
            Density values.
        """
        n = x.size
        if n == 0:
            return np.empty(0, dtype=float)

        out = np.empty(n, dtype=float)
        bs = max(1, self.eval_batch_size)
        for i in range(0, n, bs):
            j = min(i + bs, n)
            pos = np.column_stack([x[i:j], y[i:j], z[i:j]])
            if self.rotation_matrix is not None:
                pos = pos @ self.rotation_matrix.T
            out[i:j] = self.source._evaluate_density(pos).astype(float, copy=False)
        return out

    # ------------------------------------------------------------------
    # Internal grid builders
    # ------------------------------------------------------------------

    def _build_pixel_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Return flattened pixel-centre x/y coordinates."""
        x = np.linspace(self.x_range[0], self.x_range[1], self.resolution)
        y = np.linspace(self.y_range[0], self.y_range[1], self.resolution)
        x_flat = np.tile(x, self.resolution)
        y_flat = np.repeat(y, self.resolution)
        return x_flat, y_flat

    def _evaluate_pixel_z_grid(
        self,
        x_pixels: np.ndarray,
        y_pixels: np.ndarray,
        z_samples: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate density on a subset of pixels at multiple *z* values.

        Parameters
        ----------
        x_pixels, y_pixels : ndarray, shape (p,)
            Pixel-centre coordinates.
        z_samples : ndarray, shape (nz,)
            *z* values at which to evaluate.

        Returns
        -------
        ndarray, shape (p, nz)
            Density values.
        """
        p = x_pixels.size
        nz = z_samples.size
        if p == 0:
            return np.empty((0, nz), dtype=float)
        x_rep = np.repeat(x_pixels, nz)
        y_rep = np.repeat(y_pixels, nz)
        z_rep = np.tile(z_samples, p)
        return self._evaluate(x_rep, y_rep, z_rep).reshape(p, nz)

    # ------------------------------------------------------------------
    # Simpson helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _simpson(
        a: np.ndarray,
        b: np.ndarray,
        fa: np.ndarray,
        fm: np.ndarray,
        fb: np.ndarray,
    ) -> np.ndarray:
        """Vectorized composite Simpson integral over intervals [a, b]."""
        return (b - a) * (fa + 4.0 * fm + fb) / 6.0

    def _max_split_depth(self) -> int:
        """Maximum adaptive bisection depth derived from nz_min/nz_max."""
        base = self.nz_min - 1
        mx = self.nz_max - 1
        return int(np.floor(np.log2(max(1.0, mx / base))))

    # ------------------------------------------------------------------
    # Adaptive integration
    # ------------------------------------------------------------------

    def _initialize_frontier(
        self,
        x_pixels: np.ndarray,
        y_pixels: np.ndarray,
        z0: float,
        z1: float,
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray,
    ]:
        """
        Build the initial adaptive frontier from ``nz_min`` base segments.

        Reuses endpoint density values from the base grid across adjacent
        intervals to minimise function evaluations.

        Parameters
        ----------
        x_pixels, y_pixels : ndarray, shape (p,)
            Pixel coordinates.
        z0, z1 : float
            Line-of-sight integration limits.

        Returns
        -------
        tuple
            ``(pidx, a, b, fa, fm, fb, s, depth)`` — per-interval arrays
            suitable for the adaptive loop.
        """
        z_base = np.linspace(z0, z1, self.nz_min)
        p = x_pixels.size
        n_int = z_base.size - 1

        rho_base = self._evaluate_pixel_z_grid(x_pixels, y_pixels, z_base)

        pidx = np.repeat(np.arange(p, dtype=np.int32), n_int)
        seg = np.tile(np.arange(n_int, dtype=np.int32), p)

        a = z_base[seg]
        b = z_base[seg + 1]
        depth = np.zeros(pidx.size, dtype=np.int32)
        fa = rho_base[pidx, seg]
        fb = rho_base[pidx, seg + 1]

        m = 0.5 * (a + b)
        fm = self._evaluate(x_pixels[pidx], y_pixels[pidx], m)
        s = self._simpson(a, b, fa, fm, fb)
        return pidx, a, b, fa, fm, fb, s, depth

    def _adaptive_iteration(
        self,
        xp: np.ndarray,
        yp: np.ndarray,
        total: np.ndarray,
        converged: np.ndarray,
        length: float,
        max_depth: int,
        pidx: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        fa: np.ndarray,
        fm: np.ndarray,
        fb: np.ndarray,
        s: np.ndarray,
        depth: np.ndarray,
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, bool,
    ]:
        """Run one adaptive-refinement iteration on the current frontier."""
        m = 0.5 * (a + b)
        lm = 0.5 * (a + m)
        rm = 0.5 * (m + b)

        n_seg = pidx.size
        x2 = np.concatenate([xp[pidx], xp[pidx]])
        y2 = np.concatenate([yp[pidx], yp[pidx]])
        z2 = np.concatenate([lm, rm])
        rho2 = self._evaluate(x2, y2, z2)
        flm, frm = rho2[:n_seg], rho2[n_seg:]

        s_left = self._simpson(a, m, fa, flm, fm)
        s_right = self._simpson(m, b, fm, frm, fb)
        s2 = s_left + s_right

        err = np.abs(s2 - s) / 15.0
        atol_local = self.atol * (b - a) / length
        tol = atol_local + self.rtol * np.abs(s2)
        s_rich = s2 + (s2 - s) / 15.0

        accept = err <= tol
        np.add.at(total, pidx[accept], s_rich[accept])

        reject = ~accept
        if not np.any(reject):
            return (
                np.empty(0, dtype=pidx.dtype),
                np.empty(0, dtype=a.dtype),
                np.empty(0, dtype=b.dtype),
                np.empty(0, dtype=fa.dtype),
                np.empty(0, dtype=fm.dtype),
                np.empty(0, dtype=fb.dtype),
                np.empty(0, dtype=s.dtype),
                np.empty(0, dtype=depth.dtype),
                True,
            )

        too_deep = reject & (depth >= max_depth)
        if np.any(too_deep):
            np.add.at(total, pidx[too_deep], s_rich[too_deep])
            converged[pidx[too_deep]] = False

        split = reject & ~too_deep
        if not np.any(split):
            return (
                np.empty(0, dtype=pidx.dtype),
                np.empty(0, dtype=a.dtype),
                np.empty(0, dtype=b.dtype),
                np.empty(0, dtype=fa.dtype),
                np.empty(0, dtype=fm.dtype),
                np.empty(0, dtype=fb.dtype),
                np.empty(0, dtype=s.dtype),
                np.empty(0, dtype=depth.dtype),
                True,
            )

        pid_s = pidx[split]
        a_s, b_s, m_s = a[split], b[split], m[split]
        fa_s, fm_s, fb_s = fa[split], fm[split], fb[split]
        flm_s, frm_s = flm[split], frm[split]
        sl_s, sr_s = s_left[split], s_right[split]
        d_s = depth[split] + 1

        return (
            np.concatenate([pid_s, pid_s]),
            np.concatenate([a_s, m_s]),
            np.concatenate([m_s, b_s]),
            np.concatenate([fa_s, fm_s]),
            np.concatenate([flm_s, frm_s]),
            np.concatenate([fm_s, fb_s]),
            np.concatenate([sl_s, sr_s]),
            np.concatenate([d_s, d_s]),
            False,
        )

    def _integrate_pixels(
        self,
        pixel_ids: np.ndarray,
        x_flat: np.ndarray,
        y_flat: np.ndarray,
        z0: float,
        z1: float,
    ) -> np.ndarray:
        """
        Adaptively integrate the density along *z* for a set of pixels.

        Uses Richardson extrapolation to estimate errors and bisects
        intervals that have not converged.  Intervals that exceed the
        maximum split depth are accepted unconverged and trigger a
        :class:`RuntimeWarning`.

        Parameters
        ----------
        pixel_ids : ndarray, shape (m,)
            Indices into *x_flat* / *y_flat*.
        x_flat, y_flat : ndarray, shape (n,)
            Flattened pixel-centre coordinates for *all* pixels in the grid.
        z0, z1 : float
            Line-of-sight integration limits.

        Returns
        -------
        ndarray, shape (m,)
            Surface density (projected integral) for each requested pixel.
        """
        n = pixel_ids.size
        if n == 0:
            return np.empty(0, dtype=float)

        xp = x_flat[pixel_ids]
        yp = y_flat[pixel_ids]
        total = np.zeros(n, dtype=float)
        converged = np.ones(n, dtype=bool)
        length = z1 - z0
        max_depth = self._max_split_depth()
        pidx, a, b, fa, fm, fb, s, depth = self._initialize_frontier(xp, yp, z0, z1)
        done = False
        while pidx.size > 0 and not done:
            pidx, a, b, fa, fm, fb, s, depth, done = self._adaptive_iteration(
                xp=xp,
                yp=yp,
                total=total,
                converged=converged,
                length=length,
                max_depth=max_depth,
                pidx=pidx,
                a=a,
                b=b,
                fa=fa,
                fm=fm,
                fb=fb,
                s=s,
                depth=depth,
            )

        if not np.all(converged):
            warnings.warn(
                f"{(~converged).sum()} pixel(s) reached nz_max={self.nz_max} "
                "before meeting the tolerance target.",
                RuntimeWarning,
                stacklevel=3,
            )
        return total

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> np.ndarray:
        """
        Compute the 2-D projected surface-density map.

        Returns
        -------
        ndarray, shape (resolution, resolution)
            Surface-density values on the *xy* pixel grid.
        """
        x_flat, y_flat = self._build_pixel_grid()
        pixel_ids = np.arange(x_flat.size, dtype=np.int32)
        sigma = self._integrate_pixels(
            pixel_ids=pixel_ids,
            x_flat=x_flat,
            y_flat=y_flat,
            z0=self.z_range[0],
            z1=self.z_range[1],
        )
        return sigma.reshape(self.resolution, self.resolution)
