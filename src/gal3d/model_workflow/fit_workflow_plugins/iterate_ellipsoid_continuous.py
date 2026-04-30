"""
Iterative ellipsoidal shape estimation for continuous density sources.

This module implements the self-consistent angular-weighted shape tensor
method for finding isodensity ellipsoidal shells in a continuous density
field :math:`\\rho(\\mathbf{r})`.

Theoretical Background
----------------------
For a triaxial ellipsoid with semi-axes :math:`(a, b, c)`, define the
ellipsoidal coordinate:

.. math::

    m^2 = \\frac{x^2}{a^2} + \\frac{y^2}{b^2} + \\frac{z^2}{c^2}

An **isodensity ellipsoid** is a surface :math:`m = 1` on which
:math:`\\rho = \\text{const}`.

The **shape tensor** on the surface is:

.. math::

    S_{ij} = \\frac{\\oint \\rho(\\mathbf{r})\\, x_i\\, x_j\\, w\\, d\\sigma}
                   {\\oint \\rho(\\mathbf{r})\\, w\\, d\\sigma}

where :math:`d\\sigma` is the surface element and :math:`w` is a weight
function.

Weighting Scheme
~~~~~~~~~~~~~~~~
A naive area weighting :math:`w = 1` introduces a **geometric bias**: on a
prolate ellipsoid the poles are over-sampled relative to the equator in
solid-angle terms, systematically biasing the inferred axis ratios toward
rounder values.

The correct choice is :math:`w = dA / |\\nabla m|`, which equals the
**solid-angle element**:

.. math::

    \\frac{dA}{|\\nabla m|}
    = \\frac{\\sqrt{(bc)^2\\sin^2\\theta\\cos^2\\phi +
                    (ac)^2\\sin^2\\theta\\sin^2\\phi +
                    (ab)^2\\cos^2\\theta}}
             {\\sqrt{\\sin^2\\theta\\cos^2\\phi/a^2 +
                    \\sin^2\\theta\\sin^2\\phi/b^2 +
                    \\cos^2\\theta/c^2}}\\, d(\\cos\\theta)\\, d\\phi
    = abc \\cdot d(\\cos\\theta)\\, d\\phi

The :math:`abc` prefactor cancels in the ratio :math:`S_{ij}`, so the
effective weight is simply the **flat angular measure**
:math:`d\\Omega = d(\\cos\\theta)\\, d\\phi`.

Self-Consistency Condition
~~~~~~~~~~~~~~~~~~~~~~~~~~
With angular weighting, for any isodensity ellipsoid (regardless of the
radial density profile), the shape tensor satisfies **exactly**:

.. math::

    S_{ij} \\propto \\mathrm{diag}(a^2,\\, b^2,\\, c^2)

in the principal frame.  Therefore the new semi-axes are obtained directly
from the eigenvalues:

.. math::

    a_i^{\\mathrm{new}} = \\sqrt{\\lambda_i} \\times s, \\qquad
    s = \\left(\\frac{abc}{\\lambda_1^{1/2}\\lambda_2^{1/2}\\lambda_3^{1/2}}\\right)^{1/3}

where :math:`s` enforces volume conservation :math:`(abc)^{1/3} = r`.

Algorithm
---------
For each equivalent radius :math:`r = (abc)^{1/3}`:

1. **Initialise** :math:`(a,b,c) = (r,r,r)`, :math:`R = I_3`.
2. **Evaluate** the density-weighted shape tensor :math:`S` on the current
   ellipsoid surface using Gauss-Legendre quadrature (nodes in
   :math:`\\cos\\theta` and :math:`\\phi`).
3. **Diagonalise** :math:`S`; take eigenvectors as the new principal axes
   (rotation matrix :math:`R`) and :math:`\\sqrt{\\lambda_i}` as raw axis
   lengths.
4. **Damp** the axis update:
   :math:`a \\leftarrow (1-\\alpha)\\,a + \\alpha\\,a^{\\mathrm{new}}`
   (default :math:`\\alpha = 0.5`), then renormalise to conserve volume.
5. **Repeat** from step 2 until
   :math:`\\tfrac{1}{2}|\\Delta(b/a)| + \\tfrac{1}{2}|\\Delta(c/a)| < \\epsilon`.

References
----------
.. [1] Franx, M., Illingworth, G., & de Zeeuw, T. (1991), ApJ, 383, 112.
.. [2] Zemp, M. et al. (2011), ApJS, 197, 30.
       https://doi.org/10.1088/0067-0049/197/2/30

Examples
--------
Fit isodensity ellipsoid shells to a :class:`~gal3d.density.DensitySource`
``source`` over the radial range :math:`[0.5, 20]` kpc::

    from gal3d.model_workflow.fit_workflow_plugins.iterate_ellipsoid_continuous import (
        IterateEllipsoidDensity,
    )

    workflow = IterateEllipsoidDensity()

    result = workflow(
        source,
        r=np.geomspace(0.5, 20.0, 40),
        max_iterations=30,
        tol=1e-3,
        n_sample=1024,
        method="gauss_legendre",
        log=None,
        log_dynamic_range=2.0,
    )

Alternatively, supply an explicit radius array::

    import numpy as np
    result = workflow(source, r=np.geomspace(0.5, 20.0, 40))

The returned :class:`~gal3d.optimization.result.ModelResult` contains the
best-fit axis length ``a``, ellipticities ``eps_ab`` and ``eps_bc``, the
three orientation angles ``ang1``, ``ang2``, ``ang3``, and per-parameter
iteration uncertainties.
"""


import logging
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.polynomial.legendre import leggauss

from gal3d.field.spherical_field.spherical_vector import SphVector
from gal3d.model_workflow.fit_workflow import FitInput, FitWorkflowBase
from gal3d.optimization.result import ModelResult
from gal3d.point.util import solve_eigenvalues

from .util import (
    EllipsoidResultBuilder,
)

if TYPE_CHECKING:
    from gal3d.density import DensitySource

logger = logging.getLogger("gal3d.fit_workflow_plugins")


SamplingMethod = Literal["fibonacci", "gauss_legendre", "healpix"]

def _shape_tensor_to_axes(
    S: np.ndarray,
    a_old: np.ndarray,
    volume_conserve: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert angular-weighted shape tensor S → new semi-axes + rotation matrix.

    With angular (dΩ) weighting the self-consistency condition is exact:

        S_ij ∝ diag(a², b², c²) in the principal frame

    so new axes = sqrt(eigenvalues) × volume-conservation scale.
    """

    abc2, rot_mat = solve_eigenvalues(S, sort_descending=True)
    new_axes = np.sqrt(np.abs(abc2))

    if np.linalg.det(rot_mat) < 0:
        rot_mat = -rot_mat

    if volume_conserve:
        vol_old = float(np.prod(a_old))
        vol_new = float(np.prod(new_axes))
        if vol_new > 0:
            new_axes = new_axes * (vol_old / vol_new) ** (1.0 / 3.0)

    return new_axes, rot_mat

# ===========================================================================
# Continuous density: shape tensor on ellipsoid surface
# ===========================================================================

class EllipsoidShellShapeTensor:
    """
    Density-weighted angular shape tensor on an ellipsoid surface.

    Evaluates

    .. math::

        S_{ij} = \\frac{\\oint \\rho(\\mathbf{r})\\, x_i x_j\\, d\\Omega}
                       {\\oint \\rho(\\mathbf{r})\\, d\\Omega}

    Two sampling backends share a uniform ``n_sample`` API:

    ``"gauss_legendre"``
        Tensor-product Gauss-Legendre quadrature in
        :math:`(\\cos\\theta, \\phi)`.  ``n_sample`` is converted to
        ``ntheta × nphi`` with ``ntheta = max(8, int(sqrt(n_sample/2)))``
        and ``nphi = 2 * ntheta``, giving ``≈ n_sample`` total nodes.
        GL quadrature nodes and weights are precomputed in ``__init__``.

    ``"fibonacci"``
        Quasi-uniform sphere sampling via :class:`~gal3d.field.spherical_field.spherical_vector.SphVector`.
        ``n_sample`` points are placed on the unit sphere; the Voronoi cell
        areas serve as the :math:`d\\Omega` weights.  The ``SphVector``
        instance is cached (singleton) across calls.
    ``"healpix"``
        Equal-area HEALPix sampling.  Requires the optional ``healpy`` package.

    Parameters
    ----------
    n_sample : int, optional, default 512
        Target number of sample points.
    method : {"fibonacci", "gauss_legendre", "healpix"}, optional
        Sampling method.  Default is ``"fibonacci"``.
    sigma_clip : float | None, optional, default 3.0
        If not ``None``, winsorise density samples that deviate more than
        ``sigma_clip`` weighted standard deviations from the weighted mean.
    log : bool | None, optional
        If ``True``, apply logarithmic weighting to the density samples, else use linear weighting.
        If ``None`` (default), automatically enable log weighting if the density dynamic range exceeds ``log_dynamic_range``.
    log_dynamic_range : float, optional, default 2.0
        Dynamic range threshold for automatic log weighting.  Defined 1., e.g., as :math:`\\log_{10}(\\rho_\\mathrm{max}/\\rho_\\mathrm{min})`.
    """

    def __init__(
        self,
        n_sample: int = 512,
        method: SamplingMethod = "fibonacci",
        sigma_clip: float | None = 3.0,
        log: bool | None = None,
        log_dynamic_range: float = 2.0,
    ) -> None:
        self.n_sample = n_sample
        self.method   = method
        self.sigma_clip = sigma_clip
        self.log = log
        self.log_dynamic_range = log_dynamic_range

        if method == "gauss_legendre":
            ntheta = max(4, int(np.sqrt(n_sample / 2)))
            nphi   = max(8, 2 * ntheta)
            self._ntheta = ntheta
            self._nphi   = nphi
            # Precompute GL (Gauss-Legendre) nodes (fixed for this instance)
            ct_nodes, w_ct = leggauss(ntheta)
            t_phi,   w_phi = leggauss(nphi)
            self._ct_nodes   = ct_nodes
            self._w_ct       = w_ct
            self._phi_nodes  = np.pi * (t_phi + 1.0)
            self._w_phi      = np.pi * w_phi
        elif method == "healpix":
            self._hp_pos, self._hp_area = self._build_healpix(n_sample)
        else:
            self._sv = SphVector(n_sample=n_sample, method=method, verbose=False)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ellipsoid_points(
        unit_pos: np.ndarray,           # (N, 3) on unit sphere
        a: float, b: float, c: float,
        R: np.ndarray,                  # (3, 3)
    ) -> np.ndarray:
        """Scale unit-sphere points to the ellipsoid surface and rotate to lab frame."""
        return (unit_pos * np.array([a, b, c])) @ R

    @staticmethod
    def _sigma_clip_rho(
        rho: np.ndarray,
        W: np.ndarray,
        sigma: float,
    ) -> np.ndarray:
        """
        Winsorise outlier density samples by replacing them with the inlier
        weighted mean, preserving full angular (dΩ) coverage.

        Outliers are samples whose density deviates more than ``sigma``
        weighted standard deviations from the weighted mean.  Their density
        is replaced by the weighted mean of the inliers so that the angular
        measure W remains intact and the shape tensor is not angularly biased.

        Parameters
        ----------
        rho : ndarray of shape (N,)
            Sampled density values.
        W : ndarray of shape (N,)
            Angular quadrature weights (dΩ).  Not modified.
        sigma : float
            Clip threshold in units of weighted standard deviation.

        Returns
        -------
        rho_clipped : ndarray of shape (N,)
            Density array with outlier values replaced by the inlier
            weighted mean.
        """
        wsum = W.sum()
        if wsum <= 0:
            return rho
        mu  = (rho * W).sum() / wsum
        var = ((rho - mu) ** 2 * W).sum() / wsum
        std = np.sqrt(var)
        if std == 0.0:
            return rho

        inlier = np.abs(rho - mu) <= sigma * std
        inlier_wsum = W[inlier].sum()
        if inlier_wsum <= 0:
            return rho
        mu_inlier = (rho[inlier] * W[inlier]).sum() / inlier_wsum

        rho_clipped = rho.copy()
        rho_clipped[~inlier] = mu_inlier
        return rho_clipped

    @staticmethod
    def _shape_tensor_from_samples(
        pos_lab: np.ndarray,
        rho: np.ndarray,
        W: np.ndarray,
        rho_weight: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """Compute angular-weighted shape tensor and true mean density.

        rho_weight only affects the shape tensor calculation; the mean density is always the true weighted mean of the original rho values.
        """
        if rho_weight is None:
            rho_weight = rho

        rho_weight = np.asarray(rho_weight, dtype=float)
        rho_weight = np.where(np.isfinite(rho_weight) & (rho_weight > 0.0), rho_weight, 0.0)
        rho_W = rho_weight * W
        total_weight = rho_W.sum()
        if not np.isfinite(total_weight) or total_weight <= 0.0:
            raise ValueError("Invalid density weights: total tensor weight is not positive.")

        S = np.zeros((3, 3))
        for i in range(3):
            for j in range(i, 3):
                val = (rho_W * pos_lab[:, i] * pos_lab[:, j]).sum() / total_weight
                S[i, j] = val
                S[j, i] = val

        valid = np.isfinite(rho) & np.isfinite(W) & (W > 0)
        mean_rho = float((rho[valid] * W[valid]).sum() / W[valid].sum()) if np.any(valid) else 0.0
        return S, mean_rho

    @staticmethod
    def _build_healpix(n_sample: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Build equal-area HEALPix angular quadrature nodes.

        Requires the optional ``healpy`` package.
        """
        try:
            import healpy as hp
        except ImportError as exc:
            raise ImportError(
                "method='healpix' requires the optional package 'healpy'. "
                "Install it with: pip install healpy"
            ) from exc

        nside_float = np.sqrt(max(int(n_sample), 12) / 12.0)
        nside_pow = int(2 ** np.round(np.log2(nside_float)))
        nside = max(1, nside_pow)

        npix = hp.nside2npix(nside)
        pix = np.arange(npix)

        theta, phi = hp.pix2ang(nside, pix, nest=False)
        st = np.sin(theta)

        pos = np.column_stack([
            st * np.cos(phi),
            st * np.sin(phi),
            np.cos(theta),
        ])

        area = np.full(npix, 4.0 * np.pi / npix, dtype=float)
        return pos, area

    # ------------------------------------------------------------------

    def __call__(
        self,
        source: "DensitySource",
        a: float,
        b: float,
        c: float,
        rotation_matrix: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """
        Compute the shape tensor on the ellipsoid surface.

        Parameters
        ----------
        source : DensitySource
            Object with a ``_evaluate_density(pos)`` method.
        a, b, c : float
            Current semi-axes of the trial ellipsoid.
        rotation_matrix : ndarray of shape (3, 3), optional
            Rotation from ellipsoid frame to lab frame.  Defaults to :math:`I_3`.

        Returns
        -------
        S : ndarray of shape (3, 3)
            Angular-weighted shape tensor in the **lab** frame.
        mean_rho : float
            Angular-averaged volumetric density on the ellipsoid surface.
        """
        if self.method == "gauss_legendre":
            return self._compute_gauss_legendre(source, a, b, c, rotation_matrix)
        elif self.method == "healpix":
            return self._compute_healpix(source, a, b, c, rotation_matrix)
        else:
            return self._compute_fibonacci_voronoi(source, a, b, c, rotation_matrix)

    def spherical_harmonics_expansion(
        self,
        source: "DensitySource",
        a: float,
        b: float,
        c: float,
        lmax: int = 4,
        rotation_matrix: np.ndarray | None = None,
    ) -> dict[int, np.ndarray]:
        """
        Real spherical harmonics expansion of density on the ellipsoid shell.

        c_lm = ∫ ρ(n̂) Y_l^m(n̂) dΩ  ≈  Σ_i  ρ(r_i) · Y_l^m(θ_i, φ_i) · A_i

        where r_i = ellipsoid point along direction n̂_i, and A_i is the
        Voronoi area (dΩ weight) from the fibonacci sampling.

        Parameters
        ----------
        source : DensitySource
        a, b, c : float
            Semi-axes of the current trial ellipsoid.
        lmax : int
            Maximum harmonic degree.
        rotation_matrix : (3,3) ndarray, optional
            Ellipsoid-frame → lab-frame rotation.

        Returns
        -------
        coef : dict[int, ndarray]
            coef[l] has shape (2l+1,), ordered m = l, l-1, ..., -l.
        """
        from gal3d.field.spherical_field.spherical_harmonic import spherical_harmonics_in_real

        if self.method != "fibonacci":
            raise ValueError("spherical_harmonics_expansion requires method='fibonacci'")

        R = np.eye(3) if rotation_matrix is None else np.asarray(rotation_matrix)

        # Evaluate density on the ellipsoid surface (lab frame)
        pos_lab = self._ellipsoid_points(self._sv.pos, a, b, c, R)
        rho = source._evaluate_density(pos_lab)   # shape (N,)
        if self.sigma_clip is not None:
            rho = self._sigma_clip_rho(rho, self._sv.area, sigma=self.sigma_clip)

        # Angular coordinates come from the unit-sphere parameterization
        # _sv.sph columns: (r, phi, theta)  — phi ∈ [0,2π], theta ∈ [0,π]
        phi   = self._sv.sph[:, 1]   # azimuthal
        theta = self._sv.sph[:, 2]   # polar (colatitude)
        W     = self._sv.area        # Voronoi dΩ weights, sum = 4π

        coef: dict[int, np.ndarray] = {}
        for l in range(lmax + 1):
            coef[l] = np.array([
                np.sum(rho * spherical_harmonics_in_real(phi, theta, m, l) * W)
                for m in range(l, -l - 1, -1)   # m = l, l-1, ..., -l
            ])
        return coef

    # ------------------------------------------------------------------

    def _compute_gauss_legendre(
        self,
        source: "DensitySource",
        a: float, b: float, c: float,
        rotation_matrix: np.ndarray | None = None
        ) -> tuple[np.ndarray, float]:
        """ Gauss-Legendre quadrature backend for shape tensor evaluation. """
        R = np.eye(3) if rotation_matrix is None else np.asarray(rotation_matrix)
        ntheta, nphi = self._ntheta, self._nphi

        ct  = self._ct_nodes[:, None] * np.ones((ntheta, nphi))
        phi = self._phi_nodes[None, :] * np.ones((ntheta, nphi))
        st  = np.sqrt(np.clip(1.0 - ct**2, 0, None))

        unit_flat = np.stack(
            [(st * np.cos(phi)).ravel(), (st * np.sin(phi)).ravel(), ct.ravel()], axis=-1
        )
        W_flat  = (self._w_ct[:, None] * self._w_phi[None, :]).ravel()
        pos_lab = self._ellipsoid_points(unit_flat, a, b, c, R)
        rho     = source._evaluate_density(pos_lab)
        rho_weight, _ = self._effective_density_weight(
            rho,
            W_flat,
            sigma=self.sigma_clip,
            log=self.log,
            log_dynamic_range=self.log_dynamic_range,
        )
        return self._shape_tensor_from_samples(pos_lab, rho, W_flat, rho_weight=rho_weight)

    def _compute_fibonacci_voronoi(
        self,
        source: "DensitySource",
        a: float, b: float, c: float,
        rotation_matrix: np.ndarray | None = None
        ) -> tuple[np.ndarray, float]:
        """ Fibonacci sphere sampling backend for shape tensor evaluation. """
        R       = np.eye(3) if rotation_matrix is None else np.asarray(rotation_matrix)
        pos_lab = self._ellipsoid_points(self._sv.pos, a, b, c, R)
        rho     = source._evaluate_density(pos_lab)
        rho_weight, _ = self._effective_density_weight(
            rho,
            self._sv.area,
            sigma=self.sigma_clip,
            log=self.log,
            log_dynamic_range=self.log_dynamic_range,
        )
        return self._shape_tensor_from_samples(pos_lab, rho, self._sv.area, rho_weight=rho_weight)

    def _compute_healpix(
        self,
        source: "DensitySource",
        a: float, b: float, c: float,
        rotation_matrix: np.ndarray | None = None
        ) -> tuple[np.ndarray, float]:
        """HEALPix equal-area angular quadrature backend."""
        R = np.eye(3) if rotation_matrix is None else np.asarray(rotation_matrix)
        pos_lab = self._ellipsoid_points(self._hp_pos, a, b, c, R)
        rho = source._evaluate_density(pos_lab)
        rho_weight, _ = self._effective_density_weight(
            rho,
            self._hp_area,
            sigma=self.sigma_clip,
            log=self.log,
            log_dynamic_range=self.log_dynamic_range,
        )
        return self._shape_tensor_from_samples(pos_lab, rho, self._hp_area, rho_weight=rho_weight)

    @staticmethod
    def _weighted_quantile(
        values: np.ndarray,
        weights: np.ndarray,
        q: float,
    ) -> float:
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        if not np.any(mask):
            return np.nan

        values = values[mask]
        weights = weights[mask]
        order = np.argsort(values)
        values = values[order]
        weights = weights[order]
        cdf = np.cumsum(weights)
        cdf /= cdf[-1]
        return float(np.interp(q, cdf, values))

    @classmethod
    def _effective_density_weight(
        cls,
        rho: np.ndarray,
        W: np.ndarray,
        sigma: float | None,
        log: bool | None = None,
        log_dynamic_range: float = 2.0,
        tiny: float | None = None,
    ) -> tuple[np.ndarray, bool]:
        """
        Build robust positive density weights for the shape tensor.

        Parameters
        ----------
        rho : ndarray
            Raw density samples.
        W : ndarray
            Angular quadrature weights.
        sigma : float or None
            Clip threshold.  If None, no clipping is applied.
        log : bool or None
            False:
                Use rho-space clipping and rho-space tensor weights.
            True:
                Clip in log(rho) space, then use log1p-compressed positive
                weights for the tensor iteration.
            None:
                Automatically choose log=True when the central 90 percent
                density dynamic range exceeds ``log_dynamic_range`` dex.
        log_dynamic_range : float
            Dynamic range threshold in dex for automatic log mode.

        Returns
        -------
        rho_weight : ndarray
            Positive effective weights used only for the shape tensor.
        use_log : bool
            Whether log mode was used.
        """
        rho = np.asarray(rho, dtype=float)
        W = np.asarray(W, dtype=float)
        tiny = float(np.finfo(np.float64).tiny) if tiny is None else float(tiny)

        rho_safe = np.where(np.isfinite(rho) & (rho > 0.0), rho, tiny)
        log10rho = np.log10(rho_safe)

        if log is None:
            q05 = cls._weighted_quantile(log10rho, W, 0.05)
            q95 = cls._weighted_quantile(log10rho, W, 0.95)
            use_log = bool(
                np.isfinite(q05)
                and np.isfinite(q95)
                and (q95 - q05 > log_dynamic_range)
            )
        else:
            use_log = bool(log)

        if not use_log:
            if sigma is None:
                return rho_safe, False
            return cls._sigma_clip_rho(rho_safe, W, sigma=sigma), False

        logrho = np.log(rho_safe)

        if sigma is not None:
            med = cls._weighted_quantile(logrho, W, 0.50)
            mad = cls._weighted_quantile(np.abs(logrho - med), W, 0.50)
            if np.isfinite(med) and np.isfinite(mad) and mad > 0.0:
                scale = 1.4826 * mad
                logrho = np.clip(logrho, med - sigma * scale, med + sigma * scale)

        rho_clip = np.exp(logrho)

        rho_scale = cls._weighted_quantile(rho_clip, W, 0.50)
        if not np.isfinite(rho_scale) or rho_scale <= 0.0:
            rho_scale = np.nanmax(rho_clip)
        if not np.isfinite(rho_scale) or rho_scale <= 0.0:
            return np.ones_like(rho_safe), True

        rho_weight = np.log1p(rho_clip / max(rho_scale, tiny))
        rho_weight = np.maximum(rho_weight, tiny)

        return rho_weight, True

class IterateEllipsoidDensity(FitWorkflowBase, EllipsoidResultBuilder):
    """
    Iterative isodensity ellipsoid fitting for continuous density sources.

    Operates on :class:`~gal3d.density.DensitySource` objects (or a
    :class:`~gal3d.analyzer.Gal3DAnalyzer` whose ``model`` attribute is a
    ``DensitySource``).

    The algorithm uses the **angular-weighted shape tensor** on each trial
    ellipsoid surface.  The angular weighting
    :math:`w = dA/|\\nabla m| \\propto d\\Omega` removes the geometric
    bias present in naive area weighting and yields the exact
    self-consistency condition:

    .. math::

        S_{ij} \\propto \\mathrm{diag}(a^2,\\, b^2,\\, c^2)

    for any isodensity ellipsoid, regardless of the radial density profile.
    New semi-axes therefore follow directly from :math:`a_i \\propto
    \\sqrt{\\lambda_i}` with **no approximation**.

    See Also
    --------
    IterateEllipsoidParticles : Discrete-particle counterpart using the
        iterative inertia tensor method.

    Notes
    -----
    Convergence is assessed by:

    .. math::

        \\epsilon = \\tfrac{1}{2}\\left|\\Delta(b/a)\\right|
                  + \\tfrac{1}{2}\\left|\\Delta(c/a)\\right| < \\texttt{tol}

    Parameter uncertainties reported in the ``ModelResult`` are the
    absolute changes in each parameter between the **last two iterations**,
    providing a practical estimate of the numerical convergence error.
    """

    @staticmethod
    def condition(obj: FitInput) -> bool:
        """
        Return ``True`` if *obj* is a ``DensitySource`` or an analyser
        wrapping one.
        """
        from gal3d.density import DensitySource
        # accept a bare DensitySource or an Analyzer whose model is one
        if isinstance(obj, DensitySource):
            logger.debug("Select IterateEllipsoidDensityWorkflow for DensitySource")
            return True
        if hasattr(obj, "model") and isinstance(obj.model, DensitySource):
            logger.debug("Select IterateEllipsoidDensityWorkflow for Gal3DAnalyzer")
            return True
        return False

    # ------------------------------------------------------------------
    # Single-shell iteration
    # ------------------------------------------------------------------

    def _iterate_shell(
        self,
        source: "DensitySource",
        abc_init: np.ndarray,
        *,
        max_iterations: int,
        tol: float,
        volume_conserve: bool,
        damping: float,
        rot_init: np.ndarray | None = None,
        n_sample: int = 512,
        method: SamplingMethod = "fibonacci",
        sigma_clip: float | None = 3.0,
        log: bool | None = None,
        log_dynamic_range: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float, float]:
        """
        Iterate one radial shell to convergence.

        Parameters
        ----------
        source : DensitySource
            Object with a ``_evaluate_density(pos)`` method, used to evaluate the density on each trial ellipsoid surface.
        abc_init : ndarray of shape (3,)
            Initial semi-axes.
        max_iterations : int
            Maximum number of iterations.
        tol : float
            Convergence tolerance.
        volume_conserve : bool
            Whether to conserve volume during iteration. If false, the major axis is fixed.
        damping : float
            Damping coefficient :math:`\\alpha \\in (0, 1]`.
        rot_init : ndarray of shape (3, 3) or None
            Initial rotation matrix; if None, defaults to identity.
        n_sample : int
            Number of sample points on the ellipsoid surface.
        method : {"fibonacci", "gauss_legendre", "healpix"}
            Sampling method for shape tensor evaluation.
        sigma_clip : float | None
            If not ``None``, winsorise density samples that deviate more than
            ``sigma_clip`` weighted standard deviations from the weighted mean.

        Returns
        -------
        a : ndarray of shape (3,)
            Converged semi-axes :math:`(a \\geq b \\geq c)`.
        rot : ndarray of shape (3, 3)
            Rotation matrix; columns are principal axes in lab frame.
        a_prev : ndarray of shape (3,)
            Semi-axes from the penultimate iteration (for uncertainty).
        rot_prev : ndarray of shape (3, 3)
            Rotation matrix from the penultimate iteration (for uncertainty).
        n_iter_done : int
            Number of iterations actually performed.
        err : float
            Final axis-ratio convergence measure.
        mean_rho : float
            Angular-averaged **volumetric** density :math:`\\rho` evaluated
            on the converged ellipsoid surface.  Directly usable as the
            isodensity value — no further conversion needed.
        """
        abc        = abc_init.copy()
        rot      = np.eye(3) if rot_init is None else np.asarray(rot_init, dtype=float)
        abc_prev   = abc_init.copy()
        rot_prev = rot.copy()
        err      = tol + 1.0
        mean_rho = 0.0
        iteration_counter = 0

        shape_tensor = EllipsoidShellShapeTensor(
            n_sample=n_sample, method=method, sigma_clip=sigma_clip,
            log=log, log_dynamic_range=log_dynamic_range,
        )

        while (err > tol) and (iteration_counter < max_iterations):
            abc_prev   = abc.copy()
            rot_prev = rot.copy()

            S, mean_rho = shape_tensor(
                source, float(abc[0]), float(abc[1]), float(abc[2]),
                rotation_matrix=rot,
            )
            abc_new, rot_new = _shape_tensor_to_axes(S, abc, volume_conserve)

            err = self._axis_ratio_error(abc_new, abc)
            abc_damped = (1.0 - damping) * abc + damping * abc_new

            abc = self._to_new_ellipsoid(abc, abc_damped, volume_conservation=volume_conserve)
            rot = rot_new

            iteration_counter += 1

        return abc, rot, abc_prev, rot_prev, iteration_counter, err, mean_rho


    def _fit_single(self, obj: FitInput, r: float, **kwargs: Any) -> ModelResult:
        from gal3d.density import DensitySource

        sd = obj if isinstance(obj, DensitySource) else obj.density_source

        init_parameters = kwargs.get("init_parameters",{}) # initial guess for a,b,c.  Default is spherical.

        max_iterations = kwargs.get("max_iterations", 30)
        tol = kwargs.get("tol", 1e-3)
        volume_conserve = kwargs.get("volume_conserve", False)
        damping = kwargs.get("damping", 0.8)
        n_sample = kwargs.get("n_sample", 512)
        method = kwargs.get("method", "fibonacci")
        sigma_clip = kwargs.get("sigma_clip", 3.0)

        log = kwargs.get("log", None)
        log_dynamic_range = kwargs.get("log_dynamic_range", 1.0)

        # a, b, c
        abc = np.ones(3, dtype=float)*r
        abc_init = np.ones(3,dtype=float)
        rot_init = None
        if init_parameters:
            abc_init[1] = (1-init_parameters.get("eps_ab",0.)) * abc_init[0]
            abc_init[2] = (1-init_parameters.get("eps_bc",0.)) * abc_init[1]
            ang1 = float(init_parameters.get("ang1", 0.0))
            ang2 = float(init_parameters.get("ang2", 0.0))
            ang3 = float(init_parameters.get("ang3", 0.0))
            from scipy.spatial.transform import Rotation as _R
            rot_init = _R.from_euler("zyx", [ang1, ang2, ang3]).as_matrix()

        abc = self._to_new_ellipsoid(abc, abc_init, volume_conservation=volume_conserve) # type: ignore

        abc, rot, abc_prev, rot_prev, n_done, err, mean_rho = self._iterate_shell(
            sd, abc_init=abc, max_iterations=max_iterations, tol=tol,
            volume_conserve=volume_conserve, damping=damping,
            rot_init=rot_init,
            n_sample=n_sample, method=method,
            sigma_clip=sigma_clip,
            log=log, log_dynamic_range=log_dynamic_range
            )
        return self._build_model_result(abc, rot, abc_prev, rot_prev, n_done, err, mean_rho)
