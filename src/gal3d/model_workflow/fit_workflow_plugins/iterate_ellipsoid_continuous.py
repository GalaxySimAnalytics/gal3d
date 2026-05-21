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
   ellipsoid surface using the selected angular quadrature backend (nodes in
   :math:`\\cos\\theta` and :math:`\\phi`).
3. **Diagonalise** :math:`S`; take eigenvectors as the new principal axes
   (rotation matrix :math:`R`) and :math:`\\sqrt{\\lambda_i}` as raw axis
   lengths.
4. **Damp** the axis update:
   :math:`a \\leftarrow (1-\\alpha)\\,a + \\alpha\\,a^{\\mathrm{new}}`
   (default :math:`\\alpha = 0.8`), then renormalise to conserve volume.
5. **Repeat** from step 2 until
   :math:`\\tfrac{1}{2}|\\Delta(b/a)| + \\tfrac{1}{2}|\\Delta(c/a)| < \\epsilon`.

For a detailed discussion of the iterative shape-tensor method, see Zemp et al. (2011) [1]_.

References
----------
.. [1] Zemp, M. et al. (2011), ApJS, 197, 30.
       https://doi.org/10.1088/0067-0049/197/2/30

Examples
--------
Fit isodensity ellipsoid shells to a :class:`~gal3d.density.DensitySource`
``source`` over the radial range :math:`[0.5, 20]` kpc::

    from gal3d.model_workflow.fit_workflow_plugins.iterate_ellipsoid_continuous import IterateEllipsoidDensity

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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import numpy as np
from numpy.polynomial.legendre import leggauss

from gal3d.field.spherical_field.spherical_vector import SphVector
from gal3d.model_workflow.fit_workflow import FitInput, FitWorkflowBase
from gal3d.optimization.result import ModelResult
from gal3d.point.util import solve_eigenvalues

from .util import EllipsoidResultBuilder

if TYPE_CHECKING:
    from gal3d.density import DensitySource

logger = logging.getLogger("gal3d.fit_workflow_plugins")


SamplingMethod = Literal["fibonacci", "gauss_legendre", "healpix"]


@dataclass(frozen=True)
class DensityWeightConfig:
    """
    Configuration for robust density weights used in the shape tensor.

    Parameters
    ----------
    sigma_clip : float or None, optional
        Clipping threshold. If ``None``, no clipping is applied.
    log : bool or None, optional
        If ``True``, use logarithmically compressed positive weights. If
        ``False``, use clipped density values directly. If ``None``, switch
        to log mode when the central 90 percent density range exceeds
        ``log_dynamic_range`` dex.
    log_dynamic_range : float, optional
        Dynamic range threshold in dex for automatic log weighting.
    """

    sigma_clip: float | None = 3.0
    log: bool | None = None
    log_dynamic_range: float = 1.0


@dataclass(frozen=True)
class ShellTensorConfig:
    """
    Configuration for angular shell tensor evaluation.

    Parameters
    ----------
    n_sample : int, optional
        Target number of angular samples.
    method : {"fibonacci", "gauss_legendre", "healpix"}, optional
        Angular sampling backend.
    weight : DensityWeightConfig, optional
        Robust density-weight configuration.
    """

    n_sample: int = 512
    method: SamplingMethod = "fibonacci"
    weight: DensityWeightConfig = field(default_factory=DensityWeightConfig)


@dataclass(frozen=True)
class EllipsoidIterationConfig:
    """
    Iteration controls for one continuous-density ellipsoid shell.

    Parameters
    ----------
    max_iterations : int, optional
        Maximum shape-tensor iterations.
    tol : float, optional
        Axis-ratio convergence tolerance.
    volume_conserve : bool, optional
        If ``True``, preserve ellipsoid volume; if ``False``, preserve the
        major axis length and update only axis ratios.
    damping : float, optional
        Damping coefficient for axis updates.
    """

    max_iterations: int = 30
    tol: float = 1e-3
    volume_conserve: bool = False
    damping: float = 0.8


@dataclass
class EllipsoidIterationState:
    """Mutable state of one shell iteration."""

    axes: np.ndarray
    rotation: np.ndarray
    previous_axes: np.ndarray
    previous_rotation: np.ndarray
    n_iter: int = 0
    err: float = np.inf
    mean_rho: float = 0.0


class AngularSampler(Protocol):
    """Protocol for angular quadrature nodes on the unit sphere."""

    pos: np.ndarray
    weights: np.ndarray

    def angular_coordinates(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return ``(phi, theta)`` if available, otherwise ``None``."""


class FibonacciAngularSampler:
    """
    Fibonacci/Voronoi angular sampler backed by :class:`SphVector`.

    The positions are unit-sphere directions and ``weights`` are Voronoi
    angular areas whose sum is approximately :math:`4\\pi`.
    """

    def __init__(self, n_sample: int, method: SamplingMethod = "fibonacci") -> None:
        self._sv = SphVector(n_sample=n_sample, method=method, verbose=False)
        self.pos = self._sv.pos
        self.weights = self._sv.area

    def angular_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        phi = self._sv.sph[:, 1]
        theta = self._sv.sph[:, 2]
        return phi, theta


class GaussLegendreAngularSampler:
    """
    Tensor-product Gauss-Legendre angular quadrature in
    :math:`(\\cos\\theta, \\phi)`.

    ``n_sample`` is mapped to ``ntheta * nphi`` using
    ``ntheta = max(4, int(sqrt(n_sample / 2)))`` and
    ``nphi = max(8, 2 * ntheta)``.
    """

    def __init__(self, n_sample: int) -> None:
        ntheta = max(4, int(np.sqrt(n_sample / 2)))
        nphi = max(8, 2 * ntheta)

        ct_nodes, w_ct = leggauss(ntheta)
        t_phi, w_phi = leggauss(nphi)

        phi_nodes = np.pi * (t_phi + 1.0)
        phi_weights = np.pi * w_phi

        ct = np.broadcast_to(ct_nodes[:, None], (ntheta, nphi))
        phi = np.broadcast_to(phi_nodes[None, :], (ntheta, nphi))
        st = np.sqrt(np.clip(1.0 - ct**2, 0.0, None))

        self.pos = np.column_stack([(st * np.cos(phi)).ravel(), (st * np.sin(phi)).ravel(), ct.ravel()])
        self.weights = (w_ct[:, None] * phi_weights[None, :]).ravel()
        self._phi = phi.ravel()
        self._theta = np.arccos(np.clip(ct.ravel(), -1.0, 1.0))

    def angular_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        return self._phi, self._theta


class HealpixAngularSampler:
    """
    Equal-area HEALPix angular sampler.

    Requires the optional ``healpy`` package.
    """

    def __init__(self, n_sample: int) -> None:
        try:
            import healpy as hp
        except ImportError as exc:
            raise ImportError(
                "method='healpix' requires the optional package 'healpy'. Install it with: pip install healpy"
            ) from exc

        nside_float = np.sqrt(max(int(n_sample), 12) / 12.0)
        nside = max(1, int(2 ** np.round(np.log2(nside_float))))

        npix = hp.nside2npix(nside)
        pix = np.arange(npix)

        theta, phi = hp.pix2ang(nside, pix, nest=False)
        st = np.sin(theta)

        self.pos = np.column_stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)])
        self.weights = np.full(npix, 4.0 * np.pi / npix, dtype=float)
        self._phi = phi
        self._theta = theta

    def angular_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        return self._phi, self._theta


def build_angular_sampler(config: ShellTensorConfig) -> AngularSampler:
    """Create the angular sampler requested by ``config``."""
    if config.method == "gauss_legendre":
        return GaussLegendreAngularSampler(config.n_sample)
    if config.method == "healpix":
        return HealpixAngularSampler(config.n_sample)
    return FibonacciAngularSampler(config.n_sample, method=config.method)


def _shape_tensor_to_axes(
    S: np.ndarray, a_old: np.ndarray, volume_conserve: bool = True
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
class DensityWeightPolicy:
    """Robust positive density weights for shape-tensor iteration."""

    @staticmethod
    def sigma_clip_rho(rho: np.ndarray, weights: np.ndarray, sigma: float) -> np.ndarray:
        wsum = weights.sum()
        if wsum <= 0:
            return rho

        mu = (rho * weights).sum() / wsum
        var = ((rho - mu) ** 2 * weights).sum() / wsum
        std = np.sqrt(var)
        if std == 0.0:
            return rho

        inlier = np.abs(rho - mu) <= sigma * std
        inlier_wsum = weights[inlier].sum()
        if inlier_wsum <= 0:
            return rho

        mu_inlier = (rho[inlier] * weights[inlier]).sum() / inlier_wsum
        rho_clipped = rho.copy()
        rho_clipped[~inlier] = mu_inlier
        return rho_clipped

    @staticmethod
    def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float | np.ndarray) -> float | np.ndarray:
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        if not np.any(mask):
            if np.ndim(q) == 0:
                return np.nan
            return np.full_like(np.asarray(q, dtype=float), np.nan)

        values = values[mask]
        weights = weights[mask]
        order = np.argsort(values)
        values = values[order]
        weights = weights[order]

        cdf = np.cumsum(weights)
        cdf /= cdf[-1]
        out = np.interp(q, cdf, values)
        if np.ndim(q) == 0:
            return float(out)
        return out

    @classmethod
    def effective_density_weight(
        cls, rho: np.ndarray, weights: np.ndarray, config: DensityWeightConfig, tiny: float | None = None
    ) -> tuple[np.ndarray, bool]:
        """
        Build effective positive density weights.

        The returned weights affect only the shape tensor; reported mean
        density should still be computed from the original density samples.
        """
        rho = np.asarray(rho, dtype=float)
        weights = np.asarray(weights, dtype=float)
        tiny = float(np.finfo(np.float64).tiny) if tiny is None else float(tiny)

        rho_safe = np.where(np.isfinite(rho) & (rho > 0.0), rho, tiny)
        log10rho = np.log10(rho_safe)

        if config.log is None:
            q05, q95 = cast("tuple[float, float]", cls.weighted_quantile(log10rho, weights, np.array([0.05, 0.95])))
            use_log = bool(np.isfinite(q05) and np.isfinite(q95) and (q95 - q05 > config.log_dynamic_range))
        else:
            use_log = bool(config.log)

        if not use_log:
            if config.sigma_clip is None:
                return rho_safe, False
            return cls.sigma_clip_rho(rho_safe, weights, config.sigma_clip), False

        logrho = np.log(rho_safe)

        if config.sigma_clip is not None:
            med = cls.weighted_quantile(logrho, weights, 0.50)
            mad = cls.weighted_quantile(np.abs(logrho - med), weights, 0.50)
            if np.isfinite(med) and np.isfinite(mad) and mad > 0.0:
                scale = 1.4826 * mad
                logrho = np.clip(logrho, med - config.sigma_clip * scale, med + config.sigma_clip * scale)

        rho_clip = np.exp(logrho)
        rho_scale = cls.weighted_quantile(rho_clip, weights, 0.50)
        if not np.isfinite(rho_scale) or rho_scale <= 0.0:
            rho_scale = np.nanmax(rho_clip)
        if not np.isfinite(rho_scale) or rho_scale <= 0.0:
            return np.ones_like(rho_safe), True

        rho_weight = np.log1p(rho_clip / max(float(rho_scale), tiny))
        rho_weight = np.maximum(rho_weight, tiny)
        return rho_weight, True


class EllipsoidShellShapeTensor:
    """
    Density-weighted angular shape tensor on an ellipsoid surface.

    The sampler supplies unit-sphere directions and angular quadrature
    weights. This class maps those directions to the trial ellipsoid,
    evaluates the density source, builds robust density weights, and returns

    .. math::

        S_{ij} = \\frac{\\sum_k \\rho_k^\\mathrm{eff} x_{k,i}x_{k,j} W_k}
                      {\\sum_k \\rho_k^\\mathrm{eff} W_k}.

    The returned mean density is always computed from the original density
    samples, not from the robust/log-compressed tensor weights.
    """

    def __init__(self, sampler: AngularSampler, weight_config: DensityWeightConfig | None = None) -> None:
        self.sampler = sampler
        self.weight_config = weight_config or DensityWeightConfig()

    @staticmethod
    def ellipsoid_points(unit_pos: np.ndarray, axes: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
        """Scale unit-sphere points to the ellipsoid surface and rotate to lab frame."""
        return (unit_pos * axes) @ rotation_matrix

    @staticmethod
    def shape_tensor_from_samples(
        pos_lab: np.ndarray, rho: np.ndarray, weights: np.ndarray, rho_weight: np.ndarray | None = None
    ) -> tuple[np.ndarray, float]:
        """Compute the angular-weighted shape tensor and true mean density."""
        if rho_weight is None:
            rho_weight = rho

        rho_weight = np.asarray(rho_weight, dtype=float)
        rho_weight = np.where(np.isfinite(rho_weight) & (rho_weight > 0.0), rho_weight, 0.0)

        rho_weights = rho_weight * weights
        total_weight = rho_weights.sum()
        if not np.isfinite(total_weight) or total_weight <= 0.0:
            raise ValueError("Invalid density weights: total tensor weight is not positive.")

        weighted_pos = pos_lab * rho_weights[:, None]
        tensor = weighted_pos.T @ pos_lab / total_weight
        tensor = 0.5 * (tensor + tensor.T)

        valid = np.isfinite(rho) & np.isfinite(weights) & (weights > 0)
        mean_rho = float((rho[valid] * weights[valid]).sum() / weights[valid].sum()) if np.any(valid) else 0.0
        return tensor, mean_rho

    def __call__(
        self, source: "DensitySource", axes: np.ndarray, rotation_matrix: np.ndarray | None = None
    ) -> tuple[np.ndarray, float]:
        """
        Compute the shape tensor on the ellipsoid surface.

        Parameters
        ----------
        source : DensitySource
            Object with a ``_evaluate_density(pos)`` method.
        axes : ndarray of shape (3,)
            Current semi-axes of the trial ellipsoid.
        rotation_matrix : ndarray of shape (3, 3), optional
            Rotation from ellipsoid frame to lab frame.

        Returns
        -------
        S : ndarray of shape (3, 3)
            Angular-weighted shape tensor in the lab frame.
        mean_rho : float
            Angular-averaged volumetric density on the ellipsoid surface.
        """
        rotation = np.eye(3) if rotation_matrix is None else np.asarray(rotation_matrix)
        axes = np.asarray(axes, dtype=float)

        pos_lab = self.ellipsoid_points(self.sampler.pos, axes, rotation)
        rho = source._evaluate_density(pos_lab)

        rho_weight, _ = DensityWeightPolicy.effective_density_weight(rho, self.sampler.weights, self.weight_config)
        return self.shape_tensor_from_samples(pos_lab, rho, self.sampler.weights, rho_weight=rho_weight)

    def spherical_harmonics_expansion(
        self, source: "DensitySource", axes: np.ndarray, lmax: int = 4, rotation_matrix: np.ndarray | None = None
    ) -> dict[int, np.ndarray]:
        """
        Real spherical-harmonic expansion of density on the ellipsoid shell.

        This requires a sampler that exposes angular coordinates.
        """
        from gal3d.field.spherical_field.spherical_harmonic import spherical_harmonics_in_real

        coords = self.sampler.angular_coordinates()
        if coords is None:
            raise ValueError("spherical_harmonics_expansion requires angular coordinates")

        phi, theta = coords
        rotation = np.eye(3) if rotation_matrix is None else np.asarray(rotation_matrix)
        axes = np.asarray(axes, dtype=float)

        pos_lab = self.ellipsoid_points(self.sampler.pos, axes, rotation)
        rho = source._evaluate_density(pos_lab)

        if self.weight_config.sigma_clip is not None:
            rho = DensityWeightPolicy.sigma_clip_rho(rho, self.sampler.weights, sigma=self.weight_config.sigma_clip)

        return {
            ell: np.array(
                [
                    np.sum(rho * spherical_harmonics_in_real(phi, theta, m, ell) * self.sampler.weights)
                    for m in range(ell, -ell - 1, -1)
                ]
            )
            for ell in range(lmax + 1)
        }


class IterateEllipsoidDensity(FitWorkflowBase, EllipsoidResultBuilder):
    """
    Iterative isodensity ellipsoid fitting for continuous density sources.

    Operates on :class:`~gal3d.density.DensitySource` objects (or a
    :class:`~gal3d.analyzer.Gal3DAnalyzer` whose ``density_source`` attribute is a
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

        # accept a bare DensitySource or an Analyzer whose density_source is one
        if isinstance(obj, DensitySource):
            logger.debug("Select IterateEllipsoidDensityWorkflow for DensitySource")
            return True
        if hasattr(obj, "density_source") and isinstance(obj.density_source, DensitySource):
            logger.debug("Select IterateEllipsoidDensityWorkflow for Gal3DAnalyzer")
            return True
        return False

    # ------------------------------------------------------------------
    # Single-shell iteration
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_density_source(obj: FitInput) -> "DensitySource":
        """Return the underlying ``DensitySource`` from accepted inputs."""
        from gal3d.density import DensitySource

        if isinstance(obj, DensitySource):
            return obj

        density_source = getattr(obj, "density_source", None)
        if isinstance(density_source, DensitySource):
            return density_source

        raise TypeError("Expected a DensitySource or an object wrapping one.")

    @staticmethod
    def _parse_iteration_config(kwargs: dict[str, Any]) -> EllipsoidIterationConfig:
        """Build iteration config from workflow keyword arguments."""
        return EllipsoidIterationConfig(
            max_iterations=int(kwargs.get("max_iterations", 30)),
            tol=float(kwargs.get("tol", 1e-3)),
            volume_conserve=bool(kwargs.get("volume_conserve", False)),
            damping=float(kwargs.get("damping", 0.8)),
        )

    @staticmethod
    def _parse_tensor_config(kwargs: dict[str, Any]) -> ShellTensorConfig:
        """Build shell tensor config from workflow keyword arguments."""
        return ShellTensorConfig(
            n_sample=int(kwargs.get("n_sample", 512)),
            method=kwargs.get("method", "fibonacci"),
            weight=DensityWeightConfig(
                sigma_clip=kwargs.get("sigma_clip", 3.0),
                log=kwargs.get("log", None),
                log_dynamic_range=float(kwargs.get("log_dynamic_range", 1.0)),
            ),
        )

    def _init_trial_ellipsoid(
        self, r: float, init_parameters: dict[str, Any] | None, volume_conserve: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build initial axes and rotation from optional warm-start parameters.
        """
        axes_ref = np.ones(3, dtype=float) * r
        ratio_axes = np.ones(3, dtype=float)
        rotation = np.eye(3, dtype=float)

        if init_parameters:
            ratio_axes[1] = (1.0 - init_parameters.get("eps_ab", 0.0)) * ratio_axes[0]
            ratio_axes[2] = (1.0 - init_parameters.get("eps_bc", 0.0)) * ratio_axes[1]

            ang1 = float(init_parameters.get("ang1", 0.0))
            ang2 = float(init_parameters.get("ang2", 0.0))
            ang3 = float(init_parameters.get("ang3", 0.0))

            from scipy.spatial.transform import Rotation as _Rotation

            rotation = _Rotation.from_euler("zyx", [ang1, ang2, ang3]).as_matrix()

        axes = self._to_new_ellipsoid(axes_ref, ratio_axes, volume_conservation=volume_conserve)
        return axes, rotation

    def _iterate_once(
        self,
        source: "DensitySource",
        tensor: EllipsoidShellShapeTensor,
        state: EllipsoidIterationState,
        config: EllipsoidIterationConfig,
    ) -> None:
        """Perform one shape-tensor iteration in-place."""
        state.previous_axes = state.axes.copy()
        state.previous_rotation = state.rotation.copy()

        shape_tensor, state.mean_rho = tensor(source, state.axes, rotation_matrix=state.rotation)

        axes_new, rotation_new = _shape_tensor_to_axes(shape_tensor, state.axes, config.volume_conserve)

        state.err = self._axis_ratio_error(axes_new, state.axes)
        axes_damped = (1.0 - config.damping) * state.axes + config.damping * axes_new

        state.axes = self._to_new_ellipsoid(state.axes, axes_damped, volume_conservation=config.volume_conserve)
        state.rotation = rotation_new
        state.n_iter += 1

    def _iterate_shell(
        self,
        source: "DensitySource",
        axes_init: np.ndarray,
        rotation_init: np.ndarray,
        iteration_config: EllipsoidIterationConfig,
        tensor_config: ShellTensorConfig,
    ) -> EllipsoidIterationState:
        """
        Iterate one radial shell to convergence.

        Parameters
        ----------
        source : DensitySource
            Continuous density source.
        axes_init : ndarray of shape (3,)
            Initial semi-axes.
        rotation_init : ndarray of shape (3, 3)
            Initial ellipsoid-frame to lab-frame rotation.
        iteration_config : EllipsoidIterationConfig
            Iteration controls.
        tensor_config : ShellTensorConfig
            Angular sampling and density-weight controls.

        Returns
        -------
        EllipsoidIterationState
            Final axes, rotation, previous state, iteration count, convergence
            error, and angular mean density.
        """
        sampler = build_angular_sampler(tensor_config)
        tensor = EllipsoidShellShapeTensor(sampler, tensor_config.weight)

        state = EllipsoidIterationState(
            axes=axes_init.copy(),
            rotation=np.asarray(rotation_init, dtype=float),
            previous_axes=axes_init.copy(),
            previous_rotation=np.asarray(rotation_init, dtype=float).copy(),
            err=iteration_config.tol + 1.0,
        )

        while state.err > iteration_config.tol and state.n_iter < iteration_config.max_iterations:
            self._iterate_once(source, tensor, state, iteration_config)

        return state

    def _fit_single(self, obj: FitInput, r: float, **kwargs: Any) -> ModelResult:
        """
        Fit one continuous-density isodensity ellipsoid shell.

        Parameters
        ----------
        obj : FitInput
            A ``DensitySource`` or an object exposing ``density_source`` or
            ``model`` as a ``DensitySource``.
        r : float
            Target equivalent radius.
        **kwargs
            Supports ``max_iterations``, ``tol``, ``volume_conserve``,
            ``damping``, ``n_sample``, ``method``, ``sigma_clip``, ``log``,
            ``log_dynamic_range``, and ``init_parameters``.

        Returns
        -------
        ModelResult
            Converged ellipsoid parameters and iteration uncertainties.
        """
        source = self._extract_density_source(obj)
        iteration_config = self._parse_iteration_config(kwargs)
        tensor_config = self._parse_tensor_config(kwargs)

        init_parameters = kwargs.get("init_parameters", {})
        axes_init, rotation_init = self._init_trial_ellipsoid(r, init_parameters, iteration_config.volume_conserve)

        state = self._iterate_shell(source, axes_init, rotation_init, iteration_config, tensor_config)

        return self._build_model_result(
            state.axes,
            state.rotation,
            state.previous_axes,
            state.previous_rotation,
            state.n_iter,
            state.err,
            state.mean_rho,
        )
