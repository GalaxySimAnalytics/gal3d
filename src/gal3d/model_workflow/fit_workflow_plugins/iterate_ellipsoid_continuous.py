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
        rmin=0.5,
        rmax=20.0,
        nbins=40,
        bins="log",
        tol=1e-3,
        n_iter=20,
        volume_conserve=True,
        damping=0.5,
        ntheta=64,
        nphi=128,
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
from typing import TYPE_CHECKING, Any, Literal, Union

import numpy as np
from numpy.polynomial.legendre import leggauss

from gal3d.field.spherical_field.spherical_vector import SphVector
from gal3d.model_workflow.fit_workflow import FitWorkflowBase
from gal3d.optimization.result import ModelResult

from .util import (
    EllipsoidResultBuilder,
)

if TYPE_CHECKING:
    from gal3d.analyzer import Gal3DAnalyzer
    from gal3d.density import DensitySource
    from gal3d.point import Particles

logger = logging.getLogger("gal3d.fit_workflow_plugins")

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
    eigvals, eigvecs = np.linalg.eigh(S)            # ascending order

    raw_axes = np.sqrt(np.clip(eigvals, 0, None))
    idx      = np.argsort(raw_axes)[::-1]           # descending
    new_axes = raw_axes[idx]
    rot_mat  = eigvecs[:, idx]

    if np.linalg.det(rot_mat) < 0:
        rot_mat[:, -1] *= -1

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

    Parameters
    ----------
    n_sample : int, optional, default 512
        Target number of sample points.
    method : {"fibonacci", "gauss_legendre"}, optional
        Sampling method.  Default is ``"fibonacci"``.
    """

    def __init__(
        self,
        n_sample: int = 512,
        method: Literal["fibonacci", "gauss_legendre"] = "fibonacci",
    ) -> None:
        self.n_sample = n_sample
        self.method   = method

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
        return (unit_pos * np.array([a, b, c])) @ R.T

    @staticmethod
    def _shape_tensor_from_samples(
        pos_lab: np.ndarray,    # (N, 3)
        rho: np.ndarray,        # (N,)
        W: np.ndarray,          # (N,)
    ) -> tuple[np.ndarray, float]:
        """Compute angular-weighted shape tensor and mean density from flat samples."""
        rho_W        = rho * W
        total_weight = rho_W.sum()

        S = np.zeros((3, 3))
        for i in range(3):
            for j in range(i, 3):
                val     = (rho_W * pos_lab[:, i] * pos_lab[:, j]).sum() / total_weight
                S[i, j] = val
                S[j, i] = val

        mean_rho = float(rho_W.sum() / W.sum()) if W.sum() > 0 else 0.0
        return S, mean_rho

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
        return self._shape_tensor_from_samples(pos_lab, rho, W_flat)

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
        return self._shape_tensor_from_samples(pos_lab, rho, self._sv.area)

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
    def condition(obj: Union["Gal3DAnalyzer", "Particles"]) -> bool:
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
        n_sample: int = 512,
        method: Literal["fibonacci", "gauss_legendre"] = "fibonacci",
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
        n_sample : int
            Number of sample points on the ellipsoid surface.
        method : {"fibonacci", "gauss_legendre"}
            Sampling method for shape tensor evaluation.

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
        rot      = np.eye(3)
        abc_prev   = abc_init.copy()
        rot_prev = rot.copy()
        err      = tol + 1.0
        mean_rho = 0.0
        iteration_counter = 0

        shape_tensor = EllipsoidShellShapeTensor(n_sample=n_sample, method=method)

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


    def _fit_single(self, obj: Union["Gal3DAnalyzer", "DensitySource"], r: float, **kwargs: Any) -> ModelResult:
        from gal3d.density import DensitySource

        sd = obj if isinstance(obj, DensitySource) else obj.density_source # type: ignore

        init_parameters = kwargs.get("init_parameters",{}) # initial guess for a,b,c.  Default is spherical.

        max_iterations = kwargs.get("max_iterations", 20)
        tol = kwargs.get("tol", 1e-3)
        volume_conserve = kwargs.get("volume_conserve", False)
        damping = kwargs.get("damping", 0.8)
        n_sample = kwargs.get("n_sample", 512)
        method = kwargs.get("method", "fibonacci") # sampling method for shape tensor evaluation; default is fibonacci sampling on the sphere

        # a, b, c
        abc = np.ones(3, dtype=float)*r
        abc_init = np.ones(3,dtype=float)
        if init_parameters:
            abc_init[1] = (1-init_parameters.get("eps_ab",0.)) * abc_init[0]
            abc_init[2] = (1-init_parameters.get("eps_bc",0.)) * abc_init[1]

        abc = self._to_new_ellipsoid(abc, abc_init, volume_conservation=volume_conserve) # type: ignore

        abc, rot, abc_prev, rot_prev, n_done, err, mean_rho = self._iterate_shell(
            sd, abc_init=abc, max_iterations=max_iterations, tol=tol,
            volume_conserve=volume_conserve, damping=damping,
            n_sample=n_sample, method=method)
        return self._build_model_result(abc, rot, abc_prev, rot_prev, n_done, err, mean_rho)
