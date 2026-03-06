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
from tqdm import tqdm

from gal3d.model_workflow.fit_workflow import FitWorkflowBase
from gal3d.optimization.result import EmptyModelResult, ModelResult
from gal3d.shape import StructureCore

from .util import (
    EllipsoidResultBuilder,
    _prepare_bins,
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

def _ellipsoid_shell_shape_tensor(
    source: "DensitySource",
    a: float,
    b: float,
    c: float,
    rotation_matrix: np.ndarray | None = None,
    *,
    ntheta: int = 64,
    nphi: int = 128,
) -> tuple[np.ndarray, float]:
    """
    Density-weighted angular shape tensor on an ellipsoid surface.

    Evaluates

    .. math::

        S_{ij} = \\frac{\\oint \\rho(\\mathbf{r})\\, x_i x_j\\, d\\Omega}
                       {\\oint \\rho(\\mathbf{r})\\, d\\Omega}

    using a tensor-product Gauss-Legendre grid in
    :math:`(\\cos\\theta, \\phi)` – the angular solid-angle measure
    :math:`d\\Omega = d(\\cos\\theta)\\,d\\phi` (see module docstring for
    the derivation of why this equals :math:`dA/|\\nabla m|`).

    Parameters
    ----------
    source : DensitySource
        Object with a ``_evaluate_density(pos)`` method.
    a, b, c : float
        Current semi-axes of the trial ellipsoid.
    rotation_matrix : ndarray of shape (3, 3), optional
        Rotation from ellipsoid frame to lab frame.  Defaults to
        :math:`I_3`.
    ntheta : int, optional
        Number of Gauss-Legendre nodes in :math:`\\cos\\theta`.
    nphi : int, optional
        Number of Gauss-Legendre nodes in :math:`\\phi`.

    Returns
    -------
    S : ndarray of shape (3, 3)
        Angular-weighted shape tensor in the **lab** frame.
    mean_rho : float
        Angular-averaged density on the ellipsoid surface:

        .. math::

            \\bar{\\rho} = \\frac{\\oint \\rho(\\mathbf{r})\\, d\\Omega}
                                  {\\oint d\\Omega}

        For a true isodensity ellipsoid this equals the constant density
        value :math:`\\rho_0` on that surface.  For a non-isodensity
        surface it is the solid-angle-weighted mean of the local
        **volumetric** density :math:`\\rho` evaluated on the surface —
        i.e. it already has units of :math:`[\\mathrm{mass/volume}]` and
        requires no further conversion.
    """
    R = np.eye(3) if rotation_matrix is None else np.asarray(rotation_matrix)

    ct_nodes, w_ct = leggauss(ntheta)
    t_phi,   w_phi = leggauss(nphi)
    phi_nodes = np.pi * (t_phi + 1.0)
    w_phi     = np.pi * w_phi           #type: ignore[assignment]

    # Full (ntheta, nphi) broadcast
    ct  = ct_nodes[:, None]  * np.ones((ntheta, nphi))
    phi = phi_nodes[None, :] * np.ones((ntheta, nphi))
    st  = np.sqrt(np.clip(1.0 - ct**2, 0, None))

    xe = a * st * np.cos(phi)
    ye = b * st * np.sin(phi)
    ze = c * ct

    # dA/|∇m| ∝ d(cosθ)dφ  — pure angular weight (no area Jacobian)
    W = w_ct[:, None] * w_phi[None, :]          # (ntheta, nphi)

    xyz_flat = np.stack([xe.ravel(), ye.ravel(), ze.ravel()], axis=-1)
    pos_lab  = xyz_flat @ R.T

    rho_flat = source._evaluate_density(pos_lab)
    rho      = rho_flat.reshape(ntheta, nphi)

    rho_W        = rho * W
    total_weight = rho_W.sum()

    x_lab = pos_lab[:, 0].reshape(ntheta, nphi)
    y_lab = pos_lab[:, 1].reshape(ntheta, nphi)
    z_lab = pos_lab[:, 2].reshape(ntheta, nphi)

    S        = np.zeros((3, 3))
    xyz_list = [x_lab, y_lab, z_lab]
    for i in range(3):
        for j in range(i, 3):
            val      = (rho_W * xyz_list[i] * xyz_list[j]).sum() / total_weight
            S[i, j]  = val
            S[j, i]  = val

    area_total = W.sum()
    mean_rho   = float(rho_W.sum() / area_total) if area_total > 0 else 0.0
    return S, mean_rho

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
        radius: float,
        a_init: np.ndarray,
        *,
        max_iterations: int,
        tol: float,
        volume_conserve: bool,
        damping: float,
        ntheta: int,
        nphi: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float, float]:
        """
        Iterate one radial shell to convergence.

        Parameters
        ----------
        source : DensitySource
        radius : float
            Equivalent radius :math:`r = (abc)^{1/3}` for this shell.
        a_init : ndarray of shape (3,)
            Initial semi-axes.
        max_iterations : int
            Maximum number of iterations.
        tol : float
            Convergence tolerance.
        volume_conserve : bool
        damping : float
            Damping coefficient :math:`\\alpha \\in (0, 1]`.
        ntheta, nphi : int
            Quadrature resolution.

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
        a        = a_init.copy()
        rot      = np.eye(3)
        a_prev   = a_init.copy()
        rot_prev = rot.copy()
        err      = tol + 1.0
        mean_rho = 0.0
        iteration_counter = 0

        while (err > tol) and (iteration_counter < max_iterations):
            a_prev   = a.copy()
            rot_prev = rot.copy()

            S, mean_rho = _ellipsoid_shell_shape_tensor(
                source, float(a[0]), float(a[1]), float(a[2]),
                rotation_matrix=rot,
                ntheta=ntheta, nphi=nphi,
            )
            a_new, rot_new = _shape_tensor_to_axes(S, a, volume_conserve)

            err = self._axis_ratio_error(a_new, a)
            a_damped = (1.0 - damping) * a + damping * a_new
            if volume_conserve:
                a_damped *= (radius**3 / np.prod(a_damped)) ** (1.0 / 3.0)

            a = a_damped
            rot = rot_new

            iteration_counter += 1

        return a, rot, a_prev, rot_prev, iteration_counter, err, mean_rho

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        obj: Union["Gal3DAnalyzer", "DensitySource"],
        r: np.ndarray | None = None,
        rmin: float | None = None,
        rmax: float | None = None,
        nbins: int = 100,
        bins: Literal["equal", "log", "lin"] = "log",
        tol: float = 1e-3,
        max_iterations: int = 20,
        volume_conserve: bool = True,
        damping: float = 0.8,
        ntheta: int = 32,
        nphi: int = 64,
        *args: Any,
        **kwargs: Any,
    ) -> ModelResult:
        """
        Fit isodensity ellipsoid shells to a continuous density source.

        Parameters
        ----------
        obj : DensitySource or Gal3DAnalyzer
            The density field to analyse.  If a ``Gal3DAnalyzer`` is
            passed, ``obj.model`` must be a ``DensitySource``.
        r : array_like of float, optional
            Explicit sequence of equivalent radii
            :math:`r = (abc)^{1/3}` [same units as ``pos``].
            When provided, ``rmin``, ``rmax``, ``nbins``, and ``bins``
            are ignored.
        rmin : float, optional
            Minimum equivalent radius.  Required when ``r`` is ``None``.
        rmax : float, optional
            Maximum equivalent radius.  Required when ``r`` is ``None``.
        nbins : int, optional
            Number of radial shells.  Used only when ``r`` is ``None``.
            Default is ``100``.
        bins : {'log', 'lin', 'equal'}, optional
            Radial spacing.  Used only when ``r`` is ``None``.

            * ``'log'``   – logarithmically spaced (default).
            * ``'lin'``   – linearly spaced.
            * ``'equal'`` – equal spacing in a reference particle sample.
        tol : float, optional
            Convergence tolerance :math:`\\epsilon` on the axis-ratio
            change.  Default is ``1e-3``.
        max_iterations : int, optional
            Maximum iterations per shell.  Default is ``20``.
        volume_conserve : bool, optional
            If ``True`` (default), keep :math:`(abc)^{1/3} = r` fixed
            throughout the iteration.
        damping : float, optional
            Step-damping coefficient :math:`\\alpha \\in (0, 1]`.
            ``1.0`` gives full Newton steps; ``0.8`` (default) improves
            stability for strongly non-spherical systems.
        ntheta : int, optional
            Gauss-Legendre nodes in :math:`\\cos\\theta`. Default ``32``.
        nphi : int, optional
            Gauss-Legendre nodes in :math:`\\phi`. Default ``64``.

        Returns
        -------
        ModelResult
            Summed result over all shells.  Each shell contributes the
            parameters ``a``, ``eps_ab``, ``eps_bc``, ``ang1``, ``ang2``,
            ``ang3`` with associated iteration uncertainties.  Returns
            :class:`~gal3d.optimization.result.EmptyModelResult` if no
            shell converges.

        Raises
        ------
        ValueError
            If neither ``r`` nor both ``rmin`` and ``rmax`` are supplied.

        Examples
        --------
        Logarithmically spaced shells between 0.5 and 20 kpc::

            result = workflow(source, rmin=0.5, rmax=20.0, nbins=40,
                              bins="log", tol=1e-3, max_iterations=20)

        Explicit radius array::

            import numpy as np
            result = workflow(source, r=np.geomspace(0.5, 20.0, 40))

        Access fitted axis ratio at each shell::

            import numpy as np
            eps_ab = 1.0 - np.array(result["b"]) / np.array(result["a"])
        """
        source: DensitySource = obj.model if hasattr(obj, "model") else obj  # type: ignore

        # --- build radius array ---
        if r is not None:
            radii = np.asarray(r, dtype=float)
        else:
            if rmin is None or rmax is None:
                raise ValueError("Provide either `r` or both `rmin` and `rmax`.")
            _, radii = _prepare_bins(np.zeros(0), rmin, rmax, nbins, bins)

        stru          = StructureCore("RotateOnly", "Ellipsoid")
        model_results : list[ModelResult] = []

        for i in tqdm(range(len(radii)), desc="Iterative ellipsoid (density)"):
            radius = float(radii[i])

            # --- initial guess: sphere, or rescaled previous shell ---
            if i == 0:
                a_init = np.ones(3) * radius
            else:
                prev       = model_results[-1]
                a_p        = float(prev["a"])
                eps_ab     = float(prev["eps_ab"])
                eps_bc     = float(prev["eps_bc"])
                a_prev_arr = np.array([
                    a_p,
                    a_p * (1.0 - eps_ab),
                    a_p * (1.0 - eps_ab) * (1.0 - eps_bc),
                ], dtype=float)
                scale  = (radius**3 / np.prod(a_prev_arr)) ** (1.0 / 3.0)
                a_init = a_prev_arr * scale

            a, rot, a_prev_axes, rot_prev, n_done, err, mean_rho = self._iterate_shell(
                source, radius, a_init,
                max_iterations=max_iterations, tol=tol,
                volume_conserve=volume_conserve,
                damping=damping,
                ntheta=ntheta, nphi=nphi,
            )
            model_results.append(
                self._build_model_result(stru, a, rot, a_prev_axes, rot_prev, n_done, err, mean_rho)
            )

        if model_results:
            return sum(model_results[1:], model_results[0])
        logger.warning("No valid model results found.")
        return EmptyModelResult()
