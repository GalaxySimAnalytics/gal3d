"""
Iterative ellipsoidal shape estimation workflow plugin for Gal3D.

This module implements an iterative mass–moment (inertia tensor) method to
estimate the 3‑D ellipsoidal shape of a particle distribution. The basic idea
is:

1. Select particles within a radial shell or enclosed ellipsoid.
2. Compute the (possibly weighted) inertia tensor of the selected particles.
3. Diagonalize the tensor to obtain the principal axes and axis lengths.
4. Update the trial ellipsoid and repeat from step 1 until convergence.

Two families of methods are supported:

* **Shell methods (S1–S3)** – particles are selected in an *ellipsoidal shell*.
* **Enclosed methods (E1–E3)** – particles are selected inside an *enclosed
  ellipsoid*.

Within each family, three weighting schemes :math:`w(r)` can be used for the
mass when computing the inertia tensor:

* ``None`` / unweighted  (``w = 1``)
* ``"r2"``     – spherical radius weighting :math:`w \\propto r^{-2}`
* ``"rell2"``  – ellipsoidal radius weighting
  :math:`w \\propto r_\\mathrm{ell}^{-2}`,
  where

  .. math::

      r_\\mathrm{ell}^2 = x^2 +
                         \frac{y^2}{(b/a)^2} +
                         \frac{z^2}{(c/a)^2}.

Both families and all weightings are available through the
:class:`IterateEllipsoidWorkflow`.

Notes
-----
This implementation is inspired by the iterative shape measurement in
``pynbody.analysis.halo.shape``(https://github.com/pynbody/pynbody/blob/master/pynbody/analysis/halo.py#L483)
and follows the ideas discussed in:

    Zemp, M., Gnedin, O. Y., Gnedin, N. Y., & Kravtsov, A. V. (2011),
    *On Determining the Shape of Matter Distributions*, ApJS, 197(2), 30,
    https://doi.org/10.1088/0067-0049/197/2/30


Examples
--------
A minimal usage example with a :class:`gal3d.point.Particles` instance
``particles``::

    from gal3d.model_workflow.fit_workflow_plugins.iterate_ellipsoid import (
        IterateEllipsoidWorkflow,
    )

    workflow = IterateEllipsoidWorkflow()

    # S1: ellipsoidal shell, unweighted
    result_s1 = workflow(
        particles,
        nbins=50,
        bins="equal",
        max_iterations=20,
        tol=1e-3,
        is_enclosed=False,        # shell
        weight_method=None,       # w = 1
    )

    # E3: enclosed ellipsoid, w ~ r_ell^{-2}
    result_e3 = workflow(
        particles,
        nbins=50,
        bins="equal",
        max_iterations=20,
        tol=1e-3,
        is_enclosed=True,         # enclosed ellipsoid
        weight_method="rell2",    # w ~ r_ell^{-2}
    )

The returned :class:`gal3d.optimization.result.ModelResult` contains the best‑fit
axis length ``a`` and axis ratios ``eps_ab``, ``eps_bc``, the three Euler
angles ``ang1``, ``ang2``, ``ang3``, and an estimate of their iteration
errors.

"""

import logging
from typing import TYPE_CHECKING, Any, Literal, Union, cast

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from gal3d.model_workflow.fit_workflow import FitWorkflowBase
from gal3d.optimization.optimizer import OptimizeResult
from gal3d.optimization.result import EmptyModelResult, ModelResult
from gal3d.point.util import abc_vect
from gal3d.shape import StructureCore

from .util import (
    _axis_ratio_error,
    _periodic_diff,
    _prepare_bins,
)

if TYPE_CHECKING:
    from gal3d.analyzer import Gal3DAnalyzer
    from gal3d.point import Particles

logger = logging.getLogger("gal3d.fit_workflow_plugins")

#Type alias
ArrayF = NDArray[np.floating[Any]]
ArrayI = NDArray[np.int_]

class IterateEllipsoidParticles(FitWorkflowBase):
    """
    Workflow for estimating ellipsoidal shape using an iterative mass–moment
    (inertia tensor) method.
    """

    @staticmethod
    def condition(obj: Union["Gal3DAnalyzer", "Particles"]) -> bool:

        if type(obj).__name__ == "Particles":
            logger.debug("Select IterateEllipsoidWorkflow for Particles")
            return True
        else:
            raise TypeError("Unsupported object type")

    def _select_shell_indices(
        self,
        r: ArrayF,
        a: ArrayF,
        bin_edges: ArrayF,
        i: int,
        is_enclosed: bool,
    ) -> tuple[ArrayF, ArrayI]:
        if not is_enclosed:
            mult = bin_edges[[i, i + 1]] / np.prod(a) ** (1.0 / 3.0)
            shell_idx = np.where((r > a[-1] * mult[0]) & (r < a[0] * mult[1]))[0]
        else:
            mult = bin_edges[i + 1] / np.prod(a) ** (1.0 / 3.0)
            shell_idx = np.where(r < a[0] * mult)[0]
        return mult, shell_idx

    def _select_ellipse_indices(
        self,
        d_ell: ArrayF,
        mult: ArrayF | float,
        is_enclosed: bool,
    ) -> ArrayI:
        if not is_enclosed:
            lo, hi = cast("ArrayF", mult)
            return np.where((d_ell > lo) & (d_ell < hi))[0]
        else:
            hi = cast("float", mult)
            return np.where(d_ell < hi)[0]

    def _apply_weight(
        self,
        weight_method: Literal["r2", "rell2"] | None,
        ellipse_mass: ArrayF,
        r_shell: ArrayF,
        d_ell: ArrayF,
    ) -> ArrayF:
        if weight_method == "r2":
            w = 1.0 / np.maximum(r_shell, 1e-9) ** 2
            return ellipse_mass * w
        if weight_method == "rell2":
            w = 1.0 / np.maximum(d_ell, 1e-9) ** 2
            return ellipse_mass * w
        return ellipse_mass

    @staticmethod
    def _sanitize_rotation(R: ArrayF) -> ArrayF:
        if np.linalg.det(R) < 0:
            R = -R
        return R

    @staticmethod
    def _update_axes(a_old: ArrayF, a_new_raw: ArrayF) -> ArrayF:
        a_new = np.sqrt(np.abs(a_new_raw) * 3.0)
        scale = (np.prod(a_old) / np.prod(a_new)) ** (1.0 / 3.0)
        return a_new * scale


    @staticmethod
    def _compute_density(
        ellipse_mass: ArrayF, bin_edges: ArrayF, i: int, is_enclosed: bool
    ) -> float:
        if is_enclosed:
            vol = 4.0 / 3.0 * np.pi * (bin_edges[i + 1] ** 3)
        else:
            vol = 4.0 / 3.0 * np.pi * (
                bin_edges[i + 1] ** 3 - bin_edges[i] ** 3
            )
        return float(np.sum(ellipse_mass) / vol)

    # --------- main iteration ----------
    def _iterate_shell(
        self,
        i: int, rbins: np.ndarray, bin_edges: np.ndarray, pos: np.ndarray, mass: np.ndarray, r: np.ndarray,
        stru: StructureCore,
        max_iterations: int,
        tol: float,
        is_enclosed: bool = False,
        weight_method: Literal["r2", "rell2"] | None = None
    ) -> tuple[np.ndarray, np.ndarray, int, float, float, np.ndarray, np.ndarray]:
        """
        Perform the iterative shape estimation for a single radial bin.

        Parameters
        ----------
        i : int
            Bin index.
        rbins : ndarray of float
            Representative radius of each bin.
        bin_edges : ndarray of float
            Bin edges of length ``nbins + 1``.
        pos : ndarray of float, shape (N, 3)
            Particle positions.
        mass : ndarray of float, shape (N,)
            Particle masses.
        r : ndarray of float, shape (N,)
            Spherical radii corresponding to ``pos``.
        stru : StructureCore
            Structure object providing ellipsoid geometry utilities.
        max_iterations : int
            Maximum number of iterations per bin.
        tol : float
            Convergence tolerance for axis ratios.
        is_enclosed : bool, optional
            If ``True``, use enclosed‑ellipsoid selection (E1–E3);
            if ``False``, use ellipsoidal shells (S1–S3).
        weight_method : {'r2', 'rell2'}, optional
            Weighting method when computing the inertia tensor:
            * ``None``   – unweighted (``w = 1``)
            * ``'r2'``   – spherical :math:`r^{-2}` weighting
            * ``'rell2'`` – ellipsoidal :math:`r_\\mathrm{ell}^{-2}` weighting

        Returns
        -------
        a : ndarray of float, shape (3,)
            Final semi‑axis lengths ``(a, b, c)``.
        R : ndarray of float, shape (3, 3)
            Final rotation matrix from principal‑axes frame to simulation frame.
        iteration_counter : int
            Number of iterations performed.
        err : float
            Scalar convergence measure based on axis‑ratio changes.
        ellipsoid_density : float
            Mean mass density in the shell/ellipsoid.
        a_prev : ndarray of float, shape (3,)
            Semi‑axis lengths from the previous iteration.
        R_prev : ndarray of float, shape (3, 3)
            Rotation matrix from the previous iteration.
        """
        a: ArrayF = np.ones(3, dtype=float) * rbins[i]
        a_prev: ArrayF = a.copy()
        R: ArrayF = np.identity(3, dtype=float)
        R_prev: ArrayF = R.copy()
        iteration_counter = 0
        err = tol + 1.0
        ellipse_mass: ArrayF = np.zeros(3, dtype=float)

        while (err > tol) and (iteration_counter < max_iterations):
            a_prev = a.copy()
            R_prev = R.copy()

            mult, shell_idx = self._select_shell_indices(r, a, bin_edges, i, is_enclosed)
            if shell_idx.size == 0:
                break

            shell_pos = pos[shell_idx]
            shell_mass = mass[shell_idx]

            new_shape = (a[0], 1 - a[1] / a[0], 1 - a[2] / a[1])
            new_ang = stru._coordinate.mat_to_angle(R)  # type: ignore
            d_ell = stru.quick_f_ray_d(*new_ang, *new_shape, pos=shell_pos)[0]

            ellipse_idx = self._select_ellipse_indices(d_ell, mult, is_enclosed)
            if ellipse_idx.size == 0:
                break

            ellipse_pos = shell_pos[ellipse_idx]
            ellipse_mass = shell_mass[ellipse_idx]

            eff_mass = self._apply_weight(weight_method, ellipse_mass, r[shell_idx][ellipse_idx], d_ell[ellipse_idx])

            abc, axes = abc_vect(ellipse_pos, eff_mass)
            R = self._sanitize_rotation(np.array(axes, dtype=float))
            a = self._update_axes(a, abc)

            iteration_counter += 1
            err = _axis_ratio_error(a, a_prev)

        ellipsoid_density = self._compute_density(ellipse_mass, bin_edges, i, is_enclosed)
        return a, R, iteration_counter, err, ellipsoid_density, a_prev, R_prev

    def _build_model_result(
        self,
        stru: StructureCore,
        a: np.ndarray,
        R: np.ndarray,
        a_prev: np.ndarray,
        R_prev: np.ndarray,
        iteration_counter: int,
        err: float,
        ellipsoid_density: float
        ) -> ModelResult:
        """Build ModelResult from the final iteration of the ellipsoid fitting."""
        stru.parameters["a"] = a[0]
        stru.parameters.get_parameter("a").err = np.abs(a[0] - a_prev[0])
        stru.parameters["eps_ab"] = 1 - a[1] / a[0]
        stru.parameters.get_parameter("eps_ab").err = np.abs((a[1] / a[0]) - (a_prev[1] / a_prev[0]))
        stru.parameters["eps_bc"] = 1 - a[2] / a[1]
        stru.parameters.get_parameter("eps_bc").err = np.abs((a[2] / a[1]) - (a_prev[2] / a_prev[1]))

        ang = stru._coordinate.mat_to_angle(R)  # type: ignore
        stru.parameters["ang1"], stru.parameters["ang2"], stru.parameters["ang3"] = ang
        ang_prev = stru._coordinate.mat_to_angle(R_prev)  # type: ignore
        stru.parameters.get_parameter("ang1").err = _periodic_diff(ang[0], ang_prev[0])
        stru.parameters.get_parameter("ang2").err = _periodic_diff(ang[1], ang_prev[1])
        stru.parameters.get_parameter("ang3").err = _periodic_diff(ang[2], ang_prev[2])

        params = stru.parameters.deepcopy()
        params.add_info(parameter=ellipsoid_density)
        optimize_result = OptimizeResult(params=params, fun=None, start_fun=None, start_params=None,
                                         n_iterations=iteration_counter, cost=err)
        return ModelResult(stru, optimize_result, params)

    def __call__(
        self,
        obj: Union["Gal3DAnalyzer", "Particles"],
        nbins: int = 100,
        rmin: float | None = None,
        rmax: float | None = None,
        bins: Literal["equal", "log", "lin"] = "equal",
        max_iterations: int = 10,
        tol: float = 1e-3,
        is_enclosed: bool = False,
        weight_method: Literal["r2", "rell2"] | None = None,
        *args: Any,
        **kwargs: Any
    ) -> ModelResult:
        """
        Fit ellipsoidal shape using iterative mass moment method.

        Parameters
        ----------
        obj : Gal3DAnalyzer or Particles
            Input object containing particle data.
        nbins : int, optional
            Number of radial bins to use (default is 100).
        rmin : float, optional
            Minimum radius for binning. If None, set to rmax / 1E3.
        rmax : float, optional
            Maximum radius for binning. If None, set to maximum particle radius.
        bins : {'equal', 'log', 'lin'}, optional
            Radial binning scheme, by default ``'equal'``:
            * ``'equal'`` – equal number of particles per bin
            * ``'log'``   – logarithmically spaced in radius
            * ``'lin'``   – linearly spaced in radius
        max_iterations : int, optional
            Maximum number of iterations per radial bin (default is 10).
        tol : float, optional
            Convergence tolerance on axis‑ratio changes, by default ``1e-3``.
        is_enclosed : bool, optional
            If ``False`` (default) use ellipsoidal shells (S1–S3). If
            ``True`` use enclosed ellipsoids (E1–E3).
        weight_method : {'r2', 'rell2'}, optional
            Weighting method :math:`w(r)` applied to particle masses when
            computing the inertia tensor:
            * ``None``   – unweighted (``w = 1``; S1 or E1)
            * ``'r2'``   – spherical :math:`r^{-2}` (S2 or E2)
            * ``'rell2'`` – ellipsoidal :math:`r_\\mathrm{ell}^{-2}` (S3 or E3)
        *args, **kwargs :
            Additional positional and keyword arguments (currently unused).

        Returns
        -------
        ModelResult
            Summed model result over all radial bins, or EmptyModelResult if no valid results.
        """
        if hasattr(obj, "particles"):
            particles = obj.particles
        else:
            particles = obj

        if rmax is None:
            rmax = particles.r.max()
        if rmin is None:
            rmin = rmax / 1E3

        assert max_iterations > 0
        assert tol > 0
        assert rmin >= 0
        assert rmax > rmin
        assert nbins > 0
        assert np.sum((particles.r >= rmin) & (particles.r < rmax)) > nbins * 2, "Not enough particles per bin. Consider reducing nbins or adjusting rmin/rmax."
        if bins not in ["equal", "log", "lin"]:
            logger.warning("Unknown binning method '%s', defaulting to 'equal'", bins)
            bins = "equal"

        r = particles.r
        pos = particles.pos
        mass = particles.mass

        bin_edges, rbins = _prepare_bins(r, rmin, rmax, nbins, bins)
        model_results: list[ModelResult] = []
        stru = StructureCore("RotateOnly", "Ellipsoid")

        for i in tqdm(range(nbins), desc="Iterative ellipsoid shape"):
            a, R, iteration_counter, err, ellipsoid_density, a_prev, R_prev = self._iterate_shell(
                i, rbins, bin_edges, pos, mass, r, stru, max_iterations, tol, is_enclosed, weight_method
            )
            model_result = self._build_model_result(stru, a, R, a_prev, R_prev, iteration_counter, err, ellipsoid_density)
            model_results.append(model_result)

        if model_results:
            return sum(model_results[1:], model_results[0])
        else:
            logger.warning("No valid model results found.")
            return EmptyModelResult()
