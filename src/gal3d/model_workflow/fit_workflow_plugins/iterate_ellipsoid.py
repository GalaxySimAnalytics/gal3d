"""
Iterative ellipsoidal shape estimation workflow plugin for Gal3D.
This module provides a fitting workflow that estimates the ellipsoidal shape of a structure
using an iterative mass moment method.
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

if TYPE_CHECKING:
    from gal3d.analyzer import Gal3DAnalyzer
    from gal3d.point import Particles

logger = logging.getLogger("gal3d.fit_workflow_plugins")

#Type alias
ArrayI = NDArray[np.integer[Any]]


def _periodic_diff(a: float, b: float, period: float = 2 * np.pi) -> float:
    """
    Minimal absolute difference between two angles on a circle.

    Parameters
    ----------
    a, b : float
        Angles in radians.
    period : float, optional
        Period of the angles (default is 2π).

    Returns
    -------
    float
        Minimal absolute difference between a and b, accounting for periodicity.
    """
    d = (a - b + 0.5 * period) % period - 0.5 * period
    return float(np.abs(d))

class IterateEllipsoidWorkflow(FitWorkflowBase):
    """
    Workflow for estimating ellipsoidal shape using iterative mass moment method.
    Accepts either Particles or Gal3DAnalyzer as input.
    """

    @staticmethod
    def condition(obj: Union["Gal3DAnalyzer", "Particles"]) -> bool:

        if type(obj).__name__ == "Particles":
            logger.debug("Select IterateEllipsoidWorkflow for Particles")
            return True
        else:
            raise TypeError("Unsupported object type")

    def _prepare_bins(
        self,
        r: np.ndarray,
        rmin: float,
        rmax: float,
        nbins: int,
        bins: str) -> tuple[np.ndarray, np.ndarray]:
        def equal_bins(r: np.ndarray, N: int) -> np.ndarray:
            sorted_r = np.sort(r[(r >= rmin) & (r <= rmax)])
            return np.append(
                [sorted_r[i * int(len(sorted_r) / N):(1 + i) * int(len(sorted_r) / N)][0] for i in range(N)],
                sorted_r[-1])
        if bins == "equal":
            full_bins = equal_bins(r, nbins * 2)
            bin_edges = full_bins[0:nbins * 2 + 1:2]
            rbins = full_bins[1:nbins * 2 + 1:2]
        elif bins == "log":
            bin_edges = np.geomspace(rmin, rmax, nbins + 1)
            rbins = np.sqrt(bin_edges[:-1] * bin_edges[1:])
        elif bins == "lin":
            bin_edges = np.linspace(rmin, rmax, nbins + 1)
            rbins = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        return bin_edges.astype(float), rbins.astype(float)

    def _iterate_shell(
        self,
        i: int, rbins: np.ndarray, bin_edges: np.ndarray, pos: np.ndarray, mass: np.ndarray, r: np.ndarray,
        stru: StructureCore,
        max_iterations: int,
        tol: float,
        is_enclosed: bool = False,
        weight_method: Literal["r2", "rell2"] | None = None
    ) -> tuple[np.ndarray, np.ndarray, int, float, float, np.ndarray, np.ndarray]:
        a = np.ones(3) * rbins[i]
        a2 = np.zeros(3)
        a2[0] = np.inf
        R = np.identity(3)
        R_prev = np.identity(3)
        iteration_counter = 0
        err = tol + 1
        ellipse_mass = np.zeros(3, dtype=float)
        while (err > tol) and (iteration_counter < max_iterations):
            R_prev = R.copy()
            a2 = a.copy()
            if not is_enclosed: # shell
                mult = bin_edges[[i, i + 1]] / np.prod(a) ** (1 / 3)
                shell_idx = cast("ArrayI", np.where((r > a[-1] * mult[0]) & (r < a[0] * mult[1]))[0])
            else: # enclosed
                mult = bin_edges[i + 1] / np.prod(a) ** (1 / 3)
                shell_idx = cast("ArrayI", np.where(r < a[0] * mult)[0])
            if shell_idx.size == 0:
                break
            shell_pos, shell_mass = pos[shell_idx], mass[shell_idx]
            new_shape = (a[0], 1 - a[1] / a[0], 1 - a[2] / a[1])
            new_ang = stru._coordinate.mat_to_angle(R)  # type: ignore
            d = stru.quick_f_ray_d(*new_ang, *new_shape, pos=shell_pos)[0]
            ellipse_idx = cast("ArrayI", np.where(np.abs(d) < 0.5 * (bin_edges[i + 1] - bin_edges[i]))[0] if not is_enclosed else np.where(d < 0)[0])
            if ellipse_idx.size == 0:
                break
            ellipse_pos, ellipse_mass = shell_pos[ellipse_idx], shell_mass[ellipse_idx] # type: ignore
            if weight_method == "r2":
                eff_mass =ellipse_mass / r[shell_idx][ellipse_idx] ** 2
            elif weight_method == "rell2":
                eff_mass = ellipse_mass / d[ellipse_idx] ** 2
            else:
                eff_mass = ellipse_mass
            abc, axes = abc_vect(ellipse_pos, eff_mass)
            R2 = np.array(axes)
            a_new = np.sqrt(np.abs(abc) * 3)
            div = (np.prod(a) / np.prod(a_new)) ** (1 / 3)
            a = a_new * div
            R = R2
            if np.linalg.det(R) < 0:
                R = -R
            iteration_counter += 1
            err = (np.abs(a[1] / a[0] - a2[1] / a2[0]) + np.abs(a[-1] / a[0] - a2[2] / a2[0])) * 0.5
        if iteration_counter > 0:
            a_prev = a2
        else:
            a_prev = a.copy()
            R_prev = R.copy()
        volume = (bin_edges[i + 1] ** 3) if is_enclosed else (bin_edges[i + 1] ** 3 - bin_edges[i] ** 3)
        ellipsoid_density = np.sum(ellipse_mass) / (4.0 / 3.0 * np.pi * volume)
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

        bin_edges, rbins = self._prepare_bins(r, rmin, rmax, nbins, bins)
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
