"""
This module provides the ``Gal3DAnalyzer`` class for analyzing and fitting 3D galaxy structures
using particle data, spherical field generation, geometric modeling, and optimization workflows.

Usage Example
-------------
>>> analyzer = Gal3DAnalyzer.analyze(pos, mass)
>>> result = analyzer.fit(num_step=100)
"""
import logging
from collections.abc import Iterable
from typing import Any

import numpy as np
from tqdm import tqdm

from .configuration import config
from .field import SphField
from .model_workflow.fit_workflow import FitWorkflow
from .optimization.optimizer import Optimizer, OptimizerBase
from .optimization.result import EmptyModelResult, ModelResult
from .point import Particles
from .shape import Structure3D
from .util.errors import FitDataError

logger = logging.getLogger("gal3d.analyzer")

config.general.set_optimal_thread_count(logger)
class Gal3DAnalyzer:
    """
    An analyzer for fitting galaxy 3D structures using particle, field, and shape information.

    Attributes
    ----------
    fit_workflow : dict of str to tuple of (Callable, Callable)
        A registry for fitting workflow conditions and their corresponding functions.
    particles : Particles
        The input particle data.
    field : SphField
        The spherical field generator.
    structure : Structure3D
        The 3D geometric structure model to fit.
    optimizer : OptimizerBase
        The optimization engine used for fitting.
    """

    fit_workflow = FitWorkflow()

    def __init__(
        self,
        particles: Particles,
        field: SphField,
        structure: Structure3D,
        optimizer: OptimizerBase,
    ):

        self.particles = particles
        self.field = field
        self.structure = structure
        self.optimizer = optimizer

    @classmethod
    def analyze(
        cls,
        pos: np.ndarray,
        mass: np.ndarray,
        recenter: bool = True,
        inner_frac: float = 0.9,
        **kwargs: Any
    ) -> "Gal3DAnalyzer":
        """
        Analyze the given particle data.

        Parameters
        ----------
        pos : np.ndarray
            The positions of the particles. Shape: (N, 3)
        mass : np.ndarray
            The masses of the particles. Shape: (N,)
        recenter : bool, optional
            Whether to recenter the particles' positions (default is True).
        inner_frac : float, optional
            inner_frac * center_value for inner boundary definition (default is 0.9).
        **kwargs
            Additional keyword arguments for analysis.
            inner : float, optional
                used to define the inner boundary.
            inner_mode : str, optional
                method used to define the inner boundary.
            outer : float, optional
                used to define the outer boundary.
            outer_mode : str, optional
                method used to define the outer boundary.

        Notes
        -----
        Boundary definitions:
        - Inner boundary: Defined by the `inner` parameter or a fraction of the center value.
        - Outer boundary: Defined by the `outer` parameter or the mean mass of the particles.

        Returns
        -------
        Gal3DAnalyzer
            An instance of Gal3DAnalyzer initialized with the analyzed data.
        """

        particles = Particles(pos=pos, mass=mass, recenter=recenter)

        inner = kwargs.get("inner", None)
        outer = kwargs.get("outer", None)
        if inner is None:
            assert 0 < inner_frac < 1, "Inner fraction must be between 0 and 1."
            inner = particles.get_parameter([0,0,0])[0]*inner_frac
            inner_mode = "value"
        else:
            try:
                inner_mode = kwargs["inner_mode"]
            except KeyError as e:
                raise KeyError("Inner mode not specified in kwargs") from e

        if outer is None:
            outer = np.mean(mass)
            outer_mode = "value"
        else:
            try:
                outer_mode = kwargs["outer_mode"]
            except KeyError as e:
                raise KeyError("Outer mode not specified in kwargs") from e

        field = cls._build_field(particles=particles, inner=inner, outer=outer, inner_mode=inner_mode, outer_mode=outer_mode)
        structure = cls._build_structure(np.mean(field.inner_r))
        optimizer: OptimizerBase
        if "OptimizerLMFit" in Optimizer.available_plugins(): # LMFit is better, if available
            optimizer = Optimizer.get_plugin(name="OptimizerLMFit")("least_squares")#  leastsq or least_squares?, least_squares is more robust but slightly slower
        else:
            optimizer = Optimizer.get_plugin(name="OptimizerScipy")(algorithm="trf")  # trf for curve fit, OptimizerScipy Powell for sum

        return Gal3DAnalyzer(particles=particles, field=field, structure=structure, optimizer=optimizer)

    @staticmethod
    def _build_field(particles: Particles, inner: float, outer: float, inner_mode: str = "value", outer_mode: str = "value") -> SphField:
        num_ray = min(1024, int(len(particles.r) / 150))
        num_ray = max(num_ray, 64)
        logger.info("Set inner %s to %.2e", inner_mode, inner)
        logger.info("Set outer %s to %.2e", outer_mode, outer)
        field = SphField(particles, num_ray=num_ray  # a better solution, use center_parameter*0.9 as inner, but how to determine outer boundary?
                ).build_field_boundary(inner=inner, outer=outer, inner_mode=inner_mode, outer_mode=outer_mode
                ).build_profile_sample(
                ).build_profile_interpolator(
                ).build_isodensity_profile(
                )
        return field

    @staticmethod
    def _build_structure(inner_r: float) -> Structure3D:
        inner = inner_r / 2
        structure = Structure3D(coordinate="ShiftEuler", geometry="Ellipsoid_S",
                                error_func="sums_dev_rscale", error_method="isodensity_curve_dcall")
        structure.parameters.set_ub(x=inner, y=inner, z=inner)
        structure.parameters.set_lb(x=-inner, y=-inner, z=-inner)
        return structure


    def fit(self, num_step:int = 200, step_mode: str = "log", **kwargs: Any) -> ModelResult:
        """
        Fit the model to the data.

        Parameters
        ----------
        num_step : int
            The number of steps for the fitting process.
        step_mode : str
            The mode of the steps (e.g., 'log' for logarithmic spacing).

        """
        r_min = max(np.median(self.field.inner_r)*3,self.field.iso_pro_r[0]*3)
        r_max = min(self.field.iso_pro_r[-1],np.median(self.field.outer_r))

        if step_mode == "log":
            r = np.geomspace(r_min, r_max, num_step)
        else:
            r = np.linspace(r_min, r_max, num_step)

        return self._fit(r,**kwargs)


    def _fit(self, r: float | Iterable = np.geomspace(1, 10, 200), **kwargs: Any) -> ModelResult:
        """
        Fit the model to a single radius or a sequence of radii.

        Parameters
        ----------
        r : float or iterable of float, optional
            The radius or sequence of radii at which to perform the fit.
            Defaults to log-spaced radii from 1 to 10.
        **kwargs : dict
            Additional keyword arguments passed to the fitting workflow.

        Returns
        -------
        ModelResult
        """
        workflow = self.get_workflow()
        logger.info("Using workflow: %s", workflow.__class__.__name__)

        if not isinstance(r, Iterable):
            try:
                res = workflow(self, r, **kwargs)
            except FitDataError as e:
                logger.error("Error fitting radius %f: %s", r, e)
                res = EmptyModelResult()
            return res
        else:
            resall: list[ModelResult] = []
            errors: dict[str, list[str]] = {}
            for i in tqdm(r, desc="Fitting radii", disable=False):
                try:
                    if resall:
                        res_value = {key: resall[-1][key][0] for key in resall[-1].keys()}
                        kwargs["init_parameters"] = res_value
                    res = workflow(self, i, **kwargs)
                    resall.append(res)
                except FitDataError as e:
                    etype = type(e).__name__
                    if etype not in errors:
                        errors[etype] = []
                    errors[etype].append(f"{i:.2f}")

            if errors:
                for etype, radii in errors.items():
                    logger.error("%s: Skipped radii: %s", etype, ", ".join(radii))

            if len(resall) > 0:
                return sum(resall[1:], resall[0])
            else:
                return EmptyModelResult()

    def get_workflow(self):
        """
        Determine which registered workflow to use based on its condition.

        Returns
        -------
        Callable
            The fitting workflow function to be used.

        Raises
        ------
        TypeError
            If no valid workflow condition is satisfied.
        """
        return self.fit_workflow.get_workflow(self)
