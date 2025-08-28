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

from .field import SphField
from .model_workflow.fit_workflow import FitWorkflow
from .optimization.optimizer import Optimizer, OptimizerBase
from .optimization.result import EmptyModelResult, ModelResult
from .point import Particles
from .shape import Structure3D

logger = logging.getLogger("gal3d.analyzer")

class Gal3DAnalyzer:
    """
    An analyzer for fitting galaxy 3D structures using particle, field, and shape information.

    Attributes
    ----------
    fit_workflow : dict of str to tuple of (Callable, Callable)
        A registry for fitting workflow conditions and their corresponding functions.
    particle : Particles
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
        particle: Particles,
        field: SphField,
        structure: Structure3D,
        optimizer: OptimizerBase,
    ):

        self.particle = particle
        self.field = field
        self.structure = structure
        self.optimizer = optimizer

    @classmethod
    def analyze(
        cls,
        pos: np.ndarray,
        mass: np.ndarray,
        recenter: bool = True,
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
            Whether to recenter the particle positions (default is True).
        **kwargs
            Additional keyword arguments for analysis.
            res_r : float, optional
                Spatial resolution to use.
            res_r_max : float, optional
                Maximum allowed spatial resolution.

        Returns
        -------
        Gal3DAnalyzer
            An instance of Gal3DAnalyzer initialized with the analyzed data.
        """

        particle = Particles(pos=pos, mass=mass, recenter=recenter)

        res_r = kwargs.get("res_r", particle.estimate_spatial_resolution())
        res_r_max = kwargs.get("res_r_max")
        if res_r_max is not None:
            res_r = min(res_r, res_r_max)
        res_m = particle.estimate_mass_resolution()
        logger.info("Estimated mass resolution: %f, spatial resolution: %f", res_m, res_r)

        field = cls._build_field(particle=particle, res_r=res_r, res_m=res_m)
        structure = cls._build_structure(res_r)
        optimizer = Optimizer.get_plugin(name = "OptimizerScipy")(algorithm="Powell") # OptimizerScipy Powell

        return Gal3DAnalyzer(particle=particle,field=field,structure=structure,optimizer=optimizer)

    @staticmethod
    def _build_field(particle: Particles, res_r: float, res_m: float) -> SphField:
        num_ray = min(1024, int(len(particle.r) / 150))
        num_ray = max(num_ray, 64)
        inner = res_r / 2
        outer = res_m / (4 * np.pi / 3 * (3 * res_r) ** 3)
        logger.info("Set inner radius to %f", inner)
        logger.info("Set outer value to %f", outer)
        field = SphField(particle, num_ray=num_ray  # a better solution, use center_parameter*0.9 as inner, but how to determine outer boundary?
                ).build_field_boundary(inner=inner, outer=outer, inner_mode="dist", outer_mode="value"
                ).build_profile_sample(
                ).build_profile_interpolator(
                ).build_isodensity_profile(
                )
        return field

    @staticmethod
    def _build_structure(res_r: float) -> Structure3D:
        inner = res_r / 2
        structure = Structure3D(coordinate="EulerShift", geometry="Ellipsoid_S",
                                error_func="sums_dev_rscale", error_method="isodensity_dcall")
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
        r_min = max(np.median(self.field.inner_r)*6,self.field.iso_pro_r[0]*3)
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
            except Exception as e:
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
                except Exception as e:
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
