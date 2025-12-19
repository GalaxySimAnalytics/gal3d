"""
Analyze and fit 3D galaxy structures from particle data.

Overview
--------
`Gal3DAnalyzer` orchestrates a typical pipeline:
1) Build particles from position/mass arrays (with optional recentering).
2) Construct a spherical field (`SphField`) with inner/outer boundaries.
3) Build a geometric model (`Structure3D`) with parameter bounds.
4) Choose an optimizer via plugin system and run a fit workflow across radii.


Quick start
-----------
>>> import numpy as np
>>> from gal3d.analyzer import Gal3DAnalyzer
>>>
>>> # Example synthetic data: (N, 3) positions, (N,) masses
>>> N = 100_000
>>> pos = np.random.normal(size=(N, 3))
>>> pos[:, 0] *= 5.0
>>> pos[:, 1] *= 2.0
>>> mass = np.ones(N)
>>>
>>> # Analyze with sensible defaults and run a fit
>>> analyzer = Gal3DAnalyzer.analyze(pos, mass)
>>> result = analyzer.fit(num_step=100)   # log-spaced radii by default
>>>
>>> # Or fit at specific radius
>>> single = analyzer.fit(radius = 8.0)  # returns a ModelResult
>>> multiple = analyzer.fit(radius = [5.0, 10.0, 15.0])  # list of radii

Tips
----
- To see optimizer plugins available:
    >>> from gal3d.optimization.optimizer import Optimizer
    >>> Optimizer.available_plugins()
- To print all managers/plugins:
    >>> from gal3d.plugin import PluginManagerRegistry
    >>> PluginManagerRegistry.print_plugins()
"""
import logging
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

import numpy as np
from tqdm import tqdm

from .field import SphField
from .model_workflow.fit_workflow import FitWorkflow
from .optimization.optimizer import Optimizer, OptimizerBase
from .optimization.result import EmptyModelResult, ModelResult
from .point import Particles
from .shape import Structure3D
from .util.errors import FitDataError

logger = logging.getLogger("gal3d.analyzer")


T = TypeVar("T", bound="Gal3DAnalyzer")
class Gal3DAnalyzer:
    """
    Analyzer for fitting 3D galaxy structures using particle, field, and shape modules.

    Responsibilities
    ----------------
    - Build the spherical field from particle data with valid inner/outer boundaries.
    - Build the 3D structure model and set parameter bounds.
    - Select an optimizer via plugin manager.
    - Run a fit workflow over a sequence of radii and aggregate results.

    Attributes
    ----------
    fit_workflow : FitWorkflow
        Workflow manager to select and run the fitting strategy.
    particles : Particles
        The input particle data (positions, masses).
    field : SphField
        Spherical field generator and profile builder.
    structure : Structure3D
        The geometric structure model to fit.
    optimizer : OptimizerBase
        The optimization engine used for fitting.
    """

    fit_workflow = FitWorkflow

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
        cls: type[T],
        pos: np.ndarray,
        mass: np.ndarray,
        recenter: bool = True,
        inner_frac: float = 0.9,
        **kwargs: Any
    ) -> T:
        """
        Analyze the given particle data to construct field, structure, and optimizer.

        Parameters
        ----------
        pos : array_like, shape (N, 3)
            Particle positions (float).
        mass : array_like, shape (N,)
            Particle masses (float).
        recenter : bool, optional (default: True)
            Whether to recenter particle positions.
        inner_frac : float, optional (default: 0.9)
            Fraction of the center parameter used to set the inner boundary
            when `inner` is not provided. Must be in (0, 1).
        **kwargs
            Additional options:
            - inner : float, optional
                Explicit inner boundary value (overrides computed default).
            - inner_mode : str, optional
                How the inner boundary is interpreted (default "value" when `inner` provided).
            - outer : float, optional
                Explicit outer boundary value (overrides computed default).
            - outer_mode : str, optional
                How the outer boundary is interpreted (default "value" when `outer` provided).

        Notes
        -----
        Defaults:
        - Inner: `inner_frac * center_parameter` from particles.
        - Outer: robust radius heuristic from particle distances (p95 of `particles.r`).

        Returns
        -------
        Gal3DAnalyzer
            An analyzer instance initialized with the analyzed components.
        """

        particles = Particles(pos=pos, mass=mass, recenter=recenter)

        inner = kwargs.get("inner", None)
        outer = kwargs.get("outer", None)
        if inner is None:
            assert 0 < inner_frac < 1, "Inner fraction must be between 0 and 1."
            center_param = particles.get_parameter([0, 0, 0])[0]
            inner = float(center_param) * inner_frac
            inner_mode = "value"
        else:
            inner_mode = kwargs.get("inner_mode", "value")

        if outer is None:
            try:
                outer = float(np.percentile(particles.r, 95))
            except Exception:
                outer = float(np.max(particles.r))
            outer_mode = "dist"
        else:
            outer_mode = kwargs.get("outer_mode", "value")


        field = cls._build_field(
            particles=particles,
            inner=inner,
            outer=outer,
            inner_mode=inner_mode,
            outer_mode=outer_mode,
        )
        structure = cls._build_structure(float(np.mean(field.inner_r)))
        optimizer: OptimizerBase
        # Prefer LMFit if available; use keyword args consistently
        if "OptimizerLMFit" in Optimizer.available_plugins(): # LMFit is better, if available
            optimizer = Optimizer.get_plugin(name="OptimizerLMFit")("least_squares")#  leastsq or least_squares?, least_squares is more robust but slightly slower
        else:
            optimizer = Optimizer.get_plugin(name="OptimizerScipy")(algorithm="trf")  # trf for curve fit, OptimizerScipy Powell for sum

        return cls(particles=particles, field=field, structure=structure, optimizer=optimizer)

    @staticmethod
    def _build_field(particles: Particles, inner: float, outer: float, inner_mode: str = "value", outer_mode: str = "value") -> SphField:
        """
        Build spherical field and derived profiles with reasonable ray count.

        The number of rays scales with sample size but is bounded:
        min(1024, len(r) / 150) with a floor of 64.
        """
        num_ray = min(1024, int(len(particles.r) / 150))
        num_ray = max(num_ray, 64)
        logger.info("Set inner %s to %.3e", inner_mode, inner)
        logger.info("Set outer %s to %.3e", outer_mode, outer)
        field = SphField(particles, num_ray=num_ray  # a better solution, use center_parameter*0.9 as inner, but how to determine outer boundary?
                ).build_field_boundary(inner=inner, outer=outer, inner_mode=inner_mode, outer_mode=outer_mode
                ).build_profile_sample(
                ).build_profile_interpolator(
                ).build_isodensity_profile(
                )
        return field

    @staticmethod
    def _build_structure(inner_r: float) -> Structure3D:
        """
        Build a Structure3D model with bounds tied to inner radius.
        """
        inner = inner_r / 2
        structure = Structure3D(coordinate="ShiftEuler", geometry="Ellipsoid_S",
                                error_func="sums_dev_rscale", error_method="isodensity_curve_dcall")
        structure.parameters.set_ub(x=inner, y=inner, z=inner)
        structure.parameters.set_lb(x=-inner, y=-inner, z=-inner)
        return structure


    def fit(self, num_step:int = 200, step_mode: str = "log", radius: float | Iterable | None = None , **kwargs: Any) -> ModelResult:
        """
        Fit the model across a range of radii.

        Parameters
        ----------
        num_step : int
            Number of radii to evaluate.
        step_mode : str
            'log' for logarithmic spacing (default), anything else for linear.
        radius : float or iterable of float, optional
            Specific radius or sequence of radii to fit. If provided, overrides num_step and step_mode.
        **kwargs
            Additional keyword arguments passed to the fitting workflow.
            Special (consumed here, not passed through):
            - progress : bool, optional (default True)
                If False, disables the progress bar.

        Returns
        -------
        ModelResult
            Aggregated result across radii.
        """
        # Allow disabling progress bar via kwargs without leaking into workflow
        progress = kwargs.pop("progress", True)

        if radius is not None:
            return self._fit(radius, progress=progress, **kwargs)

        r_min = max(np.median(self.field.inner_r) * 3, self.field.iso_pro_r[0] * 3)
        r_max = min(self.field.iso_pro_r[-1], np.median(self.field.outer_r))

        if step_mode == "log":
            r = np.geomspace(r_min, r_max, num_step)
        else:
            r = np.linspace(r_min, r_max, num_step)

        return self._fit(r, progress=progress, **kwargs)


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
            Special (consumed here):
            - progress : bool, optional (default True)
                If False, disables the progress bar over radii.

        Returns
        -------
        ModelResult
            The aggregated result for multiple radii or the single-radius result.
        """
        workflow = self.get_workflow()
        logger.info("Using workflow: %s", workflow.__class__.__name__)

        # Consume progress from kwargs (don't pass through to workflow)
        progress = kwargs.pop("progress", True)

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
            try:
                for i in tqdm(r, desc="Fitting radii", disable=not progress):
                    try:
                        if resall:
                            # Warm start: use last solution as init for the next radius
                            res_value = {key: resall[-1][key][0] for key in resall[-1].keys()}
                            kwargs["init_parameters"] = res_value
                        res = workflow(self, i, **kwargs)
                        resall.append(res)
                    except FitDataError as e:
                        etype = type(e).__name__
                        if etype not in errors:
                            errors[etype] = []
                        errors[etype].append(i)
            except KeyboardInterrupt:
                logger.warning("Interrupted by user; returning partial results.")

            if errors:
                for etype, radii in errors.items():
                    if len(radii) > 2:
                        r_l = np.min(radii)
                        r_h = np.max(radii)
                        logger.error("%s: Skipped %d radii between %.2f and %.2f", etype, len(radii), r_l, r_h)
                    else:
                        logger.error("%s: Skipped radii: %s", etype, ", ".join(f"{rad:.2f}" for rad in radii))

            if len(resall) > 0:
                return sum(resall[1:], resall[0])
            else:
                return EmptyModelResult()

    def get_workflow(self) -> Callable[..., ModelResult]:
        """
        Choose and return the first registered workflow whose condition is satisfied.

        Returns
        -------
        Callable[..., ModelResult]
            The fitting workflow function to be used.

        Raises
        ------
        TypeError
            If no valid workflow condition is satisfied.
        """
        return self.fit_workflow.get_workflow(self)
