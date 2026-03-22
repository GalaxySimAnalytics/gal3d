"""
High-level interface for analyzing and fitting 3D galaxy structures.

This module provides the :class:`Gal3DAnalyzer` class, which orchestrates a
typical analysis pipeline:

1) Build a :class:`gal3d.point.Particles` container from position / mass arrays
   (with optional recentering and density estimator configuration).
2) Construct a spherical field :class:`gal3d.field.SphField` with inner / outer
   boundaries, profile samples and isodensity profiles.
3) Build a 3D structure model :class:`gal3d.shape.Structure3D` (geometry +
   coordinate system + error function).
4) Select an optimizer via the plugin system
   (:class:`gal3d.optimization.optimizer.Optimizer`) and run a fit workflow
   across radii (:class:`gal3d.model_workflow.fit_workflow.FitWorkflow`).

There are three main ways to construct an analyzer:

- High-level, with sensible defaults (recommended for most users):

  >>> analyzer = Gal3DAnalyzer.analyze(pos, mass)
  >>> result = analyzer.fit(num_step=100)     # log-spaced radii

- Medium-level, via configuration dataclasses:

  >>> from gal3d.analyzer import Gal3DAnalyzer, Gal3DAnalyzerCfg
  >>> cfg = Gal3DAnalyzerCfg()
  >>> cfg.particle.recenter = False
  >>> cfg.field.num_ray_max = 2048
  >>> cfg.optimizer.preferred_plugin = "OptimizerScipy"
  >>> analyzer = Gal3DAnalyzer.from_configs(pos, mass, config=cfg)

- Low-level, from fully constructed components (advanced / testing):

  >>> particles = Particles(pos, mass)
  >>> field = SphField(particles, num_ray=1024).build_field_boundary(...)
  >>> structure = Structure3D("ShiftEuler", "Ellipsoid_S",
  ...                         "sums_dev_rscale", "isodensity_curve_dcall")
  >>> optimizer = Optimizer.get_plugin("OptimizerScipy")("trf")
  >>> analyzer = Gal3DAnalyzer.from_components(
  ...     density_source=particles,
  ...     field=field,
  ...     structure=structure,
  ...     optimizer=optimizer,
  ... )

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
>>> single = analyzer.fit(radius=8.0)          # returns a ModelResult
>>> multiple = analyzer.fit(radius=[5., 10., 15.])  # list of radii

Tips
----
- To see optimizer plugins available:
    >>> from gal3d.optimization.optimizer import Optimizer
    >>> Optimizer.available_plugins()
- To print all managers/plugins:
    >>> from gal3d.plugin import PluginManagerRegistry
    >>> PluginManagerRegistry.print_plugins()
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field as dc_field
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from .field import SphField
from .model_workflow.fit_workflow import FitWorkflow, FitWorkflowBase
from .optimization.optimizer import Optimizer, OptimizerBase
from .point import Particles
from .shape import Structure3D

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .density import DensitySource
    from .optimization.result import ModelResult
    from .theoretical import TheoreticalDensityDistribution



logger = logging.getLogger("gal3d.analyzer")


T = TypeVar("T", bound="Gal3DAnalyzer")

@dataclass
class ParticleCfg:
    """
    Configuration for building a :class:`gal3d.point.Particles` instance.

    Parameters
    ----------
    recenter : bool, optional
        If True (default), recenter particle positions before analysis.
    rmax : float, optional
        Optional maximum radius cut; particles beyond this distance
        (in 3D) may be excluded by :class:`Particles`.
    density_estimator : str, optional
        Name of the registered density estimator plugin
        (e.g. "DensityEstimatorKNN").
    estimator_kwargs : dict, optional
        Extra keyword arguments passed to the density estimator
        constructor.
    """
    recenter: bool = True
    # Additional particle configuration options can be added here
    rmax : float | None = None
    density_estimator: str = "DensityEstimatorSPH"
    estimator_kwargs: dict = dc_field(default_factory=dict)

@dataclass
class FieldCfg:
    """
    Configuration for building a :class:`gal3d.field.SphField` instance.

    This includes boundary definition (inner/outer radii), sampling of
    profile points, and options for estimating isodensity profiles.

    Boundary parameters
    -------------------
    inner : float, optional
        Explicit inner boundary value. If None, it is estimated from
        particle parameters and ``inner_frac``.
    outer : float, optional
        Explicit outer boundary value. If None, it is estimated from
        particle radii (e.g. percentiles).
    inner_frac : float, optional
        Fraction of central parameter used as inner boundary when
        ``inner`` is None. Must be in (0, 1).
    inner_mode, outer_mode : {"dist","pct","value"}, optional
        How to interpret ``inner`` / ``outer`` values in
        :meth:`SphField.build_field_boundary`.

    Ray sampling
    ------------
    num_ray_min : int
        Minimum number of rays.
    num_ray_max : int
        Maximum number of rays.
    num_ray_scale : int
        Rough scaling: one ray per ``num_ray_scale`` particles.

    Profile sampling
    ----------------
    profile_num_p : int
        Number of radial points in the 1D profile.
    profile_step_mode : {"log","lin"}
        Radial spacing mode for profile sampling.

    Isodensity profile
    ------------------
    iso_method : {"pair","moi"}, optional
        Isodensity profile estimator name (registered in
        :class:`SphField`).
    iso_from_rays_func : bool, optional
        If True, use precomputed ray functions where possible.
    iso_res_b, iso_res_c : float, optional
        Resolution parameters passed down to iso-profile builders.
    """
    # boundary
    inner: float | None = None
    outer: float | None = None
    inner_frac: float = 0.9
    inner_mode: str = "value"       # {"dist","pct","value"}
    outer_mode: str = "value"       # {"dist","pct","value"}
    # sampling
    num_ray_min: int = 64
    num_ray_max: int = 1024
    num_ray_scale: int = 150    # per 150 particles one ray
    profile_num_p: int = 500
    profile_step_mode: str = "log"
    # estimate iso-density profile
    iso_method: str ="pair"
    iso_from_rays_func: bool = False
    iso_res_b: float = 0.2
    iso_res_c: float = 0.1

@dataclass
class StructureCfg:
    """
    Configuration for building a :class:`gal3d.shape.Structure3D`
    instance.

    Parameters
    ----------
    coordinate : {"ShiftEuler","Euler","Cartesian"}
        Coordinate system / transformation class name.
    geometry : {"Ellipsoid_S","Ellipsoid_T","Ellipsoid_O","Triaxial_N"}
        Geometry plugin name.
    error_func : str
        Error function name from :class:`gal3d.shape.MinimizeFunc`.
    error_method : str
        Error evaluation method name registered on
        :class:`gal3d.shape.Structure3D`.
    """
    coordinate: str = "ShiftEuler"   # {"ShiftEuler","Euler","Cartesian"}
    geometry: str = "Ellipsoid_S"    # {"Ellipsoid_S","Ellipsoid_T","Ellipsoid_O","Triaxial_N"}
    error_func: str = "sums_dev_rscale"   # {"sums_dev_rscale","sums_dev_rhalf","sums_dev_density"}
    error_method: str = "isodensity_curve_dcall"  # {"isodensity_curve_dcall","isodensity_point_dcall","profile_curve_dcall","profile_point_dcall"}

@dataclass
class OptimizerCfg:
    """
    Configuration for selecting and constructing an optimizer.

    Parameters
    ----------
    preferred_plugin : str
        Name of the preferred optimizer plugin
        (e.g. "OptimizerLMFit" or "OptimizerScipy").
    preferred_method : str
        Algorithm name passed to the preferred optimizer plugin.
    default_plugin : str
        Fallback optimizer plugin if the preferred one is not
        available.
    default_method : str
        Fallback algorithm name for the default plugin.
    algo_options : dict
        Extra keyword arguments forwarded to the optimizer constructor
        (e.g. tolerances, max iterations).
    """
    preferred_plugin: str = "OptimizerLMFit"  # or "OptimizerScipy"
    preferred_method: str = "least_squares"  # for LMFit: {"leastsq","least_squares"}, for Scipy: {"Powell","trf","dogbox"}

    default_plugin: str = "OptimizerScipy"
    default_method: str = "trf"  # for Scipy: {"Powell","trf","dogbox"}

    algo_options: dict[str, Any] = dc_field(default_factory=dict)

@dataclass
class Gal3DAnalyzerCfg:
    """
    Aggregate configuration for :class:`Gal3DAnalyzer`.

    This groups together the four sub-configurations that control how
    the main components are built:

    - :class:`ParticleCfg`   → :class:`gal3d.point.Particles`
    - :class:`FieldCfg`      → :class:`gal3d.field.SphField`
    - :class:`StructureCfg`  → :class:`gal3d.shape.Structure3D`
    - :class:`OptimizerCfg`  → optimizer plugin instance

    Typical usage
    -------------
    >>> from gal3d.analyzer import Gal3DAnalyzer, Gal3DAnalyzerCfg
    >>> cfg = Gal3DAnalyzerCfg()
    >>> cfg.particle.recenter = False
    >>> cfg.field.num_ray_max = 2048
    >>> analyzer = Gal3DAnalyzer.from_configs(pos, mass, config=cfg)
    """
    particle: ParticleCfg = dc_field(default_factory=ParticleCfg)
    field: FieldCfg = dc_field(default_factory=FieldCfg)
    structure: StructureCfg = dc_field(default_factory=StructureCfg)
    optimizer: OptimizerCfg = dc_field(default_factory=OptimizerCfg)


class Gal3DAnalyzer:
    """
    High-level analyzer for fitting 3D galaxy structures.

    This class wires together four main components:

    - :class:`gal3d.point.Particles`
    - :class:`gal3d.field.SphField`
    - :class:`gal3d.shape.Structure3D`
    - :class:`gal3d.optimization.optimizer.OptimizerBase`

    and runs a fit workflow over a range of radii.

    Construction patterns
    ---------------------
    High-level (recommended):

    >>> analyzer = Gal3DAnalyzer.analyze(pos, mass)
    >>> result = analyzer.fit(num_step=100)

    Medium-level (explicit config):

    >>> from gal3d.analyzer import Gal3DAnalyzerCfg
    >>> cfg = Gal3DAnalyzerCfg()
    >>> cfg.particle.recenter = False
    >>> cfg.optimizer.preferred_plugin = "OptimizerScipy"
    >>> analyzer = Gal3DAnalyzer.from_configs(pos, mass, config=cfg)

    Low-level (inject your own components):

    >>> particles = Particles(pos, mass)
    >>> field = SphField(particles, num_ray=1024).build_field_boundary(...)
    >>> structure = Structure3D("ShiftEuler", "Ellipsoid_S",
    ...                         "sums_dev_rscale", "isodensity_curve_dcall")
    >>> optimizer = Optimizer.get_plugin("OptimizerScipy")("trf")
    >>> analyzer = Gal3DAnalyzer.from_components(
    ...     density_source=particles,
    ...     field=field,
    ...     structure=structure,
    ...     optimizer=optimizer,
    ... )

    Once constructed, the main public API is:

    - :meth:`fit` – run a fit over one or many radii.
    - :meth:`get_workflow` – inspect which workflow implementation
      is used for the current analyzer.
    """

    fit_workflow = FitWorkflow

    def __init__(
        self,
        density_source: DensitySource,
        field: SphField,
        structure: Structure3D,
        optimizer: OptimizerBase,
        config: Gal3DAnalyzerCfg | None = None,
    ):
        """
        Create a :class:`Gal3DAnalyzer` from fully built components.

        Parameters
        ----------
        density_source : DensitySource
            The density source used for field construction and fitting.
        field : SphField
            Spherical field created from the density source, with boundaries,
            profiles, and isodensity data already built.
        structure : Structure3D
            Geometric model describing the 3D galaxy shape and error
            evaluation method.
        optimizer : OptimizerBase
            Optimization engine used by the fit workflow.
        config : Gal3DAnalyzerCfg, optional
            Configuration instance from which the components were
            derived (if any). Used only for bookkeeping and potential
            downstream inspection.
        """

        self.density_source = density_source
        self.field = field
        self.structure = structure
        self.optimizer = optimizer
        self.config = config or Gal3DAnalyzerCfg()



    # ------------------------------------------------------------------
    # Lower-level builders: direct construction from components
    # ------------------------------------------------------------------
    @classmethod
    def from_components(
        cls: type[T],
        *,
        density_source: DensitySource,
        field: SphField,
        structure: Structure3D,
        optimizer: OptimizerBase,
        config: Gal3DAnalyzerCfg | None = None,
    ) -> T:
        """
        Construct an analyzer from pre-built components.

        This is the lowest-level construction helper. It performs no
        additional checks beyond storing the references.

        Parameters
        ----------
        density_source : DensitySource
            The density source used for field construction and fitting.
        field : SphField
            The spherical field instance (with boundaries, profiles,
            etc. already built as desired).
        structure : Structure3D
            The geometric structure model (with parameters and bounds
            already configured).
        optimizer : OptimizerBase
            The optimization engine to be used by the workflow.
        config : Gal3DAnalyzerCfg, optional
            Optional configuration associated with these components.

        Returns
        -------
        Gal3DAnalyzer
            An analyzer instance initialized with the provided
            components.
        """
        return cls(density_source=density_source, field=field, structure=structure, optimizer=optimizer, config=config)



    # ------------------------------------------------------------------
    # middle-level builders: construction from arrays + config
    # ------------------------------------------------------------------
    @classmethod
    def from_configs(
        cls: type[T],
        pos: np.ndarray,
        mass: np.ndarray,
        *,
        config: Gal3DAnalyzerCfg | None = None,
    ) -> T:
        """
        Construct an analyzer from arrays and a configuration object.

        Compared to :meth:`analyze`, this entry point does not apply
        extra high-level heuristics; it simply follows the rules encoded
        in :class:`Gal3DAnalyzerCfg` and its sub-configurations.

        Parameters
        ----------
        pos : array_like, shape (N, 3)
            Particle positions.
        mass : array_like, shape (N,)
            Particle masses.
        config : Gal3DAnalyzerCfg, optional
            Configuration dataclass instance describing how to build
            particles, field, structure, and optimizer.

        Returns
        -------
        Gal3DAnalyzer
            Analyzer instance constructed using the given configuration.
        """
        cfg = config or Gal3DAnalyzerCfg()

        density_source = cls._build_particles(pos, mass, cfg.particle)
        field = cls._build_field(density_source, cfg.field)
        structure = cls._build_structure(cfg.structure)
        optimizer = cls._build_optimizer(cfg.optimizer)

        return cls.from_components(
            density_source=density_source,
            field=field,
            structure=structure,
            optimizer=optimizer,
            config=cfg,
        )


    # ------------------------------------------------------------------
    # High-level builder: analyze from arrays + auto config
    # ------------------------------------------------------------------
    @classmethod
    def analyze(
        cls: type[T],
        pos: np.ndarray,
        mass: np.ndarray,
        *,
        recenter: bool = True,
        inner_frac: float = 0.9,
        inner: float | None = None,
        inner_mode: str | None = None,
        outer: float | None = None,
        outer_mode: str | None = None,
        config: Gal3DAnalyzerCfg | None = None,
        **kwargs: Any
    ) -> T:
        """
        High-level helper to analyze particle data and build an analyzer.

        This is the main user-facing constructor. It applies reasonable
        defaults for all four components and exposes only a small number
        of high-level knobs. For fine-grained control use
        :meth:`from_configs` or :meth:`from_components`.

        Parameters
        ----------
        pos : array_like, shape (N, 3)
            Particle positions.
        mass : array_like, shape (N,)
            Particle masses.
        recenter : bool, optional
            Whether to recenter particle positions before analysis.
            Overrides ``config.particle.recenter`` if provided.
        inner_frac : float, optional
            Fraction of the central parameter used to set the inner
            boundary when ``inner`` is not provided. Must be in (0, 1).
            Overrides ``config.field.inner_frac``.
        inner : float, optional
            Explicit inner boundary value. If provided, overrides both
            the automatic inner estimate and ``config.field.inner``.
        inner_mode : {"dist","pct","value"}, optional
            Interpretation of the inner boundary. If provided, overrides
            ``config.field.inner_mode``.
        outer : float, optional
            Explicit outer boundary value. If provided, overrides both
            the automatic outer estimate and ``config.field.outer``.
        outer_mode : {"dist","pct","value"}, optional
            Interpretation of the outer boundary. If provided, overrides
            ``config.field.outer_mode``.
        config : Gal3DAnalyzerCfg, optional
            Initial configuration to start from. The scalar keyword
            arguments above are applied on top of this config.
        **kwargs : dict
            Reserved for future extensions and forwarded to
            :meth:`from_configs`.

        Returns
        -------
        Gal3DAnalyzer
            Analyzer instance ready for fitting.

        Examples
        --------
        >>> analyzer = Gal3DAnalyzer.analyze(pos, mass)
        >>> result = analyzer.fit(num_step=100)
        """
        cfg = config or Gal3DAnalyzerCfg()

        cfg.particle.recenter = recenter

        if inner is not None:
            cfg.field.inner = float(inner)
        if outer is not None:
            cfg.field.outer = float(outer)
        if inner_mode is not None:
            cfg.field.inner_mode = inner_mode
        if outer_mode is not None:
            cfg.field.outer_mode = outer_mode
        if inner_frac is not None:
            cfg.field.inner_frac = inner_frac

        return cls.from_configs(pos, mass, config = cfg, **kwargs)

    @classmethod
    def analyze_theoretical_model(
        cls: type[T],
        model: TheoreticalDensityDistribution,
        *,
        inner_frac: float = 0.9,
        inner: float | None = None,
        inner_mode: str | None = None,
        outer: float | None = 1e4,
        outer_mode: str | None = "value",
        config: Gal3DAnalyzerCfg | None = None,
        **kwargs: Any
    ) -> T:
        """
        Analyze a theoretical density distribution model.

        This method allows users to fit not only discrete particle data
        but also continuous theoretical models.

        Parameters
        ----------
        model : TheoreticalDensityDistribution
            A theoretical density distribution instance.
        inner_frac : float, optional
            Fraction of the central parameter used to set the inner
            boundary when ``inner`` is not provided. Must be in (0, 1).
            Overrides ``config.field.inner_frac``.
        inner : float, optional
            Explicit inner boundary value. If provided, overrides both
            the automatic inner estimate and ``config.field.inner``.
        inner_mode : {"dist","pct","value"}, optional
            Interpretation of the inner boundary. If provided, overrides
            ``config.field.inner_mode``.
        outer : float, optional
            Explicit outer boundary value. If provided, overrides both
            the automatic outer estimate and ``config.field.outer``.
        outer_mode : {"dist","pct","value"}, optional
            Interpretation of the outer boundary. If provided, overrides
            ``config.field.outer_mode``.
        config : Gal3DAnalyzerCfg, optional
            Initial configuration to start from. The scalar keyword
            arguments above are applied on top of this config.
        **kwargs : dict
            Reserved for future extensions and forwarded to
            :meth:`from_configs`.
        """
        cfg = config or Gal3DAnalyzerCfg()

        if inner is not None:
            cfg.field.inner = float(inner)
        if outer is not None:
            cfg.field.outer = float(outer)
        if inner_mode is not None:
            cfg.field.inner_mode = inner_mode
        if outer_mode is not None:
            cfg.field.outer_mode = outer_mode
        if inner_frac is not None:
            cfg.field.inner_frac = inner_frac

        field = cls._build_field(model, cfg.field)
        structure = cls._build_structure(cfg.structure)
        optimizer = cls._build_optimizer(cfg.optimizer)

        if kwargs:
            logger.debug("Unused analyze_theoretical_model kwargs: %s", list(kwargs.keys()))

        return cls.from_components(
            density_source=model,
            field=field,
            structure=structure,
            optimizer=optimizer,
            config=cfg,
        )


    # ------------------------------------------------------------------
    # component builders
    # ------------------------------------------------------------------
    @staticmethod
    def _build_particles(
        pos: np.ndarray,
        mass: np.ndarray,
        cfg: ParticleCfg,
    ) -> Particles:
        """
        Build the particle container from position/mass arrays.

        Parameters
        ----------
        pos : array_like, shape (N, 3)
            Particle positions.
        mass : array_like, shape (N,)
            Particle masses.
        cfg : ParticleCfg
            Configuration for particle construction.

        Returns
        -------
        Particles
            Newly constructed particle container.
        """
        particles = Particles(
            pos=pos,
            mass=mass,
            rmax=cfg.rmax,
            recenter=cfg.recenter,
            density_estimator=cfg.density_estimator,
            estimator_kwargs=cfg.estimator_kwargs,
        )
        return particles

    @classmethod
    def _auto_field_bounds(
        cls,
        density_source: DensitySource,
        cfg: FieldCfg,
    ) -> tuple[float, str, float, str]:
        """
        Automatically determine inner and outer field boundaries.

        Inner and outer radii are inferred from the particle data and
        configuration if not set explicitly.

        Parameters
        ----------
        density_source : DensitySource
            The density source used for field construction and fitting.
        cfg : FieldCfg
            Field configuration.

        Returns
        -------
        inner : float
            Chosen inner boundary value.
        inner_mode : str
            Mode describing how ``inner`` is interpreted.
        outer : float
            Chosen outer boundary value.
        outer_mode : str
            Mode describing how ``outer`` is interpreted.
        """
        inner = cfg.inner
        outer = cfg.outer
        inner_mode = cfg.inner_mode
        outer_mode = cfg.outer_mode

        if inner is None:
            assert 0 < cfg.inner_frac < 1, "Inner fraction must be between 0 and 1."
            center_param = density_source([0, 0, 0])
            inner = float(center_param) * cfg.inner_frac
            inner_mode = "value"

        if outer is None:
            if hasattr(density_source, "r"):
                try:
                    outer = float(np.percentile(density_source.r, 95))
                except Exception:
                    outer = float(np.max(density_source.r))
                outer_mode = "dist"
            else:
                raise ValueError(
                    "outer must be provided for continuous DensitySource models "
                    "without particle radii."
                )

        return inner, inner_mode, outer, outer_mode

    @classmethod
    def _build_field(
        cls,
        density_source: DensitySource,
        cfg: FieldCfg,
    ) -> SphField:
        """
        Build a :class:`SphField` from density_source and configuration.

        This method chooses the number of rays, computes inner/outer
        boundaries, constructs radial profiles and isodensity profiles.

        Parameters
        ----------
        density_source : DensitySource
            The density source used for field construction and fitting.
        cfg : FieldCfg
            Field configuration.

        Returns
        -------
        SphField
            Fully constructed spherical field instance.
        """
        inner, inner_mode, outer, outer_mode = cls._auto_field_bounds(density_source, cfg)

        if hasattr(density_source, "r"):
            num_ray = min(cfg.num_ray_max, int(len(density_source.r) / cfg.num_ray_scale))
            num_ray = max(num_ray, cfg.num_ray_min)
        else:
            num_ray = cfg.num_ray_max

        logger.info("Built spherical field with %d rays", num_ray)

        logger.info("Set inner %s to %.3e", inner_mode, inner)
        logger.info("Set outer %s to %.3e", outer_mode, outer)

        field = (
            SphField(density_source, num_ray=num_ray)
            .build_field_boundary(inner=inner, outer=outer,
                                  inner_mode=inner_mode, outer_mode=outer_mode)
            .build_profile_sample(num_p=cfg.profile_num_p,
                                  step_mode=cfg.profile_step_mode)
            .build_profile_interpolator()
            .build_isodensity_profile(
                method=cfg.iso_method,
                from_rays_func=cfg.iso_from_rays_func,
                res_b=cfg.iso_res_b,
                res_c=cfg.iso_res_c,
            )
        )
        min_r = np.min(field.inner_r)
        max_r = np.max(field.outer_r)
        logger.info("Field radius range: [%.3e, %.3e]", min_r, max_r)

        return field


    @classmethod
    def _build_structure(
        cls,
        cfg: StructureCfg,
    ) -> Structure3D:
        """
        Build a :class:`Structure3D` instance from configuration.

        Parameters
        ----------
        cfg : StructureCfg
            Structure configuration.

        Returns
        -------
        Structure3D
            Geometric model instance.
        """
        structure = Structure3D(
            coordinate=cfg.coordinate,
            geometry=cfg.geometry,
            error_func=cfg.error_func,
            error_method=cfg.error_method,
        )
        return structure

    @classmethod
    def _build_optimizer(
        cls,
        cfg: OptimizerCfg,
    ) -> OptimizerBase:
        """
        Build an optimizer from configuration.

        The preferred plugin/method is used if available; otherwise the
        default plugin/method is used as a fallback.

        Parameters
        ----------
        cfg : OptimizerCfg
            Optimizer configuration.

        Returns
        -------
        OptimizerBase
            Constructed optimizer instance.
        """
        if cfg.preferred_plugin in Optimizer.available_plugins():
            optimizer = Optimizer.get_plugin(cfg.preferred_plugin)(
                cfg.preferred_method, **cfg.algo_options
            )
        else:
            logger.info("Preferred optimizer plugin '%s' not available; using default '%s'.", cfg.preferred_plugin, cfg.default_plugin)
            optimizer = Optimizer.get_plugin(cfg.default_plugin)(
                cfg.default_method, **cfg.algo_options
            )
        return optimizer


    def fit(self, num_step: int = 200, step_mode: str = "log", radius: float | Iterable[float] | None = None , **kwargs: Any) -> ModelResult:
        """
        Fit the model across a range of radii.

        This is the main public fitting method. If ``radius`` is given,
        a single radius or list/array of radii is used directly.
        Otherwise, a range is auto-chosen from the field.

        Parameters
        ----------
        num_step : int, optional
            Number of radii to evaluate when ``radius`` is not given.
        step_mode : {"log","lin"}, optional
            Spacing mode used to generate radii when ``radius`` is
            not given. "log" (default) uses logarithmic spacing.
        radius : float or iterable of float, optional
            Specific radius or sequence of radii to fit. If provided,
            overrides ``num_step`` and ``step_mode``.
        **kwargs
            Additional keyword arguments passed to the fitting workflow.
            Special keys:
            - progress : bool, optional (default True)
                If False, disable the progress bar over radii.

        Returns
        -------
        ModelResult
            Aggregated result across radii (or a single-radius result if
            ``radius`` is scalar).

        Examples
        --------
        >>> analyzer = Gal3DAnalyzer.analyze(pos, mass)
        >>> result = analyzer.fit(num_step=100)
        >>> single = analyzer.fit(radius=8.0)
        >>> many = analyzer.fit(radius=[5.0, 10.0, 15.0])
        """
        # Allow disabling progress bar via kwargs without leaking into workflow
        progress = kwargs.pop("progress", True)
        workflow = self.get_workflow()
        logger.info("Using workflow: %s", workflow.__class__.__name__)

        if radius is not None:
            return workflow(self, radius, progress=progress, **kwargs)

        r_min = max(np.median(self.field.inner_r) * 3, self.field.iso_pro_r[0] * 3)
        r_max = min(self.field.iso_pro_r[-1], np.median(self.field.outer_r))

        if step_mode == "log":
            r = np.geomspace(r_min, r_max, num_step)
        else:
            r = np.linspace(r_min, r_max, num_step)

        return workflow(self, r, progress=progress, **kwargs)

    def get_workflow(self) -> FitWorkflowBase:
        """
        Return the fitting workflow implementation for this analyzer.

        The workflow is chosen by :class:`FitWorkflow` based on the
        current analyzer (its components and configuration). This allows
        different strategies or backends to be plugged in without
        changing the analyzer API.

        Returns
        -------
        FitWorkflowBase
            The workflow instance which accepts
            ``(analyzer, radius, **kwargs)``.
        """
        return self.fit_workflow.get_workflow(self)
