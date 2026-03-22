from collections.abc import Iterable
from typing import Any, Literal, TypeAlias, Union, overload

from _typeshed import Incomplete

from gal3d.analyzer import Gal3DAnalyzer as Gal3DAnalyzer
from gal3d.density import DensitySource as DensitySource
from gal3d.model_workflow.fit_workflow_plugins.ellipsoid_fit import EllipsoidFitWorkflow
from gal3d.model_workflow.fit_workflow_plugins.iterate_ellipsoid_continuous import IterateEllipsoidDensity
from gal3d.model_workflow.fit_workflow_plugins.iterate_ellipsoid_discrete import IterateEllipsoidParticles
from gal3d.optimization.result import EmptyModelResult as EmptyModelResult, ModelResult as ModelResult
from gal3d.plugin import PluginBase as PluginBase, PluginManager as PluginManager
from gal3d.point import Particles as Particles
from gal3d.util.errors import FitDataError as FitDataError

logger: Incomplete
FitInput: TypeAlias = Union[Gal3DAnalyzer, Particles, DensitySource]

class FitWorkflowBase(PluginBase):
    """
    Base class for all fitting workflows.

    Subclasses must implement:
    - _fit_single(self, obj, r: float, **kwargs) -> ModelResult:
        Fit the model at a single radius ``r``.
    - condition(analyzer: "Gal3DAnalyzer") -> bool:
        Returns True if this workflow should be selected for the given analyzer.

    Registration:
    Subclasses are automatically registered as plugins via __init_subclass__.
    """
    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Register the subclass as a fitting workflow plugin.
        """
    @staticmethod
    def condition(obj: FitInput) -> bool:
        """
        Condition for selecting the fitting workflow.

        Parameters
        ----------
        obj : FitInput
            The analyzer, particle, or density source instance.

        Returns
        -------
        bool
            True if this workflow should be used, False otherwise.
        """
    def __call__(
        self,
        obj: FitInput,
        r: float | Iterable[float],
        *,
        progress: bool = True,
        warm_start: bool = True,
        **kwargs: Any,
    ) -> ModelResult:
        """
        Fit the model at one or multiple radii.

        Handles scalar and sequence inputs uniformly.  For a sequence of
        radii the previous result is optionally used as the initial
        parameter guess for the next radius (warm start).

        Parameters
        ----------
        obj : FitInput
            The analyzer, particle, or density source instance.
        r : float or iterable of float
            Radius or sequence of radii at which to perform the fit.
        progress : bool, optional
            Show a ``tqdm`` progress bar when iterating over radii.
            Default is ``True``.
        warm_start : bool, optional
            If ``True`` (default), pass the best-fit parameters of the
            previous radius as ``init_parameters`` to the next call of
            :meth:`_fit_single`.  Has no effect for scalar ``r``.
        **kwargs
            Extra keyword arguments forwarded to :meth:`_fit_single`.

        Returns
        -------
        ModelResult
            * Scalar ``r``  → single :class:`ModelResult` (or
              :class:`EmptyModelResult` on :exc:`FitDataError`).
            * Sequence ``r`` → aggregated :class:`ModelResult` summed
              over all successful radii, or :class:`EmptyModelResult`
              if every radius failed.
        """

class FitWorkflow(PluginManager[FitWorkflowBase]):
    """
    Factory class for accessing registered fitting workflows.

    This manager loads all FitWorkflowBase plugins from the specified module,
    and selects the appropriate workflow based on the analyzer's condition.

    Attributes
    ----------
    _plugins : dict[str, type[FitWorkflowBase]]
        Registered workflow plugins.
    _plugin_module : str
        Module path to search for workflow plugins.
    _base_class : type
        The base class for all workflow plugins.
    """
    @classmethod
    def get_workflow(cls, obj: FitInput | str) -> FitWorkflowBase:
        """
        Select and instantiate the appropriate fitting workflow for the input object.

        Parameters
        ----------
        obj : FitInput | str
            The analyzer, particle, or density source instance, or workflow name.

        Returns
        -------
        FitWorkflowBase
            The selected workflow instance.

        Raises
        ------
        TypeError
            If no valid workflow is found for the object.
        """

    @overload
    @classmethod
    def get_plugin(cls, name: Literal["EllipsoidFitWorkflow"]) -> type[EllipsoidFitWorkflow]: ...
    @overload
    @classmethod
    def get_plugin(cls, name: Literal["IterateEllipsoidDensity"]) -> type[IterateEllipsoidDensity]: ...
    @overload
    @classmethod
    def get_plugin(cls, name: Literal["IterateEllipsoidParticles"]) -> type[IterateEllipsoidParticles]: ...
    @overload
    @classmethod
    def get_plugin(cls, name: str) -> type[FitWorkflowBase]: ...
