"""
Workflow for model fitting.
"""
from typing import TYPE_CHECKING, Any, Union

from gal3d.optimization.result import ModelResult
from gal3d.plugin import PluginBase, PluginManager

if TYPE_CHECKING:
    from gal3d.analyzer import Gal3DAnalyzer
    from gal3d.point import Particles

class FitWorkflowBase(PluginBase):
    """
    Base class for all fitting workflows.

    Subclasses must implement:
    - __call__(self, analyzer: "Gal3DAnalyzer", *args, **kwargs) -> ModelResult:
        The main fitting logic.
    - condition(analyzer: "Gal3DAnalyzer") -> bool:
        Returns True if this workflow should be selected for the given analyzer.

    Registration:
    Subclasses are automatically registered as plugins via __init_subclass__.
    """
    def __init_subclass__(cls, **kwargs):
        """
        Register the subclass as a fitting workflow plugin.
        """
        super().__init_subclass__(**kwargs)
        FitWorkflow.register(cls)

    @staticmethod
    def condition(obj: Union["Gal3DAnalyzer", "Particles"]) -> bool:
        """
        Condition for selecting the fitting workflow.

        Parameters
        ----------
        obj : Gal3DAnalyzer | Particles
            The analyzer or particle instance.

        Returns
        -------
        bool
            True if this workflow should be used, False otherwise.
        """
        return False

    def __call__(self, obj: Union["Gal3DAnalyzer", "Particles"], *args: Any, **kwargs: Any) -> ModelResult:
        """
        Fit the model using the provided arguments.

        Parameters
        ----------
        obj : Gal3DAnalyzer | Particles
            The analyzer or particle instance.
        *args, **kwargs
            Arguments for the fitting workflow.

        Returns
        -------
        ModelResult
            The result of the fitting process.

        Raises
        ------
        NotImplementedError
            If not implemented in subclass.
        """
        raise NotImplementedError


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
    _plugins = {}
    _plugin_module = "gal3d.model_workflow.fit_workflow_plugins"
    _base_class = FitWorkflowBase

    @classmethod
    def get_workflow(cls, obj: Union["Gal3DAnalyzer", "Particles"]) -> FitWorkflowBase:
        """
        Select and instantiate the appropriate fitting workflow for the input object.

        Parameters
        ----------
        obj : Gal3DAnalyzer | Particles
            The analyzer or particle instance.

        Returns
        -------
        FitWorkflowBase
            The selected workflow instance.

        Raises
        ------
        TypeError
            If no valid workflow is found for the object.
        """
        cls.available_plugins()
        for wf_cls in cls._plugins.values():
            if wf_cls.condition(obj):
                return wf_cls()
        raise TypeError("No valid workflow found for input object.")
