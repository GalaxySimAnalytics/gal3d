import numpy as np
from _typeshed import Incomplete
from gal3d.optimization.result import ModelResult as ModelResult
from gal3d.plugin import PluginBase as PluginBase, PluginManager as PluginManager
from gal3d.shape import StructureCore as StructureCore
from typing import Any
from typing import Literal, overload
from gal3d.model_workflow.error_workflow_plugins.ellipsoid_error_estimator import EllipsoidErrorEstimator

logger: Incomplete

class ErrorWorkflowBase(PluginBase):
    """
    Base class for structure error estimator workflows.

    This class defines the interface for all error estimation workflows.
    Subclasses should implement the estimate_structure_error and estimate_model_error methods.

    Registration:
    Subclasses are automatically registered as plugins via __init_subclass__.
    """
    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Register the subclass as an error estimator workflow plugin.
        """
    @classmethod
    def condition(cls, result: StructureCore | ModelResult) -> bool:
        """
        Check if this workflow can handle the given result.

        Parameters
        ----------
        result : Union[StructureCore, ModelResult]
            The result to check.

        Returns
        -------
        bool
            True if this workflow can handle the result, False otherwise.
        """
    @classmethod
    def estimate_error(cls, result: StructureCore | ModelResult, param_name: list[str] | None = None, **kwargs: Any) -> dict[str, Any]:
        """ Estimate errors"""
    @classmethod
    def estimate_structure_update(cls, result: StructureCore, pos: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        """
        Estimate updates for a structure model.

        Parameters
        ----------
        structure : StructureCore
            The structure model to evaluate
        pos : numpy.ndarray
            Point cloud positions
        **kwargs : dict
            Additional keyword arguments

        Returns
        -------
        dict
            Dictionary of parameter updates

        Raises
        ------
        NotImplementedError
            If not implemented in subclass
        """
    @classmethod
    def estimate_model_update(cls, result: ModelResult, **kwargs: Any) -> dict[str, Any]:
        """
        Estimate updates for a model result.

        Parameters
        ----------
        model : ModelResult
            Model result object
        pos : numpy.ndarray
            Point cloud positions
        **kwargs : dict
            Additional keyword arguments

        Returns
        -------
        dict
            Dictionary of parameter updates

        Raises
        ------
        NotImplementedError
            If not implemented in subclass
        """

class ErrorWorkflow(PluginManager[ErrorWorkflowBase]):
    """
    Manager class for accessing registered error estimation workflows.

    This manager loads all ErrorWorkflowBase plugins from the specified module,
    and provides methods to access and use them.

    Attributes
    ----------
    _plugins : dict[str, type[ErrorWorkflowBase]]
        Registered error estimator workflow plugins
    _plugin_module : str
        Module path to search for error estimator workflow plugins
    _base_class : type
        The base class for all error estimator workflow plugins
    """
    @classmethod
    def get_error_estimator(cls, result: StructureCore | ModelResult | None = None, name: str | None = None) -> ErrorWorkflowBase:
        """
        Get the default error estimator workflow.

        Returns
        -------
        ErrorWorkflowBase
            An instance of the default error estimator workflow

        Raises
        ------
        LookupError
            If no error estimator workflows are available
        """
    @classmethod
    def estimate_error(cls, result: StructureCore | ModelResult, param_name: list[str] | None = None, **kwargs: Any) -> dict[str, np.ndarray]:
        '''
        Estimate errors for the given result.

        Parameters
        ----------
        result : Union["StructureCore", "ModelResult"]
            The result object to estimate errors for.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        dict
            A dictionary containing the estimated errors.
        '''

    @overload
    @classmethod
    def get_plugin(cls, name: Literal["EllipsoidErrorEstimator"]) -> type[EllipsoidErrorEstimator]: ...
    @overload
    @classmethod
    def get_plugin(cls, name: str) -> type[ErrorWorkflowBase]: ...
