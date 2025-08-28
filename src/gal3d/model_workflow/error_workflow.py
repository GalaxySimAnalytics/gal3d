"""
Workflow for model error estimation.
"""
import logging
from typing import TYPE_CHECKING, Any, Union

import numpy as np

from gal3d.plugin import PluginBase, PluginManager

if TYPE_CHECKING:
    from gal3d.optimization.result import ModelResult
    from gal3d.shape import StructureCore
    from gal3d.shape.with_parameter import WithParameter

logger = logging.getLogger("gal3d.model_workflow.error")


class ErrorWorkflowBase(PluginBase):
    """
    Base class for structure error estimator workflows.

    This class defines the interface for all error estimation workflows.
    Subclasses should implement the estimate_structure_error and estimate_model_error methods.

    Registration:
    Subclasses are automatically registered as plugins via __init_subclass__.
    """

    def __init_subclass__(cls, **kwargs):
        """
        Register the subclass as an error estimator workflow plugin.
        """
        super().__init_subclass__(**kwargs)
        ErrorWorkflow.register(cls)

    @classmethod
    def condition(cls, result: Union["StructureCore", "WithParameter", "ModelResult"]) -> bool:
        """
        Check if this workflow can handle the given result.

        Parameters
        ----------
        result : Union[StructureCore, WithParameter, ModelResult]
            The result to check.

        Returns
        -------
        bool
            True if this workflow can handle the result, False otherwise.
        """
        return False

    @classmethod
    def estimate_structure_error(
        cls,
        result: Union["StructureCore", "WithParameter"],
        pos: np.ndarray,
        **kwargs: Any
    ) -> dict[str, Any]:
        """
        Estimate errors for a structure model.

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
            Dictionary of parameter errors

        Raises
        ------
        NotImplementedError
            If not implemented in subclass
        """
        raise NotImplementedError("Subclasses must implement estimate_structure_error")

    @classmethod
    def estimate_model_error(
        cls,
        result: "ModelResult",
        **kwargs: Any
    ) -> dict[str, Any]:
        """
        Estimate errors for a model result.

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
            Dictionary of parameter errors

        Raises
        ------
        NotImplementedError
            If not implemented in subclass
        """
        raise NotImplementedError("Subclasses must implement estimate_model_error")


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
    _plugins = {}
    _plugin_module = "gal3d.model_workflow.error_workflow_plugins"
    _base_class = ErrorWorkflowBase

    @classmethod
    def get_error_estimator(cls, result: Union["StructureCore", "WithParameter", "ModelResult"] | None = None, name: str | None = None) -> ErrorWorkflowBase:
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
        available = cls.available_plugins()
        if not available:
            raise LookupError("No error estimator workflows available")
        if name is not None and name in available:
            return cls.get_plugin(name)()

        if result is not None:
            for wf_cls in cls._plugins.values():
                if wf_cls.condition(result):
                    return wf_cls()

        raise LookupError("No suitable error estimator workflow found")
