import logging
from abc import abstractmethod
from typing import List, overload, Type

from gal3d.plugin import PluginBase, PluginManager

__all__ = ['Optimizer', 'OptimizerBase']

logger = logging.getLogger("gal3d.optimization.optimizer")

class OptimizerBase(PluginBase):
    """
    Abstract base class for implementing optimization algorithms.
    Subclasses must implement the fitting method and define available algorithms.

    Attributes:
    ----------
    algo_name (str): 
        Name of the optimization algorithm.
    algo_options (dict): 
        Options specific to the algorithm.
    """

    def __init__(self, algorithm: str, algo_options: dict | None = None):
        """
        Initializes an optimizer with a specified algorithm and options.

        Parameters:
        ----------
        algorithm (str): 
            The name of the optimization algorithm.
        algo_options (dict, optional): 
            A dictionary of options specific to the algorithm. Defaults to None.

        Raises:
        -------
            ValueError: If the specified algorithm is not valid.
        """

        if not self.has_algorithm(algorithm):
            raise ValueError(f"{algorithm} is not a valid algorithm name.\n")

        self.algo_name = algorithm

        self.algo_options = algo_options or {}

    def __init_subclass__(cls, **kwargs):
        """
        Register the subclass as an optimizer plugin and update the plugin stub if update_stub.
        """
        super().__init_subclass__(**kwargs)
        OptimizerManager.register(cls)

    @abstractmethod
    def fitting(
        self,
        fun,
        x0,
        bounds,
        func_args: tuple | None = None,
        func_kwargs: dict | None = None,
        **kwargs,
    ):
        """
        Perform the fitting process.

        This method must be implemented by subclasses.

        Parameters
        ----------
        fun : callable
            The objective function to minimize.
        x0 : array-like
            Initial guess for the parameters.
        bounds : sequence
            Bounds for the parameters.
        func_args : tuple, optional
            Additional arguments to pass to the objective function (default is None).
        func_kwargs : dict, optional
            Additional keyword arguments to pass to the objective function (default is None).
        **kwargs : additional keyword arguments
            Additional options for the fitting algorithm.

        Returns
        -------
        result : object
            The result of the fitting.
        """
        pass

    def set_options(self, **kwargs):
        """
        Update the algorithm options.

        Parameters
        ----------
        **kwargs : keyword arguments
            Options to update in the algorithm.
        """
        self.algo_options.update(**kwargs)

    def has_algorithm(self, algorithm: str) -> bool:
        """
        Check if the given algorithm is available.
        """
        if algorithm in self.available_algorithm():
            return True
        return False

    @classmethod
    @abstractmethod
    def available_algorithm(cls) -> List[str]:
        """
        List of available algorithms.

        Returns
        -------
        List[str]
            A list of available algorithm names.
        """
        pass


class OptimizerManager(PluginManager[OptimizerBase]):
    """
    Factory class for accessing registered optimizer plugins.
    """
    _plugins = {}
    _plugin_module = "gal3d.optimization.optimizer_plugins"
    _base_class = OptimizerBase
    
    
    
Optimizer = OptimizerManager

