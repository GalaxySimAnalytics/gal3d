import logging
import os
from abc import ABC, abstractmethod
from typing import List

from .. import config_parser
from ..util.func_decorator import classproperty
from ..util.func_signature import generate_plugin_stub

__all__ = ['Optimizer', 'OptimizerBase']

logger = logging.getLogger("gal3d.optimization.optimizer")

_OptimizerPlugins = dict()

_current_path = os.path.realpath(__file__)
_current_dir = os.path.dirname(__file__)
_current_file_name = os.path.basename(_current_path)
_pyi_name = _current_file_name.replace('.py', '.pyi')

class OptimizerBase(ABC):
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

        if not self.has_algorim(algorithm):
            raise ValueError(f"{algorithm} is not a valid algorithm name.\n")

        self.algo_name = algorithm

        self.algo_options = algo_options or {}

    def __init_subclass__(cls, **kwargs):
        """
        Register the subclass as an optimizer plugin and update the plugin stub if update_stub.
        """
        _OptimizerPlugins[cls.__name__] = cls
        logger.info(f"OptimizerPlugin found: {cls.__name__} and loaded successfully")
        if config_parser['general'].getboolean("update_stub"):
            output_path = os.path.join(_current_dir, _pyi_name)
            generate_plugin_stub(
                Optimizer, OptimizerBase, _OptimizerPlugins, output_path
            )
            logger.info(f"✅ Updated stub: {output_path}")

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

    def has_algorim(self, algorithm: str) -> bool:
        """
        Check if the given algorithm is available.

        Parameters
        ----------
        algorithm : str
            The name of the algorithm to check.

        Returns
        -------
        bool
            True if the algorithm is available, False otherwise.
        """
        if algorithm in self.available_algorithm:
            return True
        return False

    @classproperty
    @abstractmethod
    def available_algorithm(self) -> List[str]:
        """
        List of available algorithms.

        Returns
        -------
        List[str]
            A list of available algorithm names.
        """
        pass


class Optimizer:
    """
    Factory class for accessing registered optimizer plugins.

    This class provides static methods to load and retrieve available
    optimizer plugins derived from `OptimizerBase`.

    Methods
    -------
    get_plugin(plugin)
        Retrieve a specific optimizer plugin by name.
    available_plugins
        List all available optimizer plugins.
    """

    @staticmethod
    def _updata_plugin_stub():
        """
        Update the plugin stub file for the optimizer.
        """
        output_path = os.path.join(_current_dir, _pyi_name)
        generate_plugin_stub(Optimizer, OptimizerBase, _OptimizerPlugins, output_path)
        logger.info(f"✅ Updated stub: {output_path}")

    @staticmethod
    def get_plugin(plugin: str | None) -> OptimizerBase:
        """
        Get an optimizer plugin

        Parameters:
        plugin: str,
            the name of plugin, available see available_plugins

        Returns
        -------
        OptimizerBase
            The optimizer plugin corresponding to the provided name, or 
            the base OptimizerBase if no plugin is specified.
        """
        assert (isinstance(plugin, str)) or (plugin is None)

        if plugin is None:
            return OptimizerBase
        if not _OptimizerPlugins:
            Optimizer._load_plugin()

        return _OptimizerPlugins[plugin]

    @staticmethod
    def _load_plugin():
        import importlib
        importlib.import_module("gal3d.optimization.optimizer_plugins")
        logger.info("Successfully loaded optimizer plugins")
    
    @classproperty
    def available_plugins(cls) -> List[str]:
        """ A list of available optimizer plugins. """
        if not _OptimizerPlugins:
            cls._load_plugin()
        return list(_OptimizerPlugins.keys())

