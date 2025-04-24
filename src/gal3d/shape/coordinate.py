import os
import logging
from typing import List

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .with_parameter import WithParameter, abstractmethod, Parameters
from ..optimization.parameter import Parameters
from ..util.func_signature import generate_plugin_stub
from ..util.func_decorator import classproperty
from .. import config_parser


__all__ = ['Coordinate', 'CoordinateBase']

logger = logging.getLogger("gal3d.shape.coordinate")

_CoordinatePlugins = dict()

_current_path = os.path.realpath(__file__)
_current_dir = os.path.dirname(__file__)
_current_file_name = os.path.basename(_current_path)
_pyi_name = _current_file_name.replace('.py', '.pyi')


class CoordinateBase(WithParameter):
    """
    Abstract base class for coordinate transformation plugins.

    Defines the interface for forward/inverse transformations and Jacobians.

    Methods:
    -------
    __call__(pos): 
        Transforms the given position using the current coordinate system.
    jacobian(pos): 
        Computes the Jacobian of the transformation at the given positions.
    inverse(pos): 
        Inverse transforms the given position.
    """

    def __init_subclass__(cls, **kwargs):
        """
        Register a new coordinate plugin subclass and update the plugin stub.
        """

        if not super().__init_subclass__():
            logger.info(f"Find CoordinatePlugin: {cls.__name__} but fail to load")
            return

        _CoordinatePlugins[cls.__name__] = cls
        logger.info(f"Find CoordinatePlugin: {cls.__name__} and load successfully")
        if config_parser['general'].getboolean("update_stub"):
            output_path = os.path.join(_current_dir, _pyi_name)
            generate_plugin_stub(
                Coordinate, CoordinateBase, _CoordinatePlugins, output_path
            )
            logger.info(f"✅ Updated stub: {output_path}")

    @abstractmethod
    def __call__(self, pos: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Apply coordinate transformation.

        Parameters
        ----------
        pos : ndarray of float64
            The input positions.

        Returns
        -------
        ndarray of float64
            The transformed positions.
        """
        pass

    @abstractmethod
    def jacobian(self, pos: NDArray[np.float64]) -> tuple:
        """
        Compute the Jacobian matrix of the transformation at a position.

        Parameters
        ----------
        pos : ndarray of float64
            The input positions.

        Returns
        -------
        tuple
            The Jacobian matrices.
        """
        pass

    @abstractmethod
    def inverse(self, pos: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Perform the inverse coordinate transformation.

        Parameters
        ----------
        pos : ndarray of float64
            The transformed positions.

        Returns
        -------
        ndarray of float64
            The original (inverse-transformed) positions.
        """
        pass

    @staticmethod
    @abstractmethod
    def quick_call(*args, **kwargs) -> NDArray[np.float64]:
        """
        Fast version of the coordinate transformation.

        Returns
        -------
        ndarray of float64
            Transformed positions.
        """
        pass

    @staticmethod
    @abstractmethod
    def quick_jacobian(*args, **kwargs) -> tuple:
        """
        Fast version of the Jacobian computation.

        Returns
        -------
        tuple
            Jacobian matrices.
        """
        pass

    @staticmethod
    @abstractmethod
    def quick_inverse(*args, **kwargs) -> NDArray[np.float64]:
        """
        Fast version of the inverse coordinate transformation.

        Returns
        -------
        ndarray of float64
            Inverse-transformed positions.
        """
        pass


class Coordinate:
    """
    Factory class for accessing registered Coordinate plugins.

    This class provides static methods to load and retrieve available
    Coordinate plugins derived from `CoordinateBase`.

    Methods
    -------
    get_plugin(plugin)
        Retrieve a specific Coordinate plugin by name.
    available_plugins
        List all available Coordinate plugins.
    """

    @staticmethod
    def _updata_plugin_stub():
        output_path = os.path.join(_current_dir, _pyi_name)
        generate_plugin_stub(
            Coordinate, CoordinateBase, _CoordinatePlugins, output_path
        )
        logger.info(f"✅ Updated stub: {output_path}")

    @staticmethod
    def get_plugin(plugin: str | None) -> CoordinateBase:
        """
        Get an coordinate plugin

        Parameters:
        plugin: str,
            the name of plugin, available see available_plugins

        Returns:
            available_plugins of CoordinateBase
        """
        assert (isinstance(plugin, str)) or (plugin is None)

        if plugin is None:
            return CoordinateBase
        if not _CoordinatePlugins:
            Coordinate._load_plugin()
        return _CoordinatePlugins[plugin]
    
    @staticmethod
    def _load_plugin():
        import importlib
        importlib.import_module("gal3d.shape.coordinate_plugins")
        logger.info("Successfully loaded coordinate plugins")

    @classproperty
    def available_plugins(cls) -> List[str]:
        """ A list of available Coordinate plugins. """
        if not _CoordinatePlugins:
            cls._load_plugin()
        return list(_CoordinatePlugins.keys())



