import logging
import os
from typing import List

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .. import config_parser
from ..util.func_decorator import classproperty

# Utility function to generate Python interface (.pyi) stubs for geometry plugins.
from ..util.func_signature import generate_plugin_stub
from .with_parameter import Parameters, WithParameter, abstractmethod

__all__ = ['Geometry', 'GeometryBase']

logger = logging.getLogger("gal3d.shape.geometry")

_GeometryPlugins = dict()

_current_path = os.path.realpath(__file__)
_current_dir = os.path.dirname(__file__)
_current_file_name = os.path.basename(_current_path)
_pyi_name = _current_file_name.replace('.py', '.pyi')


class GeometryBase(WithParameter):
    """Abstract base class for geometry models."""

    def __init_subclass__(cls, **kwargs):
        """
        Automatically registers geometry plugins upon subclass initialization.
        """

        if not super().__init_subclass__():
            logger.warning(f"GeometryPlugin found: {cls.__name__} but failed to load")
            return

        _GeometryPlugins[cls.__name__] = cls
        logger.debug(f"GeometryPlugin found: {cls.__name__} and loaded successfully")
        if config_parser['general'].getboolean("update_stub"):
            output_path = os.path.join(_current_dir, _pyi_name)
            generate_plugin_stub(Geometry, GeometryBase, _GeometryPlugins, output_path)
            logger.info(f"✅ Updated stub: {output_path}")

    @abstractmethod
    def __call__(self, pos: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluate the geometry function at the given positions.

        Parameters
        ----------
        pos : ndarray of float64
            Input positions to evaluate the geometry function.

        Returns
        -------
        ndarray of float64
            Evaluated results at the given positions.
        """
        pass

    @abstractmethod
    def jacobian(self, pos: NDArray[np.float64]) -> tuple:
        """
        Compute the Jacobian of the geometry function with respect to parameters.

        Parameters
        ----------
        pos : ndarray of float64
            Positions at which to compute the Jacobian.

        Returns
        -------
        tuple
            The Jacobian matrix.
        """
        pass

    @abstractmethod
    def ray_intersect(self, pos: NDArray[np.float64]) -> tuple:
        """
        Compute the intersection between a ray from the center and the geometry surface.

        Parameters
        ----------
        pos : ndarray of float64
            Input position(s).

        Returns
        -------
        tuple
            A tuple (position, distance) of intersection point and distance along the ray.
        """
        pass

    @abstractmethod
    def line_intersect(
        self, pos1: NDArray[np.float64], pos2: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Compute the intersection between a line segment and the geometry surface.

        Parameters
        ----------
        pos1 : ndarray of float64
            Starting point of the line segment.
        pos2 : ndarray of float64
            Ending point of the line segment.

        Returns
        -------
        ndarray of float64
            Intersection point(s) with the geometry surface.
        """
        pass

    @abstractmethod
    def f_ray_d(self, pos: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute the normalized ray distance; 1 indicates the surface.

        Parameters
        ----------
        pos : ndarray of float64
            Input position(s).

        Returns
        -------
        ndarray of float64
            Normalized distances; 1.0 means exactly on the surface.
        """
        pass

    def ray_point(self, pos: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Get the intersection point between ray and geometry surface.

        Parameters
        ----------
        pos : ndarray of float64
            Input position(s).

        Returns
        -------
        ndarray of float64
            The intersection point(s).
        """
        return self.ray_intersect(pos)[0]

    def ray_dist(self, pos: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Get the ray distance to the geometry surface.

        Parameters
        ----------
        pos : ndarray of float64
            Input position(s).

        Returns
        -------
        ndarray of float64
            Distance from input positions to the surface along the ray.
        """
        return self.ray_intersect(pos)[1]

    @staticmethod
    @abstractmethod
    def quick_call(*args, **kwargs) -> NDArray[np.float64]:
        """
        Quickly evaluate the geometry function with given parameters.

        Returns
        -------
        ndarray of float64
            Evaluated result.
        """
        pass

    @staticmethod
    @abstractmethod
    def quick_f_ray_d(*args, **kwargs) -> NDArray[np.float64]:
        """
        Quickly compute normalized ray distance with given parameters.

        Returns
        -------
        ndarray of float64
            Normalized ray distances.
        """
        pass

    @staticmethod
    def quick_ray_dist(*args, **kwargs) -> NDArray[np.float64]:
        """
        Quickly compute distance between points and corresponding ray-surface points.

        Returns
        -------
        ndarray of float64
            Distances to the surface.
        """
        pass

    @staticmethod
    def quick_line_intersect(*args, **kwargs) -> NDArray[np.float64]:
        """
        Quickly compute the intersection between a line segment and the geometry surface.

        Returns
        -------
        ndarray of float64
            Intersection point(s).
        """
        pass

    @staticmethod
    def quick_jacobian(*args, **kwargs) -> tuple:
        """
        Quickly compute the Jacobian of the geometry function.

        Returns
        -------
        tuple
            The Jacobian matrix.
        """
        pass


class Geometry:
    """
    Factory class for accessing registered Geometry plugins.

    This class provides static methods to load and retrieve available
    Geometry plugins derived from `GeometryBase`.

    Methods
    -------
    get_plugin(plugin)
        Retrieve a specific Geometry plugin by name.
    available_plugins
        List all available Geometry plugins.
    """

    @staticmethod
    def _update_plugin_stub():
        output_path = os.path.join(_current_dir, _pyi_name)
        generate_plugin_stub(Geometry, GeometryBase, _GeometryPlugins, output_path)
        logger.info(f"✅ Updated stub: {output_path}")

    @staticmethod
    def get_plugin(plugin: str | None) -> GeometryBase:
        """
        Get a registered geometry plugin.

        Parameters
        ----------
        plugin : str or None
            Name of the plugin. If None, returns the GeometryBase itself.

        Returns
        -------
        GeometryBase
            The plugin class.
        """
        assert (isinstance(plugin, str)) or (plugin is None), "The 'plugin' parameter must be a string or None."

        if plugin is None:
            return GeometryBase
        if not _GeometryPlugins:
            Geometry._load_plugin()
            
        return _GeometryPlugins[plugin]
    @staticmethod
    def _load_plugin():
        """
        Load geometry plugin modules dynamically.
        """
        import importlib 
        try:
            importlib.import_module("gal3d.shape.geometry_plugins")
            logger.info(f"Successfully loaded geometry plugins: {list(_GeometryPlugins.keys())}")
        except ModuleNotFoundError as e:
            logger.error(f"Failed to load geometry plugins: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading geometry plugins: {e}")
        
    @classproperty
    def available_plugins(cls) -> List[str]:
        """
        List all available geometry plugins.

        Returns
        -------
        list of str
            Names of registered geometry plugins.
        """
        if not _GeometryPlugins:
            cls._load_plugin()
        return list(_GeometryPlugins.keys())

