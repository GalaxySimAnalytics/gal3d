import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from gal3d.plugin import PluginBase, PluginManager
from gal3d.util.array_operate import Auto3DShape

from .with_parameter import WithParameter, abstractmethod

__all__ = ["Geometry", "GeometryBase"]

logger = logging.getLogger("gal3d.shape.geometry")


class GeometryBase(WithParameter,PluginBase,Auto3DShape):
    """Abstract base class for geometry models."""

    def __init_subclass__(cls, **kwargs: Any):
        """
        Automatically registers geometry plugins upon subclass initialization.
        """
        super().__init_subclass__(**kwargs)
        valid = getattr(cls, "_parameter_valid", True)
        delattr(cls, "_parameter_valid")
        if not valid:
            logger.warning("GeometryPlugin found: %s but failed to load",
                           cls.__name__)
            return
        Geometry.register(cls)

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
    def quick_call(*args: Any, **kwargs: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Quickly evaluate the geometry function with given parameters.

        Returns
        -------
        ndarray of float64
            Evaluated result.
        """

    @staticmethod
    @abstractmethod
    def quick_f_ray_d(*args: Any, **kwargs: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Quickly compute normalized ray distance with given parameters.

        Returns
        -------
        ndarray of float64
            Normalized ray distances.
        """

    @staticmethod
    @abstractmethod
    def quick_ray_dist(*args: Any, **kwargs: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Quickly compute distance between points and corresponding ray-surface points.

        Returns
        -------
        ndarray of float64
            Distances to the surface.
        """

    @staticmethod
    @abstractmethod
    def quick_line_intersect(*args: Any, **kwargs: Any) -> NDArray[np.float64]:
        """
        Quickly compute the intersection between a line segment and the geometry surface.

        Returns
        -------
        ndarray of float64
            Intersection point(s).
        """

    @staticmethod
    @abstractmethod
    def quick_jacobian(*args: Any, **kwargs: Any) -> tuple:
        """
        Quickly compute the Jacobian of the geometry function.

        Returns
        -------
        tuple
            The Jacobian matrix.
        """


class Geometry(PluginManager[GeometryBase]):
    """
    Factory class for accessing registered Geometry plugins.
    """
    _plugins = {}
    _plugin_module = "gal3d.shape.geometry_plugins"
    _base_class = GeometryBase

