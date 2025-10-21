import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from gal3d.plugin import PluginBase, PluginManager
from gal3d.util.array_operate import Auto3DShape

from .with_parameter import WithParameter, abstractmethod

__all__ = ["Coordinate", "CoordinateBase"]

logger = logging.getLogger("gal3d.shape.coordinate")

class CoordinateBase(WithParameter, PluginBase,Auto3DShape):
    """
    Abstract base class for coordinate transformation plugins.

    Defines the interface for forward/inverse transformations and Jacobians.

    Methods
    -------
    __call__(pos):
        Transforms the given position using the current coordinate system.
    jacobian(pos):
        Computes the Jacobian of the transformation at the given positions.
    inverse(pos):
        Inverse transforms the given position.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Register a new coordinate plugin subclass and update the plugin stub.
        """
        super().__init_subclass__(**kwargs)
        valid = getattr(cls, "_parameter_valid", True)
        delattr(cls, "_parameter_valid")
        if not valid:
            logger.warning("CoordinatePlugin found: %s but failed to load",
                           cls.__name__)
            return
        Coordinate.register(cls)

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

    @staticmethod
    @abstractmethod
    def quick_call(*args: Any, **kwargs: Any) -> NDArray[np.float64]:
        """
        Fast version of the coordinate transformation.

        Returns
        -------
        ndarray of float64
            Transformed positions.
        """

    @staticmethod
    @abstractmethod
    def quick_jacobian(*args: Any, **kwargs: Any) -> tuple:
        """
        Fast version of the Jacobian computation.

        Returns
        -------
        tuple
            Jacobian matrices.
        """

    @staticmethod
    @abstractmethod
    def quick_inverse(*args: Any, **kwargs: Any) -> NDArray[np.float64]:
        """
        Fast version of the inverse coordinate transformation.

        Returns
        -------
        ndarray of float64
            Inverse-transformed positions.
        """


class Coordinate(PluginManager[CoordinateBase]):
    """
    Factory class for accessing registered Coordinate plugins.
    """
    _plugins = {}
    _plugin_module = "gal3d.shape.coordinate_plugins"
    _base_class = CoordinateBase

