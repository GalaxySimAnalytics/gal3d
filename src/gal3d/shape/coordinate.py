import logging
import os
from typing import List

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .. import config_parser
from .with_parameter import Parameters, WithParameter, abstractmethod

from gal3d.plugin import PluginBase, PluginManager

__all__ = ['Coordinate', 'CoordinateBase']

logger = logging.getLogger("gal3d.shape.coordinate")

class CoordinateBase(WithParameter, PluginBase):
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

    def __init_subclass__(cls, **kwargs):
        """
        Register a new coordinate plugin subclass and update the plugin stub.
        """

        if not super().__init_subclass__():
            logger.warning(f"CoordinatePlugin found: {cls.__name__} but failed to load")
            return
        CoordinateManager.register(cls)

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


class CoordinateManager(PluginManager[CoordinateBase]):
    """
    Factory class for accessing registered Coordinate plugins.
    """
    _plugins = {}
    _plugin_module = "gal3d.shape.coordinate_plugins"
    _base_class = CoordinateBase


Coordinate = CoordinateManager
