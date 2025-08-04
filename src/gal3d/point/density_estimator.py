import logging
import os
from abc import abstractmethod
from functools import cached_property
from typing import List

import numpy as np

from .. import config_parser

from gal3d.plugin import PluginBase, PluginManager

__all__ = ['DensityEstimator', 'DensityEstimatorBase']

logger = logging.getLogger("gal3d.particle.density_estimator")


# TODO kernel
class DensityEstimatorBase(PluginBase):
    """
    Abstract base class for all density estimators.

    This base class provides a unified interface for implementing different types of 
    density estimators, which take particle positions and masses as input and estimate
    a physical parameter (e.g., density) and its gradient at arbitrary positions.

    Parameters
    ----------
    pos : array_like, shape (n, 3)
        Input particle positions.
    mass : array_like, shape (n,)
        Mass associated with each particle.
    parameter_mode : str, optional
        Mode of the parameter to estimate. Default is 'Density'.
    kernel : optional
        Smoothing kernel to use. Not implemented in base class.

    Attributes
    ----------
    parameter : ndarray, shape (n,)
        Cached estimated parameter at the original positions.
    gradient : tuple
        Cached estimated gradient of the parameter.
    """

    def __init__(self, pos, mass, parameter_mode: str = 'Density', kernel: None = None):

        self.pos = self._shape_check(pos)
        self.mass = mass

        self.pa_mode = parameter_mode
        self.kernel = kernel

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        DensityEstimatorManager.register(cls)

    def _shape_check(self, pos):
        """
        Ensure that the position array has shape (n, 3).

        Parameters
        ----------
        pos : array_like
            Input positions of particles.

        Returns
        -------
        numpy.ndarray
            Reshaped position array with shape (n, 3).
        """
        if len(np.shape(pos)) != 2:
            logger.debug(f"pos is 1d array with shape={np.shape(pos)}, reshaping to (-1,3)")
            pos = np.array(pos).reshape(-1, 3)
        if np.shape(pos)[1] == 3:
            return pos
        if np.shape(pos)[0] == 3:
            logger.debug(f"pos have the shape= {np.shape(pos)}, transposing it")
            return np.array(pos).T
        logger.debug(
            f"pos have the shape={np.shape(pos)}, target shape: (n,3), reshaping it"
        )
        return np.array(pos).reshape(-1, 3)

    @cached_property
    def parameter(self):
        """
        Estimate the parameter at the original particle positions.

        Returns
        -------
        numpy.ndarray
            Estimated parameter values.
        """
        return self.get_parameter(self.pos)

    @cached_property
    def gradient(self):
        """
        Estimate the gradient of the parameter at the original particle positions.

        Returns
        -------
        tuple
            Gradient estimation result.
        """
        return self.get_gradient(self.pos)

    @abstractmethod
    def get_hsm(self, target_pos, **kwargs):
        """
        Estimate the half-smooth length at the target positions.

        Parameters
        ----------
        target_pos : ndarray, shape (m, 3)
            Positions at which to evaluate the half-smooth length.
        **kwargs : dict
            Additional arguments passed to the internal method.

        Returns
        -------
        numpy.ndarray
            Estimated half-smooth length values.
        """
        pass

    @abstractmethod
    def get_parameter(self, target_pos, **kwargs):
        """
        Estimate the parameter value at the target positions.

        Parameters
        ----------
        target_pos : ndarray, shape (m, 3)
            Positions at which to evaluate the parameter.
        **kwargs : dict
            Additional arguments passed to the internal method.

        Returns
        -------
        ndarray, shape (m,)
            Estimated parameter values.
        """
        pass

    @abstractmethod
    def get_gradient(self, target_pos, **kwargs):
        """
        Estimate the gradient of the parameter at the target positions.

        Parameters
        ----------
        target_pos : ndarray, shape (m, 3)
            Positions at which to evaluate the gradient.
        **kwargs : dict
            Additional arguments passed to the internal method.

        Returns
        -------
        tuple
            Tuple of two elements:
            - Upward gradient: (magnitude, direction)
            - Downward gradient: (magnitude, direction)
        """
        pass


class DensityEstimatorManager(PluginManager[DensityEstimatorBase]):
    """
    Factory class for accessing registered density estimator plugins.
    """
    _plugins = {}
    _plugin_module = "gal3d.point.density_estimator_plugins"
    _base_class = DensityEstimatorBase

DensityEstimator = DensityEstimatorManager
