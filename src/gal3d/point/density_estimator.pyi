import abc
from abc import abstractmethod
from functools import cached_property
from typing import Any, Literal, overload

import numpy as np
from _typeshed import Incomplete
from numpy.typing import ArrayLike

from gal3d.plugin import PluginBase, PluginManager
from gal3d.point.density_estimator_plugins.estimator_sph import DensityEstimatorSPH
from gal3d.util.array_operate import Auto3DShape

__all__ = ["DensityEstimator", "DensityEstimatorBase"]

class DensityEstimatorBase(PluginBase, Auto3DShape, metaclass=abc.ABCMeta):
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

    Attributes
    ----------
    parameter : ndarray, shape (n,)
        Cached estimated parameter at the original positions.
    gradient : tuple
        Cached estimated gradient of the parameter.
    """
    pos: Incomplete
    mass: Incomplete
    def __init__(self, pos: ArrayLike, mass: np.ndarray) -> None: ...
    def __init_subclass__(cls, **kwargs: Any) -> None: ...
    @cached_property
    def parameter(self) -> np.ndarray:
        """
        Estimate the parameter at the original particle positions.

        Returns
        -------
        numpy.ndarray
            Estimated parameter values.
        """
    @cached_property
    def gradient(self) -> np.ndarray:
        """
        Estimate the gradient of the parameter at the original particle positions.

        Returns
        -------
        tuple
            Gradient estimation result.
        """
    @cached_property
    def hsm(self) -> np.ndarray:
        """
        Estimate the half-smooth length at the original particle positions.

        Returns
        -------
        numpy.ndarray
            Estimated half-smooth length values.
        """
    @abstractmethod
    def get_hsm(self, target_pos: ArrayLike, **kwargs: Any) -> np.ndarray:
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
    @abstractmethod
    def get_parameter(self, target_pos: ArrayLike, **kwargs: Any) -> np.ndarray:
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
    @abstractmethod
    def get_gradient(self, target_pos: ArrayLike, **kwargs: Any) -> np.ndarray:
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
        gradient: array, shape(m, 3)
            The estimated gradients at the target positions.
        """

class DensityEstimator(PluginManager[DensityEstimatorBase]):
    """
    Factory class for accessing registered density estimator plugins.
    """

    @overload
    @classmethod
    def get_plugin(cls, name: Literal["DensityEstimatorSPH"]) -> type[DensityEstimatorSPH]: ...
    @overload
    @classmethod
    def get_plugin(cls, name: str) -> type[DensityEstimatorBase]: ...
