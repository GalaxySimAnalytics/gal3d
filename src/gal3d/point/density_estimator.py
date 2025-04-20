import os
import logging
from abc import ABC, abstractmethod
from typing import List
from functools import cached_property

import numpy as np

from ..util.func_signature import generate_plugin_stub
from ..util.func_decorator import classproperty
from .. import config

__all__ = ['DensityEstimator', 'DensityEstimatorBase']

logger = logging.getLogger("gal3d.particle.density_estimator")


_DensityEstimatorPlugins = dict()

_current_path = os.path.realpath(__file__)
_current_dir = os.path.dirname(__file__)
_current_file_name = os.path.basename(_current_path)
_pyi_name = _current_file_name.replace('.py', '.pyi')


# TODO kernel
class DensityEstimatorBase(ABC):
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
        _DensityEstimatorPlugins[cls.__name__] = cls
        logger.info(
            f"Find DensityEstimatorPlugin: {cls.__name__} and load successfully"
        )
        if config['update_stub']:
            output_path = os.path.join(_current_dir, _pyi_name)
            generate_plugin_stub(
                DensityEstimator,
                DensityEstimatorBase,
                _DensityEstimatorPlugins,
                output_path,
            )
            logger.info(f"✅ Updated stub: {output_path}")

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
            logger.info(f"pos is 1d array with shape={np.shape(pos)}, reshaping to (-1,3)")
            pos = np.array(pos).reshape(-1, 3)
        if np.shape(pos)[1] == 3:
            return pos
        if np.shape(pos)[0] == 3:
            logger.info(f"pos have the shape= {np.shape(pos)}, transposing it")
            return np.array(pos).T
        logger.info(
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


class DensityEstimator:
    """
    Factory class for accessing registered density estimator plugins.

    This class provides static methods to load and retrieve available
    density estimator plugins derived from `DensityEstimatorBase`.

    Methods
    -------
    get_plugin(plugin)
        Retrieve a specific DensityEstimator plugin by name.
    available_plugins
        List all available DensityEstimator plugins.
    """

    @staticmethod
    def _updata_plugin_stub():
        output_path = os.path.join(_current_dir, _pyi_name)
        generate_plugin_stub(
            DensityEstimator,
            DensityEstimatorBase,
            _DensityEstimatorPlugins,
            output_path,
        )
        logger.info(f"✅ Updated stub: {output_path}")

    @staticmethod
    def get_plugin(plugin: str | None) -> DensityEstimatorBase:
        """
        Get a specific DensityEstimator plugin class.

        Parameters
        ----------
        plugin : str or None
            The name of the plugin. If None, returns the base class.

        Returns
        -------
        type
            A class derived from DensityEstimatorBase.
        """
        assert (isinstance(plugin, str)) or (plugin is None)

        if plugin is None:
            return DensityEstimatorBase
        if not _DensityEstimatorPlugins:
            DensityEstimator._load_plugin()
        return _DensityEstimatorPlugins[plugin]

    @staticmethod
    def _load_plugin():
        """
        Load all registered plugins under 'gal3d.point.density_estimator_plugins'.
        """
        import importlib
        importlib.import_module("gal3d.point.density_estimator_plugins")
        logger.info("Successfully loaded density estimator plugins")
        
    @classproperty
    def available_plugins(cls) -> List[str]:
        """
        List all available registered DensityEstimator plugin names.

        Returns
        -------
        list of str
            Names of available plugins.
        """
        if not _DensityEstimatorPlugins:
            cls._load_plugin()
        return list(_DensityEstimatorPlugins.keys())

