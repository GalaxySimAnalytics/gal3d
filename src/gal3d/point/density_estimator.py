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
        '''
        Ensure the input positions have the correct shape (n, 3).

        Parameters:
            pos: ndarray
                The input positions to be checked and reshaped if necessary.

        Returns:
            pos: ndarray, shape(n,3)
                The reshaped positions.
        '''
        if len(np.shape(pos)) != 2:
            logger.info(f"pos is 1d array with shape={np.shape(pos)}, so we reshape it")
            pos = np.array(pos).reshape(-1, 3)
        if np.shape(pos)[1] == 3:
            return pos
        if np.shape(pos)[0] == 3:
            logger.info(f"pos have the shape= {np.shape(pos)}, so we transpose it")
            return np.array(pos).T
        logger.info(
            f"pos have the shape={np.shape(pos)}, target shape: (n,3), so we reshape this"
        )
        return np.array(pos).reshape(-1, 3)

    @cached_property
    def parameter(self):
        '''Cached property that returns the parameter values at the input positions.'''
        return self.get_parameter(self.pos)

    @cached_property
    def gradient(self):
        '''Cached property that returns the gradient of the parameter at the input positions.'''
        return self.get_gradient(self.pos)

    @abstractmethod
    def get_parameter(self, target_pos, **kwargs):
        '''
        Estimate the parameter value at the target positions.

        Parameters:
            target_pos: ndarray, shape(m,3)
                The target positions (x, y, z) where the parameter values are to be estimated.
            **kwargs: dict, optional
                Additional keyword arguments passed to the KDTree query method.

        Returns:
            results: array, shape(m,)
                The estimated parameter values at the target positions.
        '''
        pass

    @abstractmethod
    def get_gradient(self, target_pos, **kwargs):
        '''
        Estimate the gradient of the parameter at the target positions.

        Parameters:
            target_pos: ndarray, shape(m,3)
                The target positions (x, y, z) where the gradient is to be estimated.
            **kwargs: dict, optional
                Additional keyword arguments passed to the KDTree query method.

        Returns:
            gradient: tuple of tuples
                A tuple containing two tuples:
                - The first tuple contains the upward gradient magnitude and direction.
                - The second tuple contains the downward gradient magnitude and direction.
        '''
        pass


class DensityEstimator:
    """DensityEstimator"""

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
        Get an DensityEstimator plugin

        Parameters:
        plugin: str,
            the name of plugin, available see available_plugins

        Returns:
            available_plugins of DensityEstimatorBase
        """
        assert (isinstance(plugin, str)) or (plugin is None)

        if plugin is None:
            return DensityEstimatorBase
        if not _DensityEstimatorPlugins:
            DensityEstimator._load_plugin()
        return _DensityEstimatorPlugins[plugin]

    @staticmethod
    def _load_plugin():
        import importlib
        importlib.import_module("gal3d.point.density_estimator_plugins")
        logger.info("Successfully loaded density estimator plugins")
        
    @classproperty
    def available_plugins(cls) -> List[str]:
        if not _DensityEstimatorPlugins:
            cls._load_plugin()
        return list(_DensityEstimatorPlugins.keys())

