import logging
import abc
from typing import List, NoReturn, Sequence
from functools import wraps
import logging
import os

import numpy as np
from numpy.typing import NDArray

from ..util.func_cache import CacheDict
from ..util.func_signature import generate_plugin_stub
from ..util.func_decorator import classproperty
from .. import config_parser

__all__ = ['ModelProjectorBase', 'ModelProjector']

logger = logging.getLogger("gal3d.visualization.model_projector")

_ModelProjectorPlugins = dict()

_current_path = os.path.realpath(__file__)
_current_dir = os.path.dirname(__file__)
_current_file_name = os.path.basename(_current_path)
_pyi_name = _current_file_name.replace('.py', '.pyi')


class ModelProjectorBase(abc.ABC):

    def __init_subclass__(cls, **kwargs):

        _ModelProjectorPlugins[cls.__name__] = cls
        logger.info(f"Find ModelProjectorPlugin: {cls.__name__} and load successfully")
        if config_parser['general'].getboolean("update_stub"):
            output_path = os.path.join(_current_dir, _pyi_name)
            generate_plugin_stub(
                ModelProjector, ModelProjectorBase, _ModelProjectorPlugins, output_path
            )
            logger.info(f"✅ Updated stub: {output_path}")

    def __init__(self, cache_len: int = 100):
        self._image_cache = CacheDict(cache_len=cache_len)

    def ImageCache(func):
        @wraps(func)
        def wrapper(self, x_range, y_range, nbins, z_range, rotation, **kwargs):
            recod = (
                x_range[0],
                x_range[1],
                y_range[0],
                y_range[1],
                nbins,
                z_range[0],
                z_range[1],
                rotation.tobytes(),
            )
            if recod in self._image_cache:
                logger.info(f"Get image from cache for input hash: {recod}")
                return self._image_cache[recod]
            else:
                logger.info(f"Cache image, register input hash: {recod}")
                self._image_cache[recod] = func(
                    self, x_range, y_range, nbins, z_range, rotation, **kwargs
                )
            return self._image_cache[recod]

        return wrapper

    @ImageCache
    def image(
        self,
        x_range,
        y_range,
        nbins: int = 100,
        z_range: Sequence = (-20, 20),
        rotation: None | NDArray[np.float64] = None,
        **kwargs,
    ):

        if rotation is None:
            rotation = np.eye(3)

        return self._image(
            x_range, y_range, nbins, z_range=z_range, rotation=rotation, **kwargs
        )

    @abc.abstractmethod
    def _image(
        self,
        x_range,
        y_range,
        nbins: int = 100,
        z_range: Sequence = (-20, 20),
        rotation: None | NDArray[np.float64] = None,
        **kwargs,
    ):
        pass

    def image_xz(
        self,
        x_range,
        y_range,
        nbins: int = 100,
        z_range: Sequence = (-20, 20),
    ):
        return self.image(
            x_range,
            y_range,
            nbins,
            z_range,
            rotation=np.array([[1.0, 0, 0], [0, 0, 1.0], [0, 1.0, 0.0]]).T,
        )

    def image_yz(
        self,
        x_range,
        y_range,
        nbins: int = 100,
        z_range: Sequence = (-20, 20),
    ):
        return self.image(
            x_range,
            y_range,
            nbins,
            z_range,
            rotation=np.array([[0, 1.0, 0.0], [0, 0, 1.0], [1.0, 0, 0.0]]).T,
        )


class ModelProjector:

    @staticmethod
    def _updata_plugin_stub():
        output_path = os.path.join(_current_dir, _pyi_name)
        generate_plugin_stub(
            ModelProjector, ModelProjectorBase, _ModelProjectorPlugins, output_path
        )
        logger.info(f"✅ Updated stub: {output_path}")

    @staticmethod
    def get_plugin(plugin: str | None) -> ModelProjectorBase:
        """
        Get an geometry plugin

        Parameters:
        plugin: str,
            the name of plugin, available see available_plugins

        Returns:
            available_plugins of ModelProjectorBase
        """
        assert (isinstance(plugin, str)) or (plugin is None)

        if plugin is None:
            return ModelProjectorBase
        if not _ModelProjectorPlugins:
            ModelProjector._load_plugin()
        return _ModelProjectorPlugins[plugin]
    @staticmethod
    def _load_plugin():
        import importlib
        importlib.import_module("gal3d.visualization.model_projector_plugins")
        logger.info("Successfully loaded model projector plugins")
        
    @classproperty
    def available_plugins(cls) -> List[str]:
        if not _ModelProjectorPlugins:
            cls._load_plugin()
        return list(_ModelProjectorPlugins.keys())


from .model_projector_plugins import *
