import logging
import abc
from typing import List, NoReturn, Sequence, Optional, Dict, Any, Tuple, Union, Callable
from functools import wraps
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
    """Abstract base class for model projectors that generate 2D projections from 3D models.

    This class provides a framework for creating different types of model projectors
    that can generate 2D projections of 3D models. It includes functionality for:
    - Automatic plugin registration of subclasses
    - Image caching to avoid recomputation
    - Standard rotation matrices for different viewing angles
    - Abstract methods that subclasses must implement

    Attributes
    ----------
    _image_cache : CacheDict
        Cache for storing previously computed images.
    """

    def __init_subclass__(cls, **kwargs):
        """Register subclass as a ModelProjector plugin.
        
        This method is automatically called when a subclass is created.
        It registers the subclass in the _ModelProjectorPlugins dictionary
        and updates the stub file when configured to do so.
        
        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to the parent __init_subclass__.
        """

        _ModelProjectorPlugins[cls.__name__] = cls
        logger.info(f"Find ModelProjectorPlugin: {cls.__name__} and load successfully")
        if config_parser['general'].getboolean("update_stub"):
            output_path = os.path.join(_current_dir, _pyi_name)
            generate_plugin_stub(
                ModelProjector, ModelProjectorBase, _ModelProjectorPlugins, output_path
            )
            logger.info(f"✅ Updated stub: {output_path}")

    def __init__(self, cache_len: int = 100):
        """Initialize the model projector.
        
        Parameters
        ----------
        cache_len : int, default=100
            Maximum number of images to store in the cache.
        """
        self._image_cache = CacheDict(cache_len=cache_len)

    @staticmethod
    def ImageCache(func: Callable) -> Callable:
        """Decorator for caching image results based on input parameters.
        
        Prevents recomputation of images with the same input parameters.
        
        Parameters
        ----------
        func : callable
            The function to be decorated, typically the _image method.
            
        Returns
        -------
        callable
            Wrapped function that includes caching behavior.
        """
        @wraps(func)
        def wrapper(self, x_range: Sequence[float], y_range: Sequence[float], 
                   nbins: int, z_range: Sequence[float], rotation: NDArray[np.float64], 
                   **kwargs: Any) -> NDArray[np.float64]:
            rotation_bytes = rotation.tobytes()
            recod = (
                x_range[0],
                x_range[1],
                y_range[0],
                y_range[1],
                nbins,
                z_range[0],
                z_range[1],
                rotation_bytes,
            )
            if recod in self._image_cache:
                logger.info(f"Get image from cache for input: x:{x_range}, y:{y_range}, z:{z_range}, rotation:{rotation}, nbins:{nbins}")
                return self._image_cache[recod]
            else:
                logger.info(f"Cache image, register input: x:{x_range}, y:{y_range}, z:{z_range}, rotation:{rotation}, nbins:{nbins}")
                self._image_cache[recod] = func(
                    self, x_range, y_range, nbins, z_range, rotation, **kwargs
                )
            return self._image_cache[recod]

        return wrapper

    @ImageCache
    def image(
        self,
        x_range: Sequence[float],
        y_range: Sequence[float],
        nbins: int = 100,
        z_range: Sequence[float] = (-20, 20),
        rotation: Optional[NDArray[np.float64]] = None,
        **kwargs: Any
    ) -> NDArray[np.float64]:
        """Generate a projected image of the model with caching.
        
        Parameters
        ----------
        x_range : sequence of float
            The (min, max) range in x-direction for the projection.
        y_range : sequence of float
            The (min, max) range in y-direction for the projection.
        nbins : int, default=100
            Number of bins in each dimension for the projection.
        z_range : sequence of float, default=(-20, 20)
            The (min, max) range in z-direction to include in the projection.
        rotation : ndarray, optional
            3x3 rotation matrix to apply to the model before projection.
            If None, no rotation is applied (identity matrix).
        **kwargs : dict
            Additional keyword arguments for specific projection implementations.
            
        Returns
        -------
        ndarray
            The projected image.
            
        Raises
        ------
        ValueError
            If provided ranges are invalid or inconsistent.
        """
        # Input validation
        if len(x_range) != 2 or x_range[0] >= x_range[1]:
            raise ValueError(f"Invalid x_range: {x_range}. Must be (min, max) with min < max")
        if len(y_range) != 2 or y_range[0] >= y_range[1]:
            raise ValueError(f"Invalid y_range: {y_range}. Must be (min, max) with min < max")
        if len(z_range) != 2 or z_range[0] >= z_range[1]:
            raise ValueError(f"Invalid z_range: {z_range}. Must be (min, max) with min < max")
        if nbins <= 0:
            raise ValueError(f"Invalid nbins: {nbins}. Must be positive")

        if rotation is None:
            rotation = np.eye(3).copy()
        elif rotation.shape != (3, 3):
            raise ValueError(f"Rotation matrix must be 3x3, got {rotation.shape}")

        return self._image(
            x_range, y_range, nbins, z_range=z_range, rotation=rotation, **kwargs
        )

    @abc.abstractmethod
    def _image(
        self,
        x_range: Sequence[float],
        y_range: Sequence[float],
        nbins: int = 100,
        z_range: Sequence[float] = (-20, 20),
        rotation: NDArray[np.float64] = np.eye(3),
        **kwargs: Any
    ) -> NDArray[np.float64]:
        """Abstract method to generate a projected image.
        
        This method must be implemented by subclasses to perform
        the actual image generation.
        
        Parameters
        ----------
        x_range : sequence of float
            The (min, max) range in x-direction for the projection.
        y_range : sequence of float
            The (min, max) range in y-direction for the projection.
        nbins : int, default=100
            Number of bins in each dimension for the projection.
        z_range : sequence of float, default=(-20, 20)
            The (min, max) range in z-direction to include in the projection.
        rotation : ndarray, default=identity matrix
            3x3 rotation matrix to apply to the model before projection.
        **kwargs : dict
            Additional keyword arguments for specific projection methods.
            
        Returns
        -------
        ndarray
            The projected image.
        """
        pass

    def image_xz(
        self,
        x_range,
        y_range,
        nbins: int = 100,
        z_range: Sequence = (-20, 20),
    ):
        """Generate a projection in the x-z plane.
        
        This is a convenience method that applies the appropriate
        rotation to view the model from the y direction.
        
        Parameters
        ----------
        x_range : sequence of float
            The (min, max) range in x-direction for the projection.
        y_range : sequence of float
            The (min, max) range in z-direction for the projection (after rotation).
        nbins : int, default=100
            Number of bins in each dimension for the projection.
        z_range : sequence of float, default=(-20, 20)
            The (min, max) range in y-direction (after rotation) to include in the projection.
            
        Returns
        -------
        ndarray
            The projected image in the x-z plane.
        """
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
        """Generate a projection in the y-z plane.
        
        This is a convenience method that applies the appropriate
        rotation to view the model from the x direction.
        
        Parameters
        ----------
        x_range : sequence of float
            The (min, max) range in y-direction for the projection (after rotation).
        y_range : sequence of float
            The (min, max) range in z-direction for the projection (after rotation).
        nbins : int, default=100
            Number of bins in each dimension for the projection.
        z_range : sequence of float, default=(-20, 20)
            The (min, max) range in x-direction (after rotation) to include in the projection.
            
        Returns
        -------
        ndarray
            The projected image in the y-z plane.
        """
        return self.image(
            x_range,
            y_range,
            nbins,
            z_range,
            rotation=np.array([[0, 1.0, 0.0], [0, 0, 1.0], [1.0, 0, 0.0]]).T,
        )


class ModelProjector:

    @staticmethod
    def _update_plugin_stub():
        output_path = os.path.join(_current_dir, _pyi_name)
        generate_plugin_stub(
            ModelProjector, ModelProjectorBase, _ModelProjectorPlugins, output_path
        )
        logger.info(f"✅ Updated stub: {output_path}")

    @staticmethod
    def get_plugin(plugin: str | None) -> ModelProjectorBase:
        """Get an geometry plugin

        Parameters:
        plugin: str,
            the name of plugin, available see available_plugins

        Returns:
            available_plugins of ModelProjectorBase
        """
        if not isinstance(plugin, (str, type(None))):
            raise TypeError("plugin must be a string or None")

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
            if not _ModelProjectorPlugins:
                logger.warning("No plugins were loaded. The _ModelProjectorPlugins dictionary is empty.")
        return list(_ModelProjectorPlugins.keys())


from .model_projector_plugins import ProjectorLineIntegration,ProjectorSphGrid
