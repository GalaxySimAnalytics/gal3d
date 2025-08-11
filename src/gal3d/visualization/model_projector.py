from abc import abstractmethod
import logging
import os
from functools import wraps
from typing import Any, Callable, Dict, List, NoReturn, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ..util.func_cache import CacheDict

from gal3d.plugin import PluginBase, PluginManager

__all__ = ['ModelProjectorBase', 'ModelProjector']

logger = logging.getLogger("gal3d.visualization.model_projector")

class ModelProjectorBase(PluginBase):
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
        
        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to the parent __init_subclass__.
        """
        super().__init_subclass__(**kwargs)
        ModelProjector.register(cls)


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
                logger.debug(f"Get image from cache for input: x:{x_range}, y:{y_range}, z:{z_range}, nbins:{nbins}, rotation:{rotation}")
                return self._image_cache[recod]
            else:
                logger.debug(f"Cache image, register input: x:{x_range}, y:{y_range}, z:{z_range}, nbins:{nbins}, rotation:{rotation}")
                self._image_cache[recod] = func(
                    self, x_range, y_range, nbins, z_range, rotation, **kwargs
                )
            return self._image_cache[recod]

        return wrapper

    def _validate_range(self, value_range: Sequence[float], name: str) -> None:
        """Validate that a range is a tuple of two floats with min < max.
        
        Parameters
        ----------
        value_range : sequence of float
            The range to validate.
        name : str
            The name of the range for error messages.
        
        Raises
        ------
        ValueError
            If the range is invalid.
        """
        if len(value_range) != 2 or value_range[0] >= value_range[1]:
            raise ValueError(f"Invalid {name}: {value_range}. Must be (min, max) with min < max")

    @ImageCache
    def image(
        self,
        x_range: Sequence[float],
        y_range: Sequence[float],
        nbins: int = 100,
        z_range: Sequence[float] = (-20, 20),
        rotation: Optional[NDArray[np.float64]] = None,
        **kwargs: Any
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
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
        tuple
            A tuple containing:
            - deproject_array.T (numpy.ndarray): The transposed 2D array of integrated values.
            - xs (numpy.ndarray): The x-coordinates of the bin centers.
            - ys (numpy.ndarray): The y-coordinates of the bin centers.

        Raises
        ------
        ValueError
            If provided ranges are invalid or inconsistent.
        """
        # Input validation
        self._validate_range(x_range, "x_range")
        self._validate_range(y_range, "y_range")
        self._validate_range(z_range, "z_range")
        if nbins <= 0:
            raise ValueError(f"Invalid nbins: {nbins}. Must be positive")

        if rotation is None:
            rotation = np.eye(3).copy()
        elif rotation.shape != (3, 3):
            raise ValueError(f"Rotation matrix must be 3x3, got {rotation.shape}")

        return self._image(
            x_range, y_range, nbins, z_range=z_range, rotation=rotation, **kwargs
        )

    @abstractmethod
    def _image(
        self,
        x_range: Sequence[float],
        y_range: Sequence[float],
        nbins: int = 100,
        z_range: Sequence[float] = (-20, 20),
        rotation: NDArray[np.float64] = np.eye(3),
        **kwargs: Any
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
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
        tuple: A tuple containing:
            - deproject_array.T (numpy.ndarray): The transposed 2D array of integrated values.
            - xs (numpy.ndarray): The x-coordinates of the bin centers.
            - ys (numpy.ndarray): The y-coordinates of the bin centers.
        """
        raise NotImplementedError("Subclasses must implement the '_image' method to generate a projected image.")

    def image_xz(
        self,
        x_range,
        y_range,
        nbins: int = 100,
        z_range: Sequence = (-20, 20),
    ):
        """Generate a projection in the x-z plane.
        
        This is a convenience method that applies the appropriate
        rotation (transposed) to view the model from the y direction.
        
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
        tuple
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
        rotation (transposed) to view the model from the x direction.
        
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
        tuple
            The projected image in the y-z plane.
        """
        return self.image(
            x_range,
            y_range,
            nbins,
            z_range,
            rotation=np.array([[0, 1.0, 0.0], [0, 0, 1.0], [1.0, 0, 0.0]]).T,
        )


class ModelProjector(PluginManager[ModelProjectorBase]):
    """
    Factory class for accessing registered ModelProjector plugins.
    """
    
    _plugins = {}
    _plugin_module = "gal3d.visualization.model_projector_plugins"
    _base_class = ModelProjectorBase


# Removed unused imports: ProjectorLineIntegration, ProjectorSphGrid
