"""
Model projector base class and factory for generating 2D projections from 3D models.
"""
import logging
from abc import abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import wraps
from typing import Any

import numpy as np
from numpy.typing import NDArray

from gal3d.plugin import PluginBase, PluginManager
from gal3d.util.func_cache import CacheDict

__all__ = ["ModelProjectorBase", "ModelProjector"]

logger = logging.getLogger("gal3d.visualization.model_projector")


@dataclass
class ImageData:
    """ Data class to hold projected image information."""
    value: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    xrange: tuple[float, float]
    yrange: tuple[float, float]

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return (self.xrange[0], self.xrange[1], self.yrange[0], self.yrange[1])

    @property
    def nx(self) -> int:
        return self.xs.size

    @property
    def ny(self) -> int:
        return self.ys.size

    @property
    def pixel_area(self) -> float:
        return (self.xrange[1] - self.xrange[0]) * (self.yrange[1] - self.yrange[0]) / (self.nx * self.ny)
    @property
    def total_quantity(self) -> float:
        return np.sum(self.value) * self.pixel_area

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

    def __init_subclass__(cls, **kwargs: Any) -> None:
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
        self._image_cache: CacheDict[tuple[float, float, float, float, int, float, float, bytes | None], NDArray[np.float64]]
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
        def wrapper(self: "ModelProjectorBase", x_range: tuple[float, float], y_range: tuple[float, float],
                   nbins: int, z_range: tuple[float, float], rotation: NDArray[np.float64] | None = None,
                   **kwargs: Any) -> NDArray[np.float64]:
            rotation_bytes = rotation.tobytes() if rotation is not None else None
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
                logger.debug("Get image from cache for input: x:%s, y:%s, "
                             "z:%s, nbins:%d, rotation:%s",
                             x_range, y_range, z_range, nbins, rotation)
                return self._image_cache[recod]
            else:
                logger.debug("Cache image, register input: x:%s, y:%s, "
                             "z:%s, nbins:%d, rotation:%s",
                             x_range, y_range, z_range, nbins, rotation)
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
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        nbins: int = 100,
        z_range: tuple[float, float] = (-20, 20),
        rotation: NDArray[np.float64] | None = None,
        **kwargs: Any
    ) -> ImageData:
        """Generate a projected image of the model with caching.

        Parameters
        ----------
        x_range : Tuple of float
            The (min, max) range in x-direction for the projection.
        y_range : Tuple of float
            The (min, max) range in y-direction for the projection.
        nbins : int, default=100
            Number of bins in each dimension for the projection.
        z_range : Tuple of float, default=(-20, 20)
            The (min, max) range in z-direction to include in the projection.
        rotation : ndarray, optional
            3x3 rotation matrix to apply to the model before projection.
            If None, no rotation is applied (identity matrix).
        **kwargs : dict
            Additional keyword arguments for specific projection implementations.

        Returns
        -------
        ImageData
            The projected image data.

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

        if rotation is not None and rotation.shape != (3, 3):
            raise ValueError(f"Rotation matrix must be 3x3, got {rotation.shape}")

        return self._image(
            x_range, y_range, nbins, z_range=z_range, rotation=rotation, **kwargs
        )

    @abstractmethod
    def _image(
        self,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        nbins: int = 100,
        z_range: tuple[float, float] = (-20, 20),
        rotation: NDArray[np.float64] | None = None,
        **kwargs: Any
    ) -> ImageData:
        """Abstract method to generate a projected image.

        This method must be implemented by subclasses to perform
        the actual image generation.

        Parameters
        ----------
        x_range : Tuple of float
            The (min, max) range in x-direction for the projection.
        y_range : Tuple of float
            The (min, max) range in y-direction for the projection.
        nbins : int, default=100
            Number of bins in each dimension for the projection.
        z_range : Tuple of float, default=(-20, 20)
            The (min, max) range in z-direction to include in the projection.
        rotation : ndarray, default=identity matrix
            3x3 rotation matrix to apply to the model before projection.
        **kwargs : dict
            Additional keyword arguments for specific projection methods.

        Returns
        -------
        ImageData
            The projected image data.
        """
        raise NotImplementedError("Subclasses must implement the '_image' method to generate a projected image.")

    def image_xz(
        self,
        x_range: tuple[float,float],
        y_range: tuple[float,float],
        nbins: int = 100,
        z_range: tuple[float,float] = (-20, 20),
    ) -> ImageData:
        """Generate a projection in the x-z plane.

        This is a convenience method that applies the appropriate
        rotation (transposed) to view the model from the y direction.

        Parameters
        ----------
        x_range : Tuple of float
            The (min, max) range in x-direction for the projection.
        y_range : Tuple of float
            The (min, max) range in z-direction for the projection (after rotation).
        nbins : int, default=100
            Number of bins in each dimension for the projection.
        z_range : Tuple of float, default=(-20, 20)
            The (min, max) range in y-direction (after rotation) to include in the projection.

        Returns
        -------
        ImageData
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
        x_range: tuple[float,float],
        y_range: tuple[float,float],
        nbins: int = 100,
        z_range: tuple[float,float] = (-20, 20),
    ) -> ImageData:
        """Generate a projection in the y-z plane.

        This is a convenience method that applies the appropriate
        rotation (transposed) to view the model from the x direction.

        Parameters
        ----------
        x_range : Tuple of float
            The (min, max) range in y-direction for the projection (after rotation).
        y_range : Tuple of float
            The (min, max) range in z-direction for the projection (after rotation).
        nbins : int, default=100
            Number of bins in each dimension for the projection.
        z_range : Tuple of float, default=(-20, 20)
            The (min, max) range in x-direction (after rotation) to include in the projection.

        Returns
        -------
        ImageData
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
