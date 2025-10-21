import abc
import numpy as np
from _typeshed import Incomplete
from abc import abstractmethod
from dataclasses import dataclass
from gal3d.plugin import PluginBase, PluginManager
from gal3d.util.func_cache import CacheDict
from numpy.typing import NDArray
from typing import Any, Callable, Sequence
from typing import Literal, overload
from gal3d.visualization.model_projector_plugins.projector_line_integration import ProjectorLineIntegration
from gal3d.visualization.model_projector_plugins.projector_sph_grid import ProjectorSphGrid

__all__ = ['ModelProjectorBase', 'ModelProjector']

@dataclass
class ImageData:
    """ Data class to hold projected image information."""
    value: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    xrange: tuple[float, float]
    yrange: tuple[float, float]
    @property
    def extent(self) -> tuple[float, float, float, float]: ...
    @property
    def nx(self) -> int: ...
    @property
    def ny(self) -> int: ...
    @property
    def pixel_area(self) -> float: ...
    @property
    def total_quantity(self) -> float: ...

class ModelProjectorBase(PluginBase, metaclass=abc.ABCMeta):
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
    def __init_subclass__(cls, **kwargs) -> None:
        """Register subclass as a ModelProjector plugin.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to the parent __init_subclass__.
        """
    _image_cache: CacheDict[tuple[float, float, float, float, int, float, float, bytes | None], NDArray[np.float64]]
    def __init__(self, cache_len: int = 100) -> None:
        """Initialize the model projector.

        Parameters
        ----------
        cache_len : int, default=100
            Maximum number of images to store in the cache.
        """
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
    @ImageCache
    def image(self, x_range: tuple[float, float], y_range: tuple[float, float], nbins: int = 100, z_range: tuple[float, float] = (-20, 20), rotation: NDArray[np.float64] | None = None, **kwargs: Any) -> ImageData:
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
    @abstractmethod
    def _image(self, x_range: tuple[float, float], y_range: tuple[float, float], nbins: int = 100, z_range: tuple[float, float] = (-20, 20), rotation: NDArray[np.float64] | None = None, **kwargs: Any) -> ImageData:
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
    def image_xz(self, x_range: tuple[float, float], y_range: tuple[float, float], nbins: int = 100, z_range: tuple[float, float] = (-20, 20)) -> ImageData:
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
    def image_yz(self, x_range: tuple[float, float], y_range: tuple[float, float], nbins: int = 100, z_range: tuple[float, float] = (-20, 20)) -> ImageData:
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

class ModelProjector(PluginManager[ModelProjectorBase]):
    """
    Factory class for accessing registered ModelProjector plugins.
    """
    _plugins: Incomplete
    _plugin_module: str
    _base_class = ModelProjectorBase

    @overload
    @classmethod
    def get_plugin(cls, name: Literal["ProjectorLineIntegration"]) -> type[ProjectorLineIntegration]: ...
    @overload
    @classmethod
    def get_plugin(cls, name: Literal["ProjectorSphGrid"]) -> type[ProjectorSphGrid]: ...
