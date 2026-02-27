import abc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, overload

import numpy as np
from numpy.typing import NDArray

from gal3d.plugin import PluginBase, PluginManager
from gal3d.visualization.model_projector_plugins.projector_line_integration import ProjectorLineIntegration
from gal3d.visualization.model_projector_plugins.projector_sph_grid import ProjectorSphGrid

__all__ = ["ModelProjectorBase", "ModelProjector"]

@dataclass
class ImageData:
    """
    Container for a projected image and its grid information.

    Attributes
    ----------
    value : np.ndarray
        Image array of shape (ny, nx).
    xs : np.ndarray
        X bin centers of shape (nx,).
    ys : np.ndarray
        Y bin centers of shape (ny,).
    xrange : tuple[float, float]
        (xmin, xmax)
    yrange : tuple[float, float]
        (ymin, ymax)
    """
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
    """
    Abstract base class for model projectors that generate 2D projections from 3D models.

    Features
    --------
    - Automatic plugin registration of subclasses
    - Image caching to avoid recomputation (based on ranges, nbins, z-range, rotation)
    - Standard convenience rotations for XZ and YZ projections

    Subclasses must implement: `_image(...) -> ImageData`.
    """
    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register subclass as a ModelProjector plugin."""
    def __init__(self, cache_len: int = 100) -> None:
        """
        Parameters
        ----------
        cache_len : int, default=100
            Maximum number of inputs to store in the image cache.
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
    def image_xz(self, x_range: tuple[float, float], y_range: tuple[float, float], nbins: int = 100, z_range: tuple[float, float] = (-20, 20)) -> ImageData:
        """Generate a projection in the x-z plane (viewing along +y).

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
        """Generate a projection in the y-z plane (viewing along +x).

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
    """Factory class for accessing registered ModelProjector plugins."""

    @overload
    @classmethod
    def get_plugin(cls, name: Literal["ProjectorLineIntegration"]) -> type[ProjectorLineIntegration]: ...
    @overload
    @classmethod
    def get_plugin(cls, name: Literal["ProjectorSphGrid"]) -> type[ProjectorSphGrid]: ...
    @overload
    @classmethod
    def get_plugin(cls, name: str) -> type[ModelProjectorBase]: ...
