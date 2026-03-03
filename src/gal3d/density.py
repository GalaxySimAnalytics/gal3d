import numpy as np
from numpy.typing import ArrayLike

from gal3d.util.array_operate import Auto3DShape


class DensitySource(Auto3DShape):
    """
    Base class for density estimation, providing a common interface for different density estimation methods.
    """

    def __call__(self, pos: ArrayLike) -> np.ndarray:
        """
        Estimate the density at the given positions.

        Parameters
        ----------
        pos : array_like, shape (m, 3)
            Target positions where the density is to be estimated.

        Returns
        -------
        numpy.ndarray
            Estimated density values at the target positions.
        """
        pos3d = self.to_3d_array(pos)
        is_1d = np.array(pos).ndim == 1
        density = self._evaluate_density(pos3d)
        if is_1d:
            return density[0]
        return density

    def _evaluate_density(self, pos: np.ndarray) -> np.ndarray:
        """
        Internal method to evaluate the density at the given 3D positions.

        Parameters
        ----------
        pos : ndarray, shape (m, 3)
            Target positions where the density is to be estimated.

        Returns
        -------
        numpy.ndarray
            Estimated density values at the target positions.
        """
        raise NotImplementedError("Subclasses must implement this method.")
