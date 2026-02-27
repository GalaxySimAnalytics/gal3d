import logging

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .array_operate_cy import (
    Dot,
    Hadamard,
    Matmul,
    RobustLength2d,
    Rotate,
    RotateAndShift,
    Shift,
    trans_to_Cartesian_coordinates,
    trans_to_Spherical_coordinates,
    unit_vector3d,
    vector_length3d,
)

__all__ = [
    "Shift", "Rotate", "Matmul", "Hadamard", "Dot",
    "vector_length3d", "unit_vector3d",
    "trans_to_Cartesian_coordinates", "trans_to_Spherical_coordinates",
    "RobustLength2d", "RotateAndShift",
    "Auto3DShape"
]

logger = logging.getLogger("gal3d.util.array_operate")
class Auto3DShape:
    """
    Class for checking, validating, and automatically converting 3D position arrays to shape (n, 3).
    """

    @staticmethod
    def to_3d_array(pos: ArrayLike) -> NDArray:
        """
        Ensure that the position array has shape (n, 3).

        Parameters
        ----------
        pos : array_like
            Input positions. Can be in shape (3,), (3, n), or (n, 3).

        Returns
        -------
        numpy.ndarray
            Reshaped position array with shape (n, 3).
        """
        arr = np.array(pos)
        if arr.ndim != 2:
            logger.debug("pos is 1d array with shape= %s, reshaping to (-1,3)",
                         repr(arr.shape))
            arr = arr.reshape(-1, 3)
        if arr.shape[1] == 3:
            return arr
        if arr.shape[0] == 3:
            logger.debug("pos have the shape= %s, transposing it",
                         repr(arr.shape))
            return arr.T
        logger.debug(
            "pos have the shape= %s, target shape: (n,3), reshaping it",
            repr(arr.shape)
        )
        return arr.reshape(-1, 3)
