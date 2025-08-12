import logging

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .array_operate_cy import *


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
            Input positions of particles.

        Returns
        -------
        numpy.ndarray
            Reshaped position array with shape (n, 3).
        """
        arr = np.array(pos)
        if arr.ndim != 2:
            logger.debug(f"pos is 1d array with shape={arr.shape}, reshaping to (-1,3)")
            arr = arr.reshape(-1, 3)
        if arr.shape[1] == 3:
            return arr
        if arr.shape[0] == 3:
            logger.debug(f"pos have the shape= {arr.shape}, transposing it")
            return arr.T
        logger.debug(
            f"pos have the shape={arr.shape}, target shape: (n,3), reshaping it"
        )
        return arr.reshape(-1, 3)
    