import logging

logger = logging.getLogger("gal3d.optimization.util")


def truncate(num: float, n: int) -> float:
    """
    Truncate a float to n decimal places (without rounding).

    Parameters
    ----------
    num : float
        The number to truncate.
    n : int
        Number of decimal places to keep.

    Returns
    -------
    float
        The truncated number. If num is inf, -inf, or nan, returns num unchanged.
    """
    import numpy as np
    if not np.isfinite(num):
        return num
    factor = 10.0 ** n
    return float(int(num * factor) / factor)
