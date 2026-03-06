
from typing import Any

import numpy as np
from numpy.typing import NDArray

ArrayF = NDArray[np.floating[Any]]

def _axis_ratio_error(a: ArrayF, a_prev: ArrayF) -> float:
    return float(
        0.5
        * (
            np.abs(a[1] / a[0] - a_prev[1] / a_prev[0])
            + np.abs(a[2] / a[0] - a_prev[2] / a_prev[0])
        )
    )

def _periodic_diff(a: float, b: float, period: float = 2 * np.pi) -> float:
    """
    Minimal absolute difference between two angles on a circle.

    Parameters
    ----------
    a, b : float
        Angles in radians.
    period : float, optional
        Period of the angles (default is 2π).

    Returns
    -------
    float
        Minimal absolute difference between a and b, accounting for periodicity.
    """
    d = (a - b + 0.5 * period) % period - 0.5 * period
    return float(np.abs(d))


def _prepare_bins(
    r: np.ndarray,
    rmin: float,
    rmax: float,
    nbins: int,
    bins: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare radial bin edges and representative radii.

    Parameters
    ----------
    r : ndarray of float
        Particle radii.
    rmin, rmax : float
        Minimum and maximum radii for binning.
    nbins : int
        Number of radial bins.
    bins : {'equal', 'log', 'lin'}
        Binning scheme:
        * ``'equal'`` – equal number of particles per bin
        * ``'log'``   – logarithmically spaced in radius
        * ``'lin'``   – linearly spaced in radius

    Returns
    -------
    bin_edges : ndarray of float
        Array of length ``nbins + 1`` with bin edges.
    rbins : ndarray of float
        Representative radius for each bin (e.g. geometric mean).
    """

    def equal_bins(r: np.ndarray, N: int) -> np.ndarray:
        sorted_r = np.sort(r[(r >= rmin) & (r <= rmax)])
        step = max(len(sorted_r) // N, 1)
        return np.array(
            [sorted_r[i * step] for i in range(N)] + [sorted_r[-1]],
            dtype=float,
        )

    if bins == "equal":
        full = equal_bins(r, nbins * 2)
        bin_edges = full[0::2]
        rbins     = full[1::2]
    elif bins == "log":
        bin_edges = np.geomspace(rmin, rmax, nbins + 1)
        rbins     = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    elif bins == "lin":
        bin_edges = np.linspace(rmin, rmax, nbins + 1)
        rbins     = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    else:
        raise ValueError(f"Unknown bins scheme: {bins!r}")

    return bin_edges.astype(float), rbins.astype(float)
