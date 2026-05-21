import numpy as np

try:
    from scipy.special import sph_harm_y as _sph_harm

    _USE_NEW_SPH_HARM = True
except ImportError:
    from scipy.special import sph_harm as _sph_harm  # this is removed in scipy 1.17

    _USE_NEW_SPH_HARM = False


def _complex_spherical_harmonic(
    phi: np.ndarray | float, theta: np.ndarray | float, m: int, l: int
) -> np.ndarray | complex:
    """Calculate the complex spherical harmonic for given angles and indices."""
    if _USE_NEW_SPH_HARM:
        return _sph_harm(l, m, theta, phi)
    return _sph_harm(m, l, phi, theta)


def spherical_harmonics_in_real(
    phi: np.ndarray | float, theta: np.ndarray | float, m: int, l: int
) -> float | np.ndarray:
    """
    Calculate the real part of spherical harmonics for given angles and indices.

    Parameters
    ----------
    phi : float or np.ndarray
        Azimuthal angle (longitude) in radians, ranging from 0 to 2π.
    theta : float or np.ndarray
        Polar angle (colatitude) in radians, ranging from 0 to π.
    m : int
        Order of the spherical harmonic (integer, -l <= m <= l).
    l : int
        Degree of the spherical harmonic (integer, l >= 0).

    Returns
    -------
    float or np.ndarray
        The real part of the spherical harmonic :math:`Y_l^m(\\phi, \theta)`.

    Notes
    -----
    The function computes the real part of the spherical harmonic using the relation:

    - For :math:`m < 0`: :math:`Y_l^m = (-1)^m \\sqrt{2} \\, \\mathrm{Im}(Y_l^{|m|})`
    - For :math:`m > 0`: :math:`Y_l^m = (-1)^m \\sqrt{2} \\, \\mathrm{Re}(Y_l^{|m|})`
    - For :math:`m = 0`: :math:`Y_l^0 = \\mathrm{Re}(Y_l^0)`
    """

    y_lm = _complex_spherical_harmonic(phi, theta, m, l)

    if m < 0:
        return (-1) ** m * np.sqrt(2.0) * np.imag(y_lm)
    if m > 0:
        return (-1) ** m * np.sqrt(2.0) * np.real(y_lm)
    return np.real(y_lm)


def spherical_harmonics_dec(
    theta: np.ndarray, phi: np.ndarray, density: np.ndarray, lmax: int = 4
) -> dict[int, np.ndarray]:
    """
    Perform spherical harmonics decomposition on a given density distribution.

    Parameters
    ----------
    theta : np.ndarray
        Polar angles (colatitude) in radians, ranging from 0 to π.
    phi : np.ndarray
        Azimuthal angles (longitude) in radians, ranging from 0 to 2π.
    density : np.ndarray
        Density distribution on the sphere, corresponding to the given theta and phi.
    lmax : int, optional
        Maximum degree of spherical harmonics to compute (default is 4).

    Returns
    -------
    dict
        A dictionary where the keys are the degrees l (from 0 to lmax), and the values
        are arrays of spherical harmonics coefficients for each order m (from -l to l).

    Notes
    -----
    The function computes the spherical harmonics coefficients by integrating the density
    distribution multiplied by the spherical harmonics over the sphere. The integration
    is approximated by a sum over the provided grid of theta and phi values.
    """

    coef: dict[int, np.ndarray] = {}
    for l in range(lmax + 1):
        temp = []
        for m in range(l, -l - 1, -1):
            temp.append(np.sum(density * spherical_harmonics_in_real(phi, theta, m, l) * np.sin(theta)))
        coef[l] = np.array(temp)
    return coef
