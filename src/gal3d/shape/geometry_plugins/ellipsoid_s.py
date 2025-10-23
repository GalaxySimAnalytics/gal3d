import logging
from functools import cache

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator

from gal3d.configuration import config
from gal3d.shape.geometry import GeometryBase

from .ellipsoid_s_cy import (
    IntersectLinesEllipsoid_S,
    IntersectRaysEllipsoid_S,
    area_factor,
    f_ray_shaped_ellipsoid,
    f_shaped_ellipsoid,
    f_shaped_ellipsoid_jacobian,
)

__all__ = ["Ellipsoid_S"]

logger = logging.getLogger("gal3d.shape.Ellipsoid_S")

class Ellipsoid_S(GeometryBase):
    """
    A shaped ellipsoid geometry class.

    This class implements a generalized ellipsoid (superellipsoid) with shape parameters that allow
    for more flexible modeling of 3D geometric shapes.
    """

    # Using a tuple instead of a set to maintain order and allow duplicates if needed.
    PN = ("a", "eps_ab", "eps_bc", "sa", "sb", "sc")
    LB = {"a": 0.1, "eps_ab": 0.01, "eps_bc": 0.01, "sa": 0.2, "sb": 0.2, "sc": 0.2}
    UB = {"a": np.inf, "eps_ab": 0.99, "eps_bc": 0.99, "sa": 2, "sb": 2, "sc": 2}

    def __init__(self, *args, **kwargs):
        """
        Initializes the Ellipsoid_S instance with given parameters.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the initializer (currently unused).
        **kwargs : dict
            A dictionary of parameters to initialize the ellipsoid.
        """
        super().__init__(**kwargs)

    @classmethod
    def default_parameters(cls):
        """
        Return a default set of parameters for the shaped ellipsoid.

        Returns
        -------
        Parameters
            An instance of the Parameters class containing default shaped ellipsoid parameters.
        """
        return cls.create_parameters(a=3.0, eps_ab=0.2, eps_bc=0.5, sa=1.0, sb=1.0, sc=1.0)


    def __call__(self, pos):
        """
        Evaluates the ellipsoid function at the given positions.

        Parameters
        ----------
        pos : array_like
            An array of positions where the ellipsoid function is evaluated.

        Returns
        -------
        array_like
            The evaluated values of the ellipsoid function at the given positions.
        """
        pos = self.to_3d_array(pos)
        return f_shaped_ellipsoid(
            self["a"], self["b"], self["c"], self["sa"], self["sb"], self["sc"], pos
        )[0]

    def jacobian(self, pos: ArrayLike) -> tuple:
        """
        Computes the Jacobian of the ellipsoid function at the given positions.

        Parameters
        ----------
        pos : array_like
            An array of positions where the Jacobian is computed.

        Returns
        -------
        tuple
            The computed Jacobian values at the given positions.
        """
        pos = self.to_3d_array(pos)
        return f_shaped_ellipsoid_jacobian(
            self["a"], self["b"], self["c"], self["sa"], self["sb"], self["sc"], pos
        )

    def ray_intersect(self, pos: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the distance between points and ray points on the ellipsoid.

        Parameters
        ----------
        pos : array_like
            An array of positions where the distance is computed.

        Returns
        -------
        array_like
            The computed distances between points and ray points on the ellipsoid.
        """
        pos = self.to_3d_array(pos)
        return IntersectRaysEllipsoid_S(
            self["a"],
            self["b"],
            self["c"],
            self["sa"],
            self["sb"],
            self["sc"],
            pos,
            config.ellipsoid_s.MaxIterationDist,
        )

    def line_intersect(self, pos1: ArrayLike, pos2: ArrayLike) -> np.ndarray:

        pos1 = self.to_3d_array(pos1)
        pos2 = self.to_3d_array(pos2)

        return IntersectLinesEllipsoid_S(
            self["a"],
            self["b"],
            self["c"],
            self["sa"],
            self["sb"],
            self["sc"],
            pos1,
            pos2,
            config.ellipsoid_s.MaxIterationLine,
        )

    def f_ray_d(self, pos: ArrayLike) -> np.ndarray:
        """
        pos : array_like
            An array of positions where the derivative is evaluated.

        Returns
        -------
        array_like
            The evaluated distance values at the given positions.
        """
        pos = self.to_3d_array(pos)
        return f_ray_shaped_ellipsoid(
            self["a"],
            self["b"],
            self["c"],
            self["sa"],
            self["sb"],
            self["sc"],
            pos,
            config.ellipsoid_s.MaxIterationDist,
        )[0]

    def area_factor(self, pos: ArrayLike) -> np.ndarray:
        """
        Computes the area factor at the given positions.

        Parameters
        ----------
        pos : array_like
            An array of positions where the area factor is computed.

        Returns
        -------
        array_like
            The computed area factors at the given positions.
        """
        pos = self.to_3d_array(pos)
        return area_factor(
            self["a"],
            self["b"],
            self["c"],
            self["sa"],
            self["sb"],
            self["sc"],
            pos
        )

    @classmethod
    def estimate_parameters(cls, pos: ArrayLike) -> dict:

        pos = cls.to_3d_array(pos)

        a = np.percentile(np.abs(pos[:, 0]), 95)
        b = np.percentile(np.abs(pos[:, 1]), 95)
        c = np.percentile(np.abs(pos[:, 2]), 95)

        return {
            "a": a,
            "eps_ab": 1 - b / a,
            "eps_bc": 1 - c / b,
            "sa": 1.,
            "sb": 1.,
            "sc": 1.
        }

    @staticmethod
    def quick_call(a: float, eps_ab: float, eps_bc: float, sa: float, sb: float, sc: float, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Quickly evaluates the ellipsoid function with the given parameters and positions.

        This method is a simplified and faster version of the `__call__` method, designed for use in scenarios
        where performance is critical, such as optimization or error function evaluations. It computes the
        ellipsoid function values directly using the provided parameters without relying on the instance's
        internal state.

        Parameters
        ----------
        a : float
            The semi-major axis of the ellipsoid.
        eps_ab : float
            The ellipticity between the 'a' and 'b' axes.
        eps_bc : float
            The ellipticity between the 'b' and 'c' axes.
        sa : float
            Shape exponent along the 'a' axis.
        sb : float
            Shape exponent along the 'b' axis.
        sc : float
            Shape exponent along the 'c' axis.
        pos : array_like
            An array of positions where the ellipsoid function is evaluated.

        Returns
        -------
        np.ndarray
            The evaluated values of the ellipsoid function at the given positions.
        """
        b = a * (1 - eps_ab)
        c = b * (1 - eps_bc)
        return f_shaped_ellipsoid(a, b, c, sa, sb, sc, pos)

    @staticmethod
    def quick_area_factor(a: float, eps_ab: float, eps_bc: float, sa: float, sb: float, sc: float, pos: np.ndarray) -> np.ndarray:
        """
        Quickly evaluates the area factor with the given parameters and positions.

        This method is a simplified and faster version of the `area_factor` method, designed for use in scenarios
        where performance is critical, such as optimization or error function evaluations. It computes the
        area factor values directly using the provided parameters without relying on the instance's
        internal state.

        Parameters
        ----------
        a : float
            The semi-major axis of the ellipsoid.
        eps_ab : float
            The ellipticity between the 'a' and 'b' axes.
        eps_bc : float
            The ellipticity between the 'b' and 'c' axes.
        sa : float
            Shape exponent along the 'a' axis.
        sb : float
            Shape exponent along the 'b' axis.
        sc : float
            Shape exponent along the 'c' axis.
        pos : array_like
            An array of positions where the area factor is evaluated.

        Returns
        -------
        np.ndarray
            The evaluated values of the area factor at the given positions.
        """
        b = a * (1 - eps_ab)
        c = b * (1 - eps_bc)
        return area_factor(a, b, c, sa, sb, sc, pos)

    @staticmethod
    def quick_f_ray_d(a: float, eps_ab: float, eps_bc: float, sa: float, sb: float, sc: float, pos: np.ndarray,) -> tuple[np.ndarray, np.ndarray]:
        """Quickly evaluates the distance fraction of the geometry function with given parameters and positions, useful in error function"""

        b = a * (1 - eps_ab)
        c = b * (1 - eps_bc)
        return f_ray_shaped_ellipsoid(
            a, b, c, sa, sb, sc, pos, config.ellipsoid_s.MaxIterationDist
        )

    @staticmethod
    def quick_ray_dist(a: float, eps_ab: float, eps_bc: float, sa: float, sb: float, sc: float, pos: np.ndarray,) -> tuple[np.ndarray, np.ndarray]:
        """
        Quickly computes the distance between points and ray points on the ellipsoid.

        This method calculates the distance fraction of the geometry function using the provided parameters
        and positions, without relying on the instance's internal state.

        Parameters
        ----------
        a : float
            The semi-major axis of the ellipsoid.
        eps_ab : float
            The ellipticity between the 'a' and 'b' axes.
        eps_bc : float
            The ellipticity between the 'b' and 'c' axes.
        sa : float
            Shape exponent along the 'a' axis.
        sb : float
            Shape exponent along the 'b' axis.
        sc : float
            Shape exponent along the 'c' axis.
        pos : array_like
            An array of positions where the distance is computed.

        Returns
        -------
        np.ndarray
            The computed distances between points and ray points on the ellipsoid.
        """

        b = a * (1 - eps_ab)
        c = b * (1 - eps_bc)
        return IntersectRaysEllipsoid_S(
            float(a), b, c, sa, sb, sc, pos, config.ellipsoid_s.MaxIterationDist
        )[1:]

    @staticmethod
    def quick_line_intersect(a: float, eps_ab: float, eps_bc: float, sa: float, sb: float, sc: float, pos1: np.ndarray, pos2: np.ndarray) -> np.ndarray:
        b = a * (1 - eps_ab)
        c = b * (1 - eps_bc)
        return IntersectLinesEllipsoid_S(
            float(a),
            float(b),
            float(c),
            float(sa),
            float(sb),
            float(sc),
            pos1,
            pos2,
            config.ellipsoid_s.MaxIterationLine,
        )

    @staticmethod
    def quick_jacobian(a: float, b: float, c: float, sa: float, sb: float, sc: float, pos: np.ndarray) -> tuple:
        """
        Quickly computes the Jacobian of the ellipsoid function at the given positions.

        This method calculates the Jacobian using the provided parameters and positions
        without relying on the instance's internal state.

        Parameters
        ----------
        a : float
            The semi-major axis of the ellipsoid.
        b : float
            The semi-minor axis of the ellipsoid along the 'b' axis.
        c : float
            The semi-minor axis of the ellipsoid along the 'c' axis.
        sa : float
            Shape exponent along the 'a' axis.
        sb : float
            Shape exponent along the 'b' axis.
        sc : float
            Shape exponent along the 'c' axis.
        pos : array_like
            An array of positions where the Jacobian is computed.

        Returns
        -------
        tuple
            The computed Jacobian values at the given positions.
        """
        return f_shaped_ellipsoid_jacobian(
            float(a), float(b), float(c), float(sa), float(sb), float(sc), pos
        )


@Ellipsoid_S.derived
def eps_ab(params):
    return 1.0 - params["b"] / params["a"]

@Ellipsoid_S.derived
def eps_bc(params):
    return 1.0 - params["c"] / params["b"]

@Ellipsoid_S.derived
def eps_ac(params):
    return 1.0 - params["c"] / params["a"]


@Ellipsoid_S.derived
def T(params):
    return (params["a"]**2 - params["b"]**2)/(params["a"]**2 - params["c"]**2)

@Ellipsoid_S.derived
def b(params):
    if "eps_ab" in params:
        return params["a"] * (1 - params["eps_ab"])
    else:
        return params["c"] / (1 - params["eps_bc"])


@Ellipsoid_S.derived
def c(params):
    if "eps_bc" in params:
        return params["b"] * (1 - params["eps_bc"])
    else:
        return params["a"] * (1 - params["eps_ac"])


@Ellipsoid_S.derived
def a(params):
    if "eps_ab" in params:
        return params["b"] / (1 - params["eps_ab"])
    else:
        return params["c"] / (1 - params["eps_ac"])

@Ellipsoid_S.derived
def eps_ac_err(params):
    eps_ab = params["eps_ab"]
    eps_bc = params["eps_bc"]
    return np.sqrt(
        (1.0 - eps_bc)**2 * eps_ab.err**2 +
        (1.0 - eps_ab)**2 * eps_bc.err**2
    )

@Ellipsoid_S.derived
def T_err(params):
    eps_ab = params["eps_ab"]
    eps_bc = params["eps_bc"]
    p = 1 - eps_ab
    r = 1 - eps_bc
    q = p * r
    N = 1- p*p
    D = 1 - q * q

    dT_deab = 2.0*p/D - (2.0*N*p*r**2)/(D**2)
    dT_debc = - (2.0*N*p**2*r)/(D**2)

    return np.sqrt((dT_deab**2)*(eps_ab.err**2) + (dT_debc**2)*(eps_bc.err**2))

# Constants for normalization:
# For a sphere (sa = sc = 1), I_sphere = ∫0^1 2x sqrt(1 - x^2) dx = 2/3
_SPHERE_I = 2.0 / 3.0
_SPHERE_CENTER = 1.0 - _SPHERE_I        # = 1/3, value when eps_ac = 0 at sphere
_SPHERE_SLOPE = 1.0 - _SPHERE_CENTER    # = 2/3, delta from eps_ac=0 to eps_ac=1
_SPHERE_SCALE = 1.0 / _SPHERE_SLOPE     # = 3/2, used to recover eps_ac at sphere

def _build_I_table(n_samples: int | None = None) -> tuple[np.ndarray,np.ndarray,np.ndarray,RegularGridInterpolator]:
    """
    Build a lookup table for I(sa, sc) over a regular grid and return
    (sa_grid, sc_grid, I_table, interpolator).

    Parameters
    ----------
    n_samples : int | None
        Number of samples along each axis. If None, read from environment variable

    Notes
    -----
    - Grid resolution n is read from config.ellipsoid_s.EpsTableN (default 81).
    - Integration uses scipy.integrate.quad with epsabs=1e-5.
    """
    logger.info(
        "Building eps_ac_s table"
    )

    n = int(config.ellipsoid_s.EpsTableN) if n_samples is None else int(n_samples)
    sa_min = Ellipsoid_S.LB["sa"]
    sa_max =  Ellipsoid_S.UB["sa"]
    sc_min = Ellipsoid_S.LB["sc"]
    sc_max = Ellipsoid_S.UB["sc"]

    sa_grid = np.linspace(sa_min, sa_max, n)
    sc_grid = np.linspace(sc_min, sc_max, n)

    integr_epsabs = 1e-5

    logger.debug(
        "Building eps_ac_s lookup table: n=%d, sa=[%.1f, %.1f], sc=[%.1f, %.1f], epsabs=%.1e",
        n, sa_min, sa_max, sc_min, sc_max, integr_epsabs
    )

    def integrand(x, sa, sc):
        return 2.0 * x * np.sqrt((1.0 - (x * x) ** sa) ** (1.0 / sc))

    I = np.empty((sa_grid.size, sc_grid.size), dtype=float)
    for i, sa in enumerate(sa_grid):
        for j, sc in enumerate(sc_grid):
            I[i, j], _ = quad(integrand, 0.0, 1.0, args=(sa, sc), epsabs=integr_epsabs)

    interp = RegularGridInterpolator(
        (sa_grid, sc_grid), I, bounds_error=False, fill_value=None
    )
    return sa_grid, sc_grid, I, interp

@cache
def _I_interp_for_n(n: int) -> RegularGridInterpolator:
    return _build_I_table(n)[3]

def _get_I_interp() -> RegularGridInterpolator:
    """
    Lazily get the I(sa, sc) interpolator cached per table size (EpsTableN).
    """
    return _I_interp_for_n(int(config.ellipsoid_s.EpsTableN))

def reset_I_interp_cache() -> None:
    """
    Clear cached interpolators. Call after changing config.ellipsoid_s.EpsTableN.
    """
    _I_interp_for_n.cache_clear()
    logger.info("Cleared eps_ac_s interpolator cache")

@Ellipsoid_S.derived
def eps_ac_s(params):
    """
    Shape-adjusted ellipticity between 'a' and 'c'.

    Formula:
      I(sa, sc) = ∫_0^1 2x * sqrt((1 - (x^2)^sa)^(1/sc)) dx
      val_raw   = 1 - (1 - eps_ac) * I(sa, sc)

    Normalize so that for a sphere (sa=sc=1, I=2/3) we have eps_ac_s == eps_ac:
      center = 1 - I_sphere = 1/3
      slope  = 2/3
      eps_ac_s = clip((val_raw - center) / slope, 0, 1)

    Notes
    -----
    - The I(sa, sc) table/interpolator is cached and built with resolution
      n = config.ellipsoid_s.EpsTableN.
    """
    sa = float(params["sa"])
    sc = float(params["sc"])
    eps_ac = float(params["eps_ac"])

    interp = _get_I_interp()
    I = float(interp([[sa, sc]])[0])

    # Basic sanity clamp for numerical stability if extrapolation occurs.
    I = float(np.clip(I, 0.0, 1.0))

    val_raw = 1.0 - (1.0 - eps_ac) * I
    val = _SPHERE_SCALE * (val_raw - _SPHERE_CENTER)
    return float(np.clip(val, 0.0, 1.0))
