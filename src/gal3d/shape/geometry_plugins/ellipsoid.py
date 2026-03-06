
import logging
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from gal3d.optimization.parameter import Parameters
from gal3d.shape.geometry import GeometryBase

from .ellipsoid_cy import (
    IntersectLinesEllipsoid,
    IntersectRaysEllipsoid,
    area_factor,
    f_ellipsoid,
    f_ellipsoid_jacobian,
    f_ray_ellipsoid,
)

__all__ = ["Ellipsoid"]

logger = logging.getLogger("gal3d.shape.geometry.ellipsoid")

class Ellipsoid(GeometryBase):
    """
    Ellipsoid geometry class with semi-axes a, b, c.

    This class represents an ellipsoid defined by three semi-axes
    a >= b >= c. It implements various geometric functions like
    surface evaluation, ray intersection, etc.
    """

    # Parameter names for the ellipsoid, representing the semi-major axis and ellipticities.
    PN = ("a", "eps_ab", "eps_bc")  ### not use set !!!
    LB = {"a": 0.1, "eps_ab": 0.001, "eps_bc": 0.001}
    UB = {"a": np.inf, "eps_ab": 0.999, "eps_bc": 0.999}

    def __init__(self, *args: Any, **kwargs: float):
        """
        Initialize the ellipsoid with given parameters.

        Parameters
        ----------
        *args : tuple
            Default parameters: a, b, c.
        **kwargs : dict
            Additional parameters to initialize the ellipsoid.

        Notes
        -----
        The parameters a, b, c must satisfy a > b > c.
        The derived parameters eps_ab and eps_bc are defined as:
        eps_ab = 1 - b/a : 0~1
        eps_bc = 1 - c/b : 0~1
        """
        super().__init__(**kwargs)

    @classmethod
    def default_parameters(cls) -> Parameters:
        """
        Return a default set of parameters for the ellipsoid.

        Returns
        -------
        Parameters
            An instance of the Parameters class containing default ellipsoid parameters.
        """
        return cls.create_parameters(a=3.0, eps_ab=0.2, eps_bc=0.5)

    def __call__(self, pos):
        """
        Evaluate the ellipsoid function at given positions.

        Parameters
        ----------
        pos : array_like
            Positions at which to evaluate the ellipsoid function.

        Returns
        -------
        float or ndarray
            The value of the ellipsoid function at the given positions.
        """
        pos = self.to_3d_array(pos)
        return f_ellipsoid(self["a"], self["b"], self["c"], pos)[0]

    def jacobian(self, pos: ArrayLike) -> tuple:
        """
        Compute the Jacobian of the ellipsoid function at given positions.

        Parameters
        ----------
        pos : array_like
            Positions at which to compute the Jacobian.

        Returns
        -------
        tuple
            The Jacobian matrix of the ellipsoid function at the given positions.
        """
        pos = self.to_3d_array(pos)
        return f_ellipsoid_jacobian(self["a"], self["b"], self["c"], pos)

    def ray_intersect(self, pos: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the distance between points and the ray point on the ellipsoid.

        Parameters
        ----------
        pos : array_like
            Positions for which to compute the distance.

        Returns
        -------
        float or ndarray
            The distance between the points and the ray point on the ellipsoid.
        """
        pos = self.to_3d_array(pos)
        return IntersectRaysEllipsoid(self["a"], self["b"], self["c"], pos)

    def line_intersect(self, pos1: ArrayLike, pos2: ArrayLike) -> np.ndarray:

        pos1 = self.to_3d_array(pos1)
        pos2 = self.to_3d_array(pos2)

        return IntersectLinesEllipsoid(self["a"], self["b"], self["c"], pos1, pos2)

    def f_ray_d(self, pos):
        pos = self.to_3d_array(pos)
        return f_ray_ellipsoid(self["a"], self["b"], self["c"], pos)[0]

    def area_factor(self, pos):
        pos = self.to_3d_array(pos)
        return area_factor(self["a"], self["b"], self["c"], pos)

    @classmethod
    def estimate_parameters(cls, pos: ArrayLike) -> dict:
        """
        Estimate the parameters of the ellipsoid from the given positions.
        """
        pos = cls.to_3d_array(pos)

        a = np.percentile(np.abs(pos[:, 0]), 95)
        b = np.percentile(np.abs(pos[:, 1]), 95)
        c = np.percentile(np.abs(pos[:, 2]), 95)

        return {
            "a": a,
            "eps_ab": 1 - b / a,
            "eps_bc": 1 - c / b,
        }

    @staticmethod
    def quick_call(a: float, eps_ab: float, eps_bc: float, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Quickly evaluate the ellipsoid function with given parameters and positions.

        Parameters
        ----------
        a : float
            The semi-major axis of the ellipsoid.
        eps_ab : float
            The ellipticity between the a and b axes.
        eps_bc : float
            The ellipticity between the b and c axes.
        pos : array_like
            Positions at which to evaluate the ellipsoid function.

        Returns
        -------
        float or ndarray
            The value of the ellipsoid function at the given positions.
        """
        b = a * (1.0 - eps_ab)
        c = b * (1.0 - eps_bc)
        return f_ellipsoid(float(a), b, c, pos)

    @staticmethod
    def quick_f_ray_d(a: float, eps_ab: float, eps_bc: float, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Quickly evaluate the distance fraction of the ellipsoid function with given parameters and positions.

        Parameters
        ----------
        a : float
            The semi-major axis of the ellipsoid.
        eps_ab : float
            The ellipticity between the a and b axes.
        eps_bc : float
            The ellipticity between the b and c axes.
        pos : array_like
            Positions at which to evaluate the distance fraction.

        Returns
        -------
        float or ndarray
            The distance fraction of the ellipsoid function at the given positions.
        """
        b = a * (1.0 - eps_ab)
        c = b * (1.0 - eps_bc)
        return f_ray_ellipsoid(float(a), b, c, pos)

    @staticmethod
    def quick_ray_dist(a: float, eps_ab: float, eps_bc: float, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Quickly compute the distance between points and the ray point on the ellipsoid.

        Parameters
        ----------
        a : float
            The semi-major axis of the ellipsoid.
        eps_ab : float
            The ellipticity between the a and b axes.
        eps_bc : float
            The ellipticity between the b and c axes.
        pos : array_like
            Positions for which to compute the distance.

        Returns
        -------
        float or ndarray
            The distance between the points and the ray point on the ellipsoid.
        """
        b = a * (1.0 - eps_ab)
        c = b * (1.0 - eps_bc)
        return IntersectRaysEllipsoid(float(a), b, c, pos)[1:]

    @staticmethod
    def quick_area_factor(a: float, eps_ab: float, eps_bc: float, pos: np.ndarray) -> np.ndarray:
        b = a * (1.0 - eps_ab)
        c = b * (1.0 - eps_bc)
        return area_factor(float(a), b, c, pos)

    @staticmethod
    def quick_line_intersect(a: float, eps_ab: float, eps_bc: float, pos1: np.ndarray, pos2: np.ndarray) -> np.ndarray:
        b = a * (1 - eps_ab)
        c = b * (1 - eps_bc)
        return IntersectLinesEllipsoid(float(a), float(b), float(c), pos1, pos2)

    @staticmethod
    def quick_jacobian(a: float, b: float, c: float, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the Jacobian of the ellipsoid function with given parameters and positions.

        Parameters
        ----------
        a : float
            The semi-major axis of the ellipsoid.
        b : float
            The semi-intermediate axis of the ellipsoid.
        c : float
            The semi-minor axis of the ellipsoid.
        pos : array_like
            Positions at which to compute the Jacobian.

        Returns
        -------
        tuple
            The Jacobian matrix of the ellipsoid function at the given positions.
        """
        return f_ellipsoid_jacobian(float(a), float(b), float(c), pos)

    @property
    def _latex_equation(self) -> str:
        return r"([\frac{x}{a}]^2 + [\frac{y}{b}]^2 + [\frac{z}{c}]^2)"

    @classmethod
    def PNlatex(cls):
        return {
            "a": "a",
            "eps_ab": r"\epsilon_{ab}",
            "eps_bc": r"\epsilon_{bc}",
            "eps_ac": r"\epsilon_{ac}",
        }

    @property
    def _latex_other(self) -> str:
        b = self.parameters["b"]
        c = self.parameters["c"]
        return rf" b = a(1 - \epsilon_{{ab}})={b:.2f},\ c = b(1 - \epsilon_{{bc}})={c:.2f}"


@Ellipsoid.derived
def eps_ab(params):
    return 1.0 - params["b"] / params["a"]

@Ellipsoid.derived
def eps_bc(params):
    return 1.0 - params["c"] / params["b"]

@Ellipsoid.derived
def eps_ac(params):
    return 1.0 - params["c"] / params["a"]


@Ellipsoid.derived
def T(params):
    return (params["a"]**2 - params["b"]**2)/(params["a"]**2 - params["c"]**2)

@Ellipsoid.derived
def b(params):
    if "eps_ab" in params:
        return params["a"] * (1 - params["eps_ab"])
    else:
        return params["c"] / (1 - params["eps_bc"])


@Ellipsoid.derived
def c(params):
    if "eps_bc" in params:
        return params["b"] * (1 - params["eps_bc"])
    else:
        return params["a"] * (1 - params["eps_ac"])


@Ellipsoid.derived
def a(params):
    if "eps_ab" in params:
        return params["b"] / (1 - params["eps_ab"])
    else:
        return params["c"] / (1 - params["eps_ac"])

@Ellipsoid.derived
def eps_ac_err(params):
    eps_ab = params["eps_ab"]
    eps_bc = params["eps_bc"]
    return np.sqrt(
        (1.0 - eps_bc)**2 * eps_ab.err**2 +
        (1.0 - eps_ab)**2 * eps_bc.err**2
    )

@Ellipsoid.derived
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

@Ellipsoid.derived
def eps_ac_s(params):
    return params["eps_ac"]

