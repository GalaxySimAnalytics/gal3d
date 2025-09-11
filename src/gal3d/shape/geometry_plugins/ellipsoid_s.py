
import numpy as np
from numpy.typing import ArrayLike

from gal3d.config import config
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

class Ellipsoid_S(GeometryBase):
    """
    A shaped ellipsoid geometry class.

    This class implements a generalized ellipsoid with shape parameters that allow
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

    @classmethod
    def derived_param_funcs(cls):
        return {
            "eps_ab": lambda d: 1.0 - d["b"] / d["a"],
            "eps_bc": lambda d: 1.0 - d["c"] / d["b"],
            "eps_ac": lambda d: 1.0 - d["c"] / d["a"],
            "b": lambda d: (
                d["a"] * (1 - d["eps_ab"]) if "eps_ab" in d else d["c"] / (1 - d["eps_bc"])
            ),
            "c": lambda d: (
                d["b"] * (1 - d["eps_bc"]) if "eps_bc" in d else d["a"] * (1 - d["eps_ac"])
            ),
            "a": lambda d: (
                d["b"] / (1 - d["eps_ab"]) if "eps_ab" in d else d["c"] / (1 - d["eps_ac"])
            ),
        }


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
            Shape index along the 'a' axis.
        sb : float
            Shape index along the 'b' axis.
        sc : float
            Shape index along the 'c' axis.
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
            Shape index along the 'a' axis.
        sb : float
            Shape index along the 'b' axis.
        sc : float
            Shape index along the 'c' axis.
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
            Shape index along the 'a' axis.
        sb : float
            Shape index along the 'b' axis.
        sc : float
            Shape index along the 'c' axis.
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
            Shape index along the 'a' axis.
        sb : float
            Shape index along the 'b' axis.
        sc : float
            Shape index along the 'c' axis.
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
