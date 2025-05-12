import numpy as np
from numpy.typing import ArrayLike

from ._ellipsoid_s_util import *
from ..geometry import GeometryBase, classproperty, Parameters

__all__ = ['Ellipsoid_S']


class Ellipsoid_S(GeometryBase):

    # Using a tuple instead of a set to maintain order and allow duplicates if needed.
    PN = ('a', 'eps_ab', 'eps_bc', 'sa', 'sb', 'sc')
    LB = {'a': 0.1, 'eps_ab': 0.01, 'eps_bc': 0.01, 'sa': 0.2, 'sb': 0.2, 'sc': 0.2}
    UB = {'a': np.inf, 'eps_ab': 0.99, 'eps_bc': 0.99, 'sa': 2, 'sb': 2, 'sc': 2}

    MaxIterationDist = 100
    MaxIterationLine = 100
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
        self.parameters = self.init_parameters(**kwargs)
        self.parameters = self.init_parameters(**kwargs)

    @staticmethod
    def init_parameters(**kwargs):
        """
        Initializes and returns the parameters with derived values.

        Parameters
        ----------
        **kwargs : dict
            A dictionary of parameters to initialize the ellipsoid.

        Returns
        -------
        Parameters
            An instance of Parameters with initialized and derived values.
        """
        param = Parameters(**kwargs)
        param._derived['eps_ab'] = lambda d: 1.0 - d['b'] / d['a']
        param._derived['eps_bc'] = lambda d: 1.0 - d['c'] / d['b']
        param._derived['eps_ac'] = lambda d: 1.0 - d['c'] / d['a']
        param._derived['b'] = lambda d: (
            d['a'] * (1 - d['eps_ab']) if 'eps_ab' in d else d['c'] / (1 - d['eps_bc'])
        )
        param._derived['c'] = lambda d: (
            d['b'] * (1 - d['eps_bc']) if 'eps_bc' in d else d['a'] * (1 - d['eps_ac'])
        )
        param._derived['a'] = lambda d: (
            d['b'] / (1 - d['eps_ab']) if 'eps_ab' in d else d['c'] / (1 - d['eps_ac'])
        )

        parameters = Parameters(**{i: param[i] for i in Ellipsoid_S.PN})
        parameters._derived.update(param._derived)
        parameters.set_lb(**Ellipsoid_S.LB)
        parameters.set_ub(**Ellipsoid_S.UB)
        return parameters

    @staticmethod
    def get_parameters():
        """
        Returns a default set of parameters for the ellipsoid.

        Returns
        -------
        Parameters
            An instance of Parameters with default values.
        """
        return Ellipsoid_S.init_parameters(
            a=3.0, eps_ab=0.2, eps_bc=0.5, sa=1.0, sb=1.0, sc=1.0
        )

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
        if (len(np.shape(pos)) == 2) and (np.shape(pos)[1] == 3):
            pos = np.float64(pos)
        if len(np.shape(pos)) == 1:
            pos = np.float64([pos])
        return f_shaped_ellipsoid(
            self['a'], self['b'], self['c'], self['sa'], self['sb'], self['sc'], pos
        )

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
        if (len(np.shape(pos)) == 2) and (np.shape(pos)[1] == 3):
            pos = np.float64(pos)
        if len(np.shape(pos)) == 1:
            pos = np.float64([pos])
        return f_shaped_ellipsoid_jacobian(
            self['a'], self['b'], self['c'], self['sa'], self['sb'], self['sc'], pos
        )

    def ray_intersect(self, pos: ArrayLike) -> tuple:
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
        if (len(np.shape(pos)) == 2) and (np.shape(pos)[1] == 3):
            pos = np.float64(pos)
        if len(np.shape(pos)) == 1:
            pos = np.float64([pos])
        return IntersectRaysEllipsoid_S(
            self['a'],
            self['b'],
            self['c'],
            self['sa'],
            self['sb'],
            self['sc'],
            pos,
            Ellipsoid_S.MaxIterationDist,
        )

    def line_intersect(self, pos1: ArrayLike, pos2: ArrayLike) -> np.ndarray:

        if (len(np.shape(pos1)) == 2) and (np.shape(pos1)[1] == 3):
            pos1 = np.float64(pos1)
        if len(np.shape(pos1)) == 1:
            pos1 = np.float64([pos1])
        if (len(np.shape(pos2)) == 2) and (np.shape(pos2)[1] == 3):
            pos2 = np.float64(pos2)
        if len(np.shape(pos2)) == 1:
            pos2 = np.float64([pos2])

        return IntersectLinesEllipsoid_S(
            self['a'],
            self['b'],
            self['c'],
            self['sa'],
            self['sb'],
            self['sc'],
            pos1,
            pos2,
            Ellipsoid_S.MaxIterationLine,
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
        if (len(np.shape(pos)) == 2) and (np.shape(pos)[1] == 3):
            pos = np.float64(pos)
        if len(np.shape(pos)) == 1:
            pos = np.float64([pos])
        return f_ray_shaped_ellipsoid(
            self['a'],
            self['b'],
            self['c'],
            self['sa'],
            self['sb'],
            self['sc'],
            pos,
            Ellipsoid_S.MaxIterationDist,
        )

    @staticmethod
    def quick_call(a, eps_ab, eps_bc, sa, sb, sc, pos) -> np.ndarray:
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
            The eccentricity between the 'a' and 'b' axes.
        eps_bc : float
            The eccentricity between the 'b' and 'c' axes.
        sa : float
            Scaling factor along the 'a' axis.
        sb : float
            Scaling factor along the 'b' axis.
        sc : float
            Scaling factor along the 'c' axis.
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
    def quick_f_ray_d(a, eps_ab, eps_bc, sa, sb, sc, pos) -> np.ndarray:
        """Quickly evaluates the distance fraction of the geometry function with given parameters and positions, useful in error function"""

        b = a * (1 - eps_ab)
        c = b * (1 - eps_bc)
        return f_ray_shaped_ellipsoid(
            a, b, c, sa, sb, sc, pos, Ellipsoid_S.MaxIterationDist
        )

    @staticmethod
    def quick_ray_dist(a, eps_ab, eps_bc, sa, sb, sc, pos) -> np.ndarray:
        """
        Quickly computes the distance between points and ray points on the ellipsoid.

        This method calculates the distance fraction of the geometry function using the provided parameters
        and positions, without relying on the instance's internal state.

        Parameters
        ----------
        a : float
            The semi-major axis of the ellipsoid.
        eps_ab : float
            The eccentricity between the 'a' and 'b' axes.
        eps_bc : float
            The eccentricity between the 'b' and 'c' axes.
        sa : float
            Scaling factor along the 'a' axis.
        sb : float
            Scaling factor along the 'b' axis.
        sc : float
            Scaling factor along the 'c' axis.
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
            float(a), b, c, sa, sb, sc, pos, Ellipsoid_S.MaxIterationDist
        )[1]

    @staticmethod
    def quick_line_intersect(a, eps_ab, eps_bc, sa, sb, sc, pos1, pos2) -> np.ndarray:
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
            Ellipsoid_S.MaxIterationLine,
        )

    @staticmethod
    def quick_jacobian(a, b, c, sa, sb, sc, pos) -> tuple:
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
            Scaling factor along the 'a' axis.
        sb : float
            Scaling factor along the 'b' axis.
        sc : float
            Scaling factor along the 'c' axis.
        pos : array_like
            An array of positions where the Jacobian is computed.

        Returns
        -------
        tuple
            The computed Jacobian values at the given positions.
        """
        if (len(np.shape(pos)) == 2) and (np.shape(pos)[1] == 3):
            pos = np.float64(pos)
        if len(np.shape(pos)) == 1:
            pos = np.float64([pos])
        return f_shaped_ellipsoid_jacobian(
            float(a), float(b), float(c), float(sa), float(sb), float(sc), pos
        )
