
import logging


import numpy as np

from gal3d import config
from ..geometry import GeometryBase, Parameters

if config['general']['use_cython']:
    from .ellipsoid_cy import f_ellipsoid,f_ellipsoid_jacobian,IntersectRaysEllipsoid,IntersectLinesEllipsoid,f_ray_ellipsoid
else:
    from .ellipsoid_nb import f_ellipsoid,f_ellipsoid_jacobian,IntersectRaysEllipsoid,IntersectLinesEllipsoid,f_ray_ellipsoid

__all__ = ['Ellipsoid']

logger = logging.getLogger("gal3d.shape.geometry.ellipsoid")

class Ellipsoid(GeometryBase):
    """
    Ellipsoid geometry class with semi-axes a, b, c.
    
    This class represents an ellipsoid defined by three semi-axes
    a >= b >= c. It implements various geometric functions like
    surface evaluation, ray intersection, etc.
    """
    
    # Parameter names for the ellipsoid, representing the semi-major axis and ellipticities.
    PN = ('a', 'eps_ab', 'eps_bc')  ### not use set !!!
    LB = {'a': 0.1, 'eps_ab': 0.01, 'eps_bc': 0.01}
    UB = {'a': np.inf, 'eps_ab': 0.99, 'eps_bc': 0.99}

    MaxIterationClosed = 100

    def __init__(self, *args, **kwargs):
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
        self.parameters = self.init_parameters(**kwargs)

    @staticmethod
    def init_parameters(**kwargs):
        """
        Initialize and return the parameters of the ellipsoid.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to initialize the ellipsoid.

        Returns
        -------
        Parameters
            An instance of the Parameters class containing the ellipsoid parameters.
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

        parameters = Parameters(**{i: param[i] for i in Ellipsoid.PN})
        parameters._derived.update(param._derived)
        parameters.set_lb(**Ellipsoid.LB)
        parameters.set_ub(**Ellipsoid.UB)
        return parameters

    @staticmethod
    def get_parameters():
        """
        Return a default set of parameters for the ellipsoid.

        Returns
        -------
        Parameters
            An instance of the Parameters class containing default ellipsoid parameters.
        """
        return Ellipsoid.init_parameters(a=3.0, eps_ab=0.2, eps_bc=0.5)

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
        pos = self._check_pos(pos)
        return f_ellipsoid(self['a'], self['b'], self['c'], pos)

    def jacobian(self, pos) -> tuple:
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
        pos = self._check_pos(pos)
        return f_ellipsoid_jacobian(self['a'], self['b'], self['c'], pos)

    def ray_intersect(self, pos):
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
        pos = self._check_pos(pos)
        return IntersectRaysEllipsoid(self['a'], self['b'], self['c'], pos)

    def line_intersect(self, pos1, pos2):

        pos1 = self._check_pos(pos1)
        pos2 = self._check_pos(pos2)

        return IntersectLinesEllipsoid(self['a'], self['b'], self['c'], pos1, pos2)

    def f_ray_d(self, pos):
        pos = self._check_pos(pos)
        return f_ray_ellipsoid(self['a'], self['b'], self['c'], pos)

    @staticmethod
    def quick_call(a, eps_ab, eps_bc, pos):
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
    def quick_f_ray_d(a, eps_ab, eps_bc, pos):
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
    def quick_ray_dist(a, eps_ab, eps_bc, pos):
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
        return IntersectRaysEllipsoid(float(a), b, c, pos)[1]

    @staticmethod
    def quick_line_intersect(a, eps_ab, eps_bc, pos1, pos2):
        b = a * (1 - eps_ab)
        c = b * (1 - eps_bc)
        return IntersectLinesEllipsoid(float(a), float(b), float(c), pos1, pos2)

    @staticmethod
    def quick_jacobian(a, b, c, pos):
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

    @staticmethod
    def _check_pos(pos):
        """
        Ensure pos is a 2D numpy array of shape (N, 3).
        """
        pos = np.asarray(pos, dtype=np.float64)
        if pos.ndim == 1:
            pos = pos[np.newaxis, :]
        if pos.shape[1] != 3:
            raise ValueError("Input position must have shape (N, 3)")
        return pos