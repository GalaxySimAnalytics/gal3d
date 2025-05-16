# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from cython.parallel import prange
from numpy.typing import ArrayLike
cimport cython
cimport openmp
import logging

from ..geometry import GeometryBase, Parameters
from gal3d import config

# Initialize numpy
np.import_array()

# Configure logger
logger = logging.getLogger("gal3d.shape.geometry.ellipsoid")

# Get config for threads

#openmp.omp_set_num_threads(num_threads)

# -------------------- Utility functions (merged from _ellipsoid_util_cy.pyx) --------------------

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=1] vector_length3d(np.ndarray[np.float64_t, ndim=2] pos):
    """Calculate the length of 3D vectors."""
    cdef int n = pos.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros(n, dtype=np.float64)
    cdef double x, y, z
    cdef int i
    cdef int num_threads = config['general']['number_of_threads']
    for i in prange(n, nogil=True, schedule='static', num_threads=num_threads):
        x = pos[i, 0]
        y = pos[i, 1]
        z = pos[i, 2]
        result[i] = sqrt(x*x + y*y + z*z)
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[np.float64_t, ndim=2] unit_vector3d(np.ndarray[np.float64_t, ndim=2] pos):
    """Normalize 3D vectors to unit length."""
    cdef int n = pos.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] result = np.zeros((n, 3), dtype=np.float64)
    cdef double length
    cdef double x, y, z
    cdef int i
    cdef int num_threads = config['general']['number_of_threads']
    for i in prange(n, nogil=True, schedule='static', num_threads=num_threads):
        x = pos[i, 0]
        y = pos[i, 1]
        z = pos[i, 2]
        length = sqrt(x*x + y*y + z*z)
        if length != 0:
            result[i, 0] = x / length
            result[i, 1] = y / length
            result[i, 2] = z / length
        else:
            result[i, 0] = 0.0
            result[i, 1] = 0.0
            result[i, 2] = 0.0
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t, ndim=1] f_ellipsoid(double a, double b, double c, 
                                                np.ndarray[np.float64_t, ndim=2] pos):
    """Evaluate the ellipsoid function at the given positions."""
    cdef int n = pos.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros(n, dtype=np.float64)
    cdef double x, y, z
    cdef int i
    cdef int num_threads = config['general']['number_of_threads']
    for i in prange(n, nogil=True, schedule='static', num_threads=num_threads):
        x = pos[i, 0]
        y = pos[i, 1]
        z = pos[i, 2]
        result[i] = x*x/(a*a) + y*y/(b*b) + z*z/(c*c)
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef tuple f_ellipsoid_jacobian(double a, double b, double c, 
                              np.ndarray[np.float64_t, ndim=2] pos):
    """
    Calculate the Jacobian of the ellipsoid function.
    Returns (da, db, dc, dx, dy, dz)
    """
    cdef int n = pos.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] dx = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] dy = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] dz = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] da = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] db = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] dc = np.zeros(n, dtype=np.float64)
    cdef double x, y, z
    cdef int i
    cdef int num_threads = config['general']['number_of_threads']
    for i in prange(n, nogil=True, schedule='static', num_threads=num_threads):
        x = pos[i, 0]
        y = pos[i, 1]
        z = pos[i, 2]

        dx[i] = 2 * x / (a * a)
        dy[i] = 2 * y / (b * b)
        dz[i] = 2 * z / (c * c)
        da[i] = -dx[i] * x / a
        db[i] = -dy[i] * y / b
        dc[i] = -dz[i] * z / c
    return (da, db, dc, dx, dy, dz)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef tuple IntersectRaysEllipsoid(double a, double b, double c, 
                                np.ndarray[np.float64_t, ndim=2] pos):
    """
    Compute the intersection points of rays with the ellipsoid.
    Returns (tarpos, L - d) - intersection points and distances
    """
    cdef int n = pos.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] tarpos = np.zeros((n, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] d = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] L = np.zeros(n, dtype=np.float64)
    cdef double x, y, z, xi, yi, zi, Li, di, denom
    cdef int i
    cdef int num_threads = config['general']['number_of_threads']
    for i in prange(n, nogil=True, schedule='static', num_threads=num_threads):
        x = pos[i, 0]
        y = pos[i, 1]
        z = pos[i, 2]

        Li = sqrt(x*x + y*y + z*z)
        
        # Avoid division by zero
        if Li == 0.0:
            tarpos[i, 0] = tarpos[i, 1] = tarpos[i, 2] = 0.0
            d[i] = 0.0
            L[i] = 0.0
            continue

        xi = x / Li
        yi = y / Li
        zi = z / Li

        denom = (xi / a) ** 2 + (yi / b) ** 2 + (zi / c) ** 2
        di = sqrt(1.0 / denom)

        tarpos[i, 0] = xi * di
        tarpos[i, 1] = yi * di
        tarpos[i, 2] = zi * di
        d[i] = di
        L[i] = Li

    return tarpos, L - d

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t, ndim=1] f_ray_ellipsoid(double a, double b, double c, 
                                                    np.ndarray[np.float64_t, ndim=2] pos):
    """
    Computes the ray distance function for the ellipsoid.
    """
    cdef int n = pos.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] r = np.zeros(n, dtype=np.float64)
    cdef double x, y, z, xi, yi, zi, Li, di, denom
    cdef int i
    cdef int num_threads = config['general']['number_of_threads']
    # Use sequential processing for small arrays
    if n < 1000:  # Higher threshold for small arrays
        for i in range(n):
            x = pos[i, 0]
            y = pos[i, 1]
            z = pos[i, 2]
            
            Li = sqrt(x*x + y*y + z*z)
            
            # Avoid division by zero
            if Li == 0.0:
                r[i] = 0.0
                continue
                
            xi = x / Li
            yi = y / Li
            zi = z / Li
            
            denom = (xi / a) ** 2 + (yi / b) ** 2 + (zi / c) ** 2
            di = sqrt(1.0 / denom)
            r[i] = Li / di
    else:
        
        for i in prange(n, nogil=True, schedule='static', num_threads=num_threads):
            x = pos[i, 0]
            y = pos[i, 1]
            z = pos[i, 2]

            Li = sqrt(x*x + y*y + z*z)

            # Avoid division by zero
            if Li == 0.0:
                r[i] = 0.0
                continue

            xi = x / Li
            yi = y / Li
            zi = z / Li

            denom = (xi / a) ** 2 + (yi / b) ** 2 + (zi / c) ** 2
            di = sqrt(1.0 / denom)
            r[i] = Li / di

    return r

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t, ndim=2] IntersectLinesEllipsoid(
    double a, double b, double c,
    np.ndarray[np.float64_t, ndim=2] pos1, 
    np.ndarray[np.float64_t, ndim=2] pos2
):
    """
    Compute the intersection of lines with the ellipsoid.
    Returns a Nx2 array of t values where the line intersects the ellipsoid.
    """
    cdef int n = pos1.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] vects = unit_vector3d(pos2 - pos1)
    cdef np.ndarray[np.float64_t, ndim=2] ts = -np.ones((n, 2), dtype=np.float64)
    cdef int i

    cdef double a2 = a*a
    cdef double b2 = b*b
    cdef double c2 = c*c

    cdef double ell11, ell01, ell00, D, SqrtD
    cdef double vx, vy, vz
    cdef double x, y, z
    cdef int num_threads = config['general']['number_of_threads']
    for i in prange(n, nogil=True, schedule='static', num_threads=num_threads):
        x = pos1[i, 0]
        y = pos1[i, 1]
        z = pos1[i, 2]

        vx = vects[i, 0]
        vy = vects[i, 1]
        vz = vects[i, 2]

        ell11 = vx * vx / a2 + vy * vy / b2 + vz * vz / c2
        ell01 = vx * x / a2 + vy * y / b2 + vz * z / c2
        ell00 = x * x / a2 + y * y / b2 + z * z / c2

        D = ell01*ell01 - ell11 * (ell00 - 1)
        if D > 0:
            SqrtD = sqrt(D)
            ts[i, 0] = (-ell01 - SqrtD) / ell11
            ts[i, 1] = (-ell01 + SqrtD) / ell11
        elif D == 0:
            ts[i, 0] = ts[i, 1] = (-ell01) / ell11

    return ts

# -------------------- Main Ellipsoid Class --------------------

class Ellipsoid(GeometryBase):
    """
    Ellipsoid geometry class with semi-axes a, b, c.
    
    This class represents an ellipsoid defined by three semi-axes
    a >= b >= c. It implements various geometric functions like
    surface evaluation, ray intersection, etc.
    """
    
    # Class attributes
    PN = ('a', 'eps_ab', 'eps_bc')
    LB = {'a': 0.1, 'eps_ab': 0.01, 'eps_bc': 0.01}
    UB = {'a': np.inf, 'eps_ab': 0.99, 'eps_bc': 0.99}
    MaxIterationClosed = 100

    def __init__(self, *args, **kwargs):
        """
        Initialize the ellipsoid with given parameters.

        Parameters
        ----------
        *args : tuple
            Positional arguments (unused).
        **kwargs : dict
            Parameters to initialize the ellipsoid.
        """
        self.parameters = self.init_parameters(**kwargs)
    
    @staticmethod
    def init_parameters(**kwargs):
        """
        Initialize and return the parameters of the ellipsoid.

        Parameters
        ----------
        **kwargs : dict
            Parameters to initialize.

        Returns
        -------
        Parameters
            An instance with initialized parameters.
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
        Return default parameters for the ellipsoid.

        Returns
        -------
        Parameters
            Default parameters instance.
        """
        return Ellipsoid.init_parameters(a=3.0, eps_ab=0.2, eps_bc=0.5)

    def __call__(self, pos):
        """
        Evaluate the ellipsoid function at given positions.

        Parameters
        ----------
        pos : array_like
            Positions to evaluate.

        Returns
        -------
        ndarray
            Evaluated values.
        """
        cdef np.ndarray[np.float64_t, ndim=2] pos_array
        if (len(np.shape(pos)) == 2) and (np.shape(pos)[1] == 3):
            pos_array = np.asarray(pos, dtype=np.float64)
        else:
            pos_array = np.asarray([pos], dtype=np.float64)
            
        return f_ellipsoid(self['a'], self['b'], self['c'], pos_array)

    def jacobian(self, pos):
        """
        Compute the Jacobian at given positions.

        Parameters
        ----------
        pos : array_like
            Positions to evaluate.

        Returns
        -------
        tuple
            Jacobian components.
        """
        cdef np.ndarray[np.float64_t, ndim=2] pos_array
        if (len(np.shape(pos)) == 2) and (np.shape(pos)[1] == 3):
            pos_array = np.asarray(pos, dtype=np.float64)
        else:
            pos_array = np.asarray([pos], dtype=np.float64)
            
        return f_ellipsoid_jacobian(self['a'], self['b'], self['c'], pos_array)

    def ray_intersect(self, pos):
        """
        Compute ray intersections with the ellipsoid.

        Parameters
        ----------
        pos : array_like
            Ray origin positions.

        Returns
        -------
        tuple
            Intersection points and distances.
        """
        cdef np.ndarray[np.float64_t, ndim=2] pos_array
        if (len(np.shape(pos)) == 2) and (np.shape(pos)[1] == 3):
            pos_array = np.asarray(pos, dtype=np.float64)
        else:
            pos_array = np.asarray([pos], dtype=np.float64)
            
        return IntersectRaysEllipsoid(self['a'], self['b'], self['c'], pos_array)

    def line_intersect(self, pos1, pos2):
        """
        Compute line intersections with the ellipsoid.

        Parameters
        ----------
        pos1, pos2 : array_like
            Start and end points of lines.

        Returns
        -------
        ndarray
            Intersection t-values.
        """
        cdef np.ndarray[np.float64_t, ndim=2] pos1_array, pos2_array
        
        if (len(np.shape(pos1)) == 2) and (np.shape(pos1)[1] == 3):
            pos1_array = np.asarray(pos1, dtype=np.float64)
        else:
            pos1_array = np.asarray([pos1], dtype=np.float64)
            
        if (len(np.shape(pos2)) == 2) and (np.shape(pos2)[1] == 3):
            pos2_array = np.asarray(pos2, dtype=np.float64)
        else:
            pos2_array = np.asarray([pos2], dtype=np.float64)

        return IntersectLinesEllipsoid(self['a'], self['b'], self['c'], pos1_array, pos2_array)

    def f_ray_d(self, pos):
        """
        Compute ray distance function.

        Parameters
        ----------
        pos : array_like
            Positions to evaluate.

        Returns
        -------
        ndarray
            Ray distances.
        """
        cdef np.ndarray[np.float64_t, ndim=2] pos_array
        if (len(np.shape(pos)) == 2) and (np.shape(pos)[1] == 3):
            pos_array = np.asarray(pos, dtype=np.float64)
        else:
            pos_array = np.asarray([pos], dtype=np.float64)
            
        return f_ray_ellipsoid(self['a'], self['b'], self['c'], pos_array)

    @staticmethod
    def quick_call(double a, double eps_ab, double eps_bc, pos):
        """
        Quickly evaluate ellipsoid function.

        Parameters
        ----------
        a, eps_ab, eps_bc : float
            Ellipsoid parameters.
        pos : array_like
            Positions to evaluate.

        Returns
        -------
        ndarray
            Function values.
        """
        cdef double b = a * (1.0 - eps_ab)
        cdef double c = b * (1.0 - eps_bc)
        
        cdef np.ndarray[np.float64_t, ndim=2] pos_array
        if (len(np.shape(pos)) == 2) and (np.shape(pos)[1] == 3):
            pos_array = np.asarray(pos, dtype=np.float64)
        else:
            pos_array = np.asarray([pos], dtype=np.float64)
            
        return f_ellipsoid(a, b, c, pos_array)

    @staticmethod
    def quick_f_ray_d(double a, double eps_ab, double eps_bc, pos):
        """
        Quickly evaluate ray distance function.

        Parameters
        ----------
        a, eps_ab, eps_bc : float
            Ellipsoid parameters.
        pos : array_like
            Positions to evaluate.

        Returns
        -------
        ndarray
            Ray distances.
        """
        cdef double b = a * (1.0 - eps_ab)
        cdef double c = b * (1.0 - eps_bc)
        
        cdef np.ndarray[np.float64_t, ndim=2] pos_array
        if (len(np.shape(pos)) == 2) and (np.shape(pos)[1] == 3):
            pos_array = np.asarray(pos, dtype=np.float64)
        else:
            pos_array = np.asarray([pos], dtype=np.float64)
            
        return f_ray_ellipsoid(a, b, c, pos_array)

    @staticmethod
    def quick_ray_dist(double a, double eps_ab, double eps_bc, pos):
        """
        Quickly compute ray intersection distances.

        Parameters
        ----------
        a, eps_ab, eps_bc : float
            Ellipsoid parameters.
        pos : array_like
            Ray origin positions.

        Returns
        -------
        ndarray
            Ray distances.
        """
        cdef double b = a * (1.0 - eps_ab)
        cdef double c = b * (1.0 - eps_bc)
        
        cdef np.ndarray[np.float64_t, ndim=2] pos_array
        if (len(np.shape(pos)) == 2) and (np.shape(pos)[1] == 3):
            pos_array = np.asarray(pos, dtype=np.float64)
        else:
            pos_array = np.asarray([pos], dtype=np.float64)
            
        return IntersectRaysEllipsoid(a, b, c, pos_array)[1]

    @staticmethod
    def quick_line_intersect(double a, double eps_ab, double eps_bc, pos1, pos2):
        """
        Quickly compute line intersections.

        Parameters
        ----------
        a, eps_ab, eps_bc : float
            Ellipsoid parameters.
        pos1, pos2 : array_like
            Start and end points of lines.

        Returns
        -------
        ndarray
            Intersection t-values.
        """
        cdef double b = a * (1.0 - eps_ab)
        cdef double c = b * (1.0 - eps_bc)
        
        cdef np.ndarray[np.float64_t, ndim=2] pos1_array, pos2_array
        
        if (len(np.shape(pos1)) == 2) and (np.shape(pos1)[1] == 3):
            pos1_array = np.asarray(pos1, dtype=np.float64)
        else:
            pos1_array = np.asarray([pos1], dtype=np.float64)
            
        if (len(np.shape(pos2)) == 2) and (np.shape(pos2)[1] == 3):
            pos2_array = np.asarray(pos2, dtype=np.float64)
        else:
            pos2_array = np.asarray([pos2], dtype=np.float64)
            
        return IntersectLinesEllipsoid(a, b, c, pos1_array, pos2_array)

    @staticmethod
    def quick_jacobian(double a, double b, double c, pos):
        """
        Quickly compute Jacobian.

        Parameters
        ----------
        a, b, c : float
            Semi-axis lengths.
        pos : array_like
            Positions to evaluate.

        Returns
        -------
        tuple
            Jacobian components.
        """
        cdef np.ndarray[np.float64_t, ndim=2] pos_array
        if (len(np.shape(pos)) == 2) and (np.shape(pos)[1] == 3):
            pos_array = np.asarray(pos, dtype=np.float64)
        else:
            pos_array = np.asarray([pos], dtype=np.float64)
            
        return f_ellipsoid_jacobian(a, b, c, pos_array)