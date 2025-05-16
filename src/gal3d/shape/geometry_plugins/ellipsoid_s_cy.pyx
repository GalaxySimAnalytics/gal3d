# distutils: language=c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
from libc.math cimport pow as c_pow, log, abs, sqrt
from cython.parallel import prange
import cython

from ..geometry import GeometryBase, Parameters
from ...util.array_operate import unit_vector3d, vector_length3d
from gal3d import config

np.import_array()

__all__ = ['Ellipsoid_S']
ctypedef np.float64_t DTYPE_t

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def f_shaped_ellipsoid(double a, double b, double c, double Sa, double Sb, double Sc, 
                      np.ndarray[DTYPE_t, ndim=2] pos):
    """Compute the shaped ellipsoid function values for given positions."""
    cdef:
        int i, n = pos.shape[0]
        np.ndarray[DTYPE_t, ndim=1] result = np.zeros(n, dtype=np.float64)
        double h1, h2, h3
    cdef int num_threads = config['general']['number_of_threads']
    # Skip parallelization for small arrays    
    if n < 500:
        for i in range(n):
            h1 = pos[i, 0] * pos[i, 0] / (a * a)
            h2 = pos[i, 1] * pos[i, 1] / (b * b)
            h3 = pos[i, 2] * pos[i, 2] / (c * c)
            result[i] = c_pow(h1, Sa) + c_pow(h2, Sb) + c_pow(h3, Sc)
    else:
        # Use prange for larger arrays
        for i in prange(n, nogil=True, num_threads=num_threads, schedule='static'):
            h1 = pos[i, 0] * pos[i, 0] / (a * a)
            h2 = pos[i, 1] * pos[i, 1] / (b * b)
            h3 = pos[i, 2] * pos[i, 2] / (c * c)
            result[i] = c_pow(h1, Sa) + c_pow(h2, Sb) + c_pow(h3, Sc)
        
    return result


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def f_shaped_ellipsoid_jacobian(double a, double b, double c, double Sa, double Sb, double Sc,
                               np.ndarray[DTYPE_t, ndim=2] pos):
    """Compute the Jacobian of the shaped ellipsoid function."""
    cdef:
        int i, n = pos.shape[0]
        np.ndarray[DTYPE_t, ndim=1] da = np.zeros(n, dtype=np.float64)
        np.ndarray[DTYPE_t, ndim=1] db = np.zeros(n, dtype=np.float64)
        np.ndarray[DTYPE_t, ndim=1] dc = np.zeros(n, dtype=np.float64)
        np.ndarray[DTYPE_t, ndim=1] dSa = np.zeros(n, dtype=np.float64)
        np.ndarray[DTYPE_t, ndim=1] dSb = np.zeros(n, dtype=np.float64)
        np.ndarray[DTYPE_t, ndim=1] dSc = np.zeros(n, dtype=np.float64)
        np.ndarray[DTYPE_t, ndim=1] dx = np.zeros(n, dtype=np.float64)
        np.ndarray[DTYPE_t, ndim=1] dy = np.zeros(n, dtype=np.float64)
        np.ndarray[DTYPE_t, ndim=1] dz = np.zeros(n, dtype=np.float64)
        double cof0, cof1, cof2
    
    for i in range(n):
        cof0 = c_pow(pos[i, 0] * pos[i, 0], Sa)
        cof1 = c_pow(pos[i, 1] * pos[i, 1], Sb)
        cof2 = c_pow(pos[i, 2] * pos[i, 2], Sc)
        
        # Skip computation for zero values to avoid division by zero
        if pos[i, 0] != 0:
            dx[i] = 2 * Sa * cof0 / pos[i, 0] / c_pow(a*a, Sa)
        if pos[i, 1] != 0:
            dy[i] = 2 * Sb * cof1 / pos[i, 1] / c_pow(b*b, Sb)
        if pos[i, 2] != 0:
            dz[i] = 2 * Sc * cof2 / pos[i, 2] / c_pow(c*c, Sc)

        da[i] = -2 * Sa * cof0 / c_pow(a*a, 2 * Sa + 1)
        db[i] = -2 * Sb * cof1 / c_pow(b*b, 2 * Sb + 1)
        dc[i] = -2 * Sc * cof2 / c_pow(c*c, 2 * Sc + 1)

        dSa[i] = 2 * cof0 * (log(abs(pos[i, 0])) - log(a)) / c_pow(a*a, Sa)
        dSb[i] = 2 * cof1 * (log(abs(pos[i, 1])) - log(b)) / c_pow(b*b, Sb)
        dSc[i] = 2 * cof2 * (log(abs(pos[i, 2])) - log(c)) / c_pow(c*c, Sc)
    
    return (da, db, dc, dSa, dSb, dSc, dx, dy, dz)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def f_shaped_ellipsoids(np.ndarray[DTYPE_t, ndim=1] a, 
                       np.ndarray[DTYPE_t, ndim=1] b, 
                       np.ndarray[DTYPE_t, ndim=1] c, 
                       np.ndarray[DTYPE_t, ndim=1] Sa, 
                       np.ndarray[DTYPE_t, ndim=1] Sb, 
                       np.ndarray[DTYPE_t, ndim=1] Sc, 
                       np.ndarray[DTYPE_t, ndim=2] pos):
    """Compute multiple shaped ellipsoids for an array of parameters."""
    cdef:
        int i, j, n_ellipsoids = len(a), n_pos = pos.shape[0]
        np.ndarray[DTYPE_t, ndim=2] res = np.zeros((n_ellipsoids, n_pos), dtype=np.float64)
        double h1, h2, h3
    
    for i in range(n_ellipsoids):
        for j in range(n_pos):
            h1 = pos[j, 0] * pos[j, 0] / (a[i] * a[i])
            h2 = pos[j, 1] * pos[j, 1] / (b[i] * b[i])
            h3 = pos[j, 2] * pos[j, 2] / (c[i] * c[i])
            res[i, j] = c_pow(h1, Sa[i]) + c_pow(h2, Sb[i]) + c_pow(h3, Sc[i])
    
    return res


@cython.cdivision(True)
cdef double _iter_f_IntersectRayEllipsoid_S(double d, double Sa, double Sb, double Sc, 
                                          double Ex, double Ey, double Ez) nogil :
    """Helper function for ray-ellipsoid intersection."""
    cdef:
        double dd = d * d
        double ExddSa = Ex * c_pow(dd, Sa)
        double EyddSb = Ey * c_pow(dd, Sb)
        double EzddSc = Ez * c_pow(dd, Sc)
        double f = ExddSa + EyddSb + EzddSc - 1
        double df = 2 * (Sa * ExddSa + Sb * EyddSb + Sc * EzddSc) / d
    
    return -f / df


@cython.cdivision(True)
cdef (double, double, double, double, double) IntersectRayEllipsoid_S(
        double a, double b, double c, double Sa, double Sb, double Sc,
        double x, double y, double z, int maxIterations) nogil:
    """Find the intersection of a ray with the shaped ellipsoid."""
    cdef:
        double L, xi, yi, zi, Ex, Ey, Ez, epsilon, d0, d1
        int i = 0
    
    # Calculate ray direction
    L = sqrt(x*x + y*y + z*z)  # Replace RobustLength3d with direct calculation
    xi = x / L
    yi = y / L
    zi = z / L
    
    # Calculate ellipsoid parameters
    Ex = c_pow((xi / a) * (xi / a), Sa)
    Ey = c_pow((yi / b) * (yi / b), Sb)
    Ez = c_pow((zi / c) * (zi / c), Sc)
    
    # Iteratively solve for intersection
    epsilon = 1e-9
    d0 = (a + c) / 2
    
    while True:
        if i > maxIterations:
            break
        d1 = d0 + _iter_f_IntersectRayEllipsoid_S(d0, Sa, Sb, Sc, Ex, Ey, Ez)
        if abs(d1 - d0) < epsilon:
            break
        d0 = d1
        i += 1
    
    return d0*xi, d0*yi, d0*zi, d0, L


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def IntersectRaysEllipsoid_S(double a, double b, double c, double Sa, double Sb, double Sc,
                            np.ndarray[DTYPE_t, ndim=2] pos, int maxIterations):
    """Find intersections of multiple rays with the shaped ellipsoid."""
    cdef:
        int i, n = pos.shape[0]
        np.ndarray[DTYPE_t, ndim=2] tarpos = np.zeros((n, 3), dtype=np.float64)
        np.ndarray[DTYPE_t, ndim=1] d = np.zeros(n, dtype=np.float64)
        np.ndarray[DTYPE_t, ndim=1] L = np.zeros(n, dtype=np.float64)
        double x, y, z, d_val, L_val
    cdef int num_threads = config['general']['number_of_threads']
    # Skip parallelization for small arrays
    if n < 500:
        for i in range(n):
            x, y, z, d_val, L_val = IntersectRayEllipsoid_S(
                a, b, c, Sa, Sb, Sc, pos[i, 0], pos[i, 1], pos[i, 2], maxIterations
            )
            tarpos[i, 0] = x
            tarpos[i, 1] = y
            tarpos[i, 2] = z
            d[i] = d_val
            L[i] = L_val
    else:
        # Use nogil block for parallel execution
        with nogil:
            for i in prange(n, num_threads=num_threads, schedule='static'):
                x, y, z, d_val, L_val = IntersectRayEllipsoid_S(
                    a, b, c, Sa, Sb, Sc, pos[i, 0], pos[i, 1], pos[i, 2], maxIterations
                )
                tarpos[i, 0] = x
                tarpos[i, 1] = y
                tarpos[i, 2] = z
                d[i] = d_val
                L[i] = L_val
    
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros(n, dtype=np.float64)
    for i in range(n):
        result[i] = L[i] - d[i]
    return tarpos, result


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def f_ray_shaped_ellipsoid(double a, double b, double c, double Sa, double Sb, double Sc,
                          np.ndarray[DTYPE_t, ndim=2] pos, int maxIterations):
    """Compute ray-ellipsoid distance ratios (OpenMP-friendly, inlined)."""
    cdef:
        int i, n = pos.shape[0]
        int iteration  # Use iteration instead of j
        np.ndarray[DTYPE_t, ndim=2] tarpos = np.zeros((n, 3), dtype=np.float64)
        np.ndarray[DTYPE_t, ndim=1] d = np.zeros(n, dtype=np.float64)
        np.ndarray[DTYPE_t, ndim=1] L = np.zeros(n, dtype=np.float64)
        np.ndarray[DTYPE_t, ndim=1] r = np.zeros(n, dtype=np.float64)
        double x, y, z, d_val, L_val
        double xi, yi, zi, Ex, Ey, Ez, epsilon, d0, d1, Ltmp
        int maxIter
        double dd, ExddSa, EyddSb, EzddSc, f, df
        bint converged  # Flag to indicate convergence

    cdef int num_threads = config['general']['number_of_threads']
    epsilon = 1e-9
    maxIter = maxIterations

    if n < 500:
        for i in range(n):
            j = 0
            x = pos[i, 0]
            y = pos[i, 1]
            z = pos[i, 2]
            # Inline Newton iteration
            Ltmp = sqrt(x*x + y*y + z*z)
            xi = x / Ltmp
            yi = y / Ltmp
            zi = z / Ltmp
            Ex = c_pow((xi / a) * (xi / a), Sa)
            Ey = c_pow((yi / b) * (yi / b), Sb)
            Ez = c_pow((zi / c) * (zi / c), Sc)
            d0 = (a + c) / 2
            while True:
                if j > maxIter:
                    break
                dd = d0 * d0
                ExddSa = Ex * c_pow(dd, Sa)
                EyddSb = Ey * c_pow(dd, Sb)
                EzddSc = Ez * c_pow(dd, Sc)
                f = ExddSa + EyddSb + EzddSc - 1
                df = 2 * (Sa * ExddSa + Sb * EyddSb + Sc * EzddSc) / d0
                d1 = d0 - f / df
                if abs(d1 - d0) < epsilon:
                    break
                d0 = d1
                j += 1
            d_val = d0
            L_val = Ltmp
            tarpos[i, 0] = d_val * xi
            tarpos[i, 1] = d_val * yi
            tarpos[i, 2] = d_val * zi
            d[i] = d_val
            L[i] = L_val
            r[i] = L_val / d_val
    else:
        with nogil:
            for i in prange(n, num_threads=num_threads, schedule='static'):
                x = pos[i, 0]
                y = pos[i, 1]
                z = pos[i, 2]
                # Inline Newton iteration
                Ltmp = sqrt(x*x + y*y + z*z)
                xi = x / Ltmp
                yi = y / Ltmp
                zi = z / Ltmp
                Ex = c_pow((xi / a) * (xi / a), Sa)
                Ey = c_pow((yi / b) * (yi / b), Sb)
                Ez = c_pow((zi / c) * (zi / c), Sc)
                d0 = (a + c) / 2
                
                # Use a for loop with fixed iterations instead of while loop
                converged = False
                for iteration in range(maxIter):
                    dd = d0 * d0
                    ExddSa = Ex * c_pow(dd, Sa)
                    EyddSb = Ey * c_pow(dd, Sb)
                    EzddSc = Ez * c_pow(dd, Sc)
                    f = ExddSa + EyddSb + EzddSc - 1
                    df = 2 * (Sa * ExddSa + Sb * EyddSb + Sc * EzddSc) / d0
                    d1 = d0 - f / df
                    
                    if abs(d1 - d0) < epsilon:
                        converged = True
                        break
                    
                    d0 = d1
                
                d_val = d0
                L_val = Ltmp
                tarpos[i, 0] = d_val * xi
                tarpos[i, 1] = d_val * yi
                tarpos[i, 2] = d_val * zi
                d[i] = d_val
                L[i] = L_val
                r[i] = L_val / d_val

    return r


@cython.cdivision(True)
cdef (double, double) _iter_IntersectLineEllipsoid_S(
        np.ndarray[DTYPE_t, ndim=1] pos1,
        np.ndarray[DTYPE_t, ndim=1] vect,
        double t0, double a, double b, double c,
        double Sa, double Sb, double Sc, int maxIteration, 
        double epsilon, double delta_cut):
    """Iterative helper for line-ellipsoid intersection."""
    cdef:
        int i = 0
        double Ex, Ey, Ez, f, df_x, df_y, df_z, df, delta
        np.ndarray[DTYPE_t, ndim=1] posi = np.zeros(3, dtype=np.float64)
    
    posi = pos1 + t0 * vect
    
    while True:
        Ex = c_pow((posi[0] / a) * (posi[0] / a), Sa)
        Ey = c_pow((posi[1] / b) * (posi[1] / b), Sb)
        Ez = c_pow((posi[2] / c) * (posi[2] / c), Sc)
        f = Ex + Ey + Ez - 1
        
        if abs(f) < epsilon:
            break
        
        df_x = 0.0 if posi[0] == 0 else Ex * Sa * vect[0] / posi[0]
        df_y = 0.0 if posi[1] == 0 else Ey * Sb * vect[1] / posi[1]
        df_z = 0.0 if posi[2] == 0 else Ez * Sc * vect[2] / posi[2]
        df = 4 * (df_x + df_y + df_z)
        
        delta = -f / df
        if f < 2.0:  # when near target pos
            delta = min(delta_cut, delta)  # avoid large update 
            delta = max(-delta_cut, delta)
        t0 = t0 + delta
        posi = pos1 + t0 * vect
        i += 1
        
        if abs(delta) < epsilon or i > maxIteration:
            break
    
    Ex = c_pow((posi[0] / a) * (posi[0] / a), Sa)
    Ey = c_pow((posi[1] / b) * (posi[1] / b), Sb)
    Ez = c_pow((posi[2] / c) * (posi[2] / c), Sc)
    f = Ex + Ey + Ez - 1
    
    return t0, f


@cython.cdivision(True)
@cython.boundscheck(False)
def IntersectLineEllipsoid_S(double a, double b, double c, double Sa, double Sb, double Sc, 
                            np.ndarray[DTYPE_t, ndim=1] pos1, 
                            np.ndarray[DTYPE_t, ndim=1] vect,
                            double tmax, int maxIteration):
    """Find the intersection of a line with the shaped ellipsoid."""
    cdef:
        double delta_cut = c/2.0
        double epsilon = 1e-9
        double t0, t1, f0, f1, t2, f2, ti_min, ti_max
        np.ndarray[DTYPE_t, ndim=1] ts = -np.ones(2, dtype=np.float64)
    
    # From both ends of the line segment
    t0, f0 = _iter_IntersectLineEllipsoid_S(
        pos1, vect, epsilon, a, b, c, Sa, Sb, Sc, maxIteration, epsilon, delta_cut
    )
    t1, f1 = _iter_IntersectLineEllipsoid_S(
        pos1, vect, tmax, a, b, c, Sa, Sb, Sc, maxIteration, epsilon, delta_cut
    )
    
    # Check convergence cases
    if abs(t0 - t1) <= 10 * epsilon:
        if (abs(f0) >= 10 * epsilon) and (abs(f1) >= 10 * epsilon):
            return ts
        else:
            ts[0] = t0
            ts[1] = t1
            return ts
    
    if (abs(f0) <= 10 * epsilon) and (abs(f1) <= 10 * epsilon):
        ts[0] = t0
        ts[1] = t1
        return ts
    
    if (abs(f0) <= 10 * epsilon) or (abs(f1) <= 10 * epsilon):
        ti_min = t0
        ti_max = t1
        delta_cut = c/4.0
        
        while True:
            if abs(f0) > 10 * epsilon:
                t2, f2 = _iter_IntersectLineEllipsoid_S(
                    pos1, vect, ti_min + (ti_max - ti_min) / 3,
                    a, b, c, Sa, Sb, Sc, maxIteration, epsilon, delta_cut
                )
            else:
                t2, f2 = _iter_IntersectLineEllipsoid_S(
                    pos1, vect, ti_min + 2 * (ti_max - ti_min) / 3,
                    a, b, c, Sa, Sb, Sc, maxIteration, epsilon, delta_cut
                )
            
            if ((abs(t2 - ti_min) >= 10 * epsilon) and 
                (abs(t2 - ti_max) >= 10 * epsilon) and 
                (abs(f2) <= 10 * epsilon)):
                break
            
            if abs(f0) > 10 * epsilon:
                ti_min = ti_min + (ti_max - ti_min) / 3
            else:
                ti_max = ti_min + 2 * (ti_max - ti_min) / 3
                
            if abs(ti_min - ti_max) < 10 * epsilon:
                break
        
        if abs(f0) <= 10 * epsilon:
            ts[0] = t0
            ts[1] = t2
            return ts
        ts[0] = t2
        ts[1] = t1
        return ts
    
    return ts


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def IntersectLinesEllipsoid_S(double a, double b, double c, double Sa, double Sb, double Sc,
                             np.ndarray[DTYPE_t, ndim=2] pos1, 
                             np.ndarray[DTYPE_t, ndim=2] pos2,
                             int maxIteration):
    """Find intersections of multiple lines with the shaped ellipsoid."""
    cdef:
        int i, n = pos1.shape[0]
        np.ndarray[DTYPE_t, ndim=2] vects = unit_vector3d(pos2 - pos1)
        np.ndarray[DTYPE_t, ndim=1] tmax = vector_length3d(pos2 - pos1)
        np.ndarray[DTYPE_t, ndim=2] ts = np.ones((n, 2), dtype=np.float64)
    
    # Skip parallelization for small arrays
    if n < 500:
        for i in range(n):
            ts[i] = IntersectLineEllipsoid_S(
                a, b, c, Sa, Sb, Sc, pos1[i], vects[i], tmax[i], maxIteration
            )
    else:
        # Process in serial for now - this can't easily be parallelized due to Python object returns
        # If performance is critical, you could rewrite IntersectLineEllipsoid_S to work with nogil
        for i in range(n):
            ts[i] = IntersectLineEllipsoid_S(
                a, b, c, Sa, Sb, Sc, pos1[i], vects[i], tmax[i], maxIteration
            )
    
    return ts


class Ellipsoid_S(GeometryBase):
    """
    A shaped ellipsoid geometry class.
    
    This class implements a generalized ellipsoid with shape parameters that allow
    for more flexible modeling of 3D geometric shapes.
    """
    
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
        """Evaluates the ellipsoid function at given positions."""
        pos = self._check_pos(pos)
        return f_shaped_ellipsoid(
            self['a'], self['b'], self['c'], self['sa'], self['sb'], self['sc'], pos
        )

    def jacobian(self, pos):
        """Computes the Jacobian of the ellipsoid function."""
        pos = self._check_pos(pos)
        return f_shaped_ellipsoid_jacobian(
            self['a'], self['b'], self['c'], self['sa'], self['sb'], self['sc'], pos
        )

    def ray_intersect(self, pos):
        """Computes intersections between rays and the ellipsoid."""
        pos = self._check_pos(pos)
        return IntersectRaysEllipsoid_S(
            self['a'], self['b'], self['c'], self['sa'], self['sb'], self['sc'],
            pos, Ellipsoid_S.MaxIterationDist
        )

    def line_intersect(self, pos1, pos2):
        """Computes intersections between lines and the ellipsoid."""
        pos1 = self._check_pos(pos1)
        pos2 = self._check_pos(pos2)
        return IntersectLinesEllipsoid_S(
            self['a'], self['b'], self['c'], self['sa'], self['sb'], self['sc'],
            pos1, pos2, Ellipsoid_S.MaxIterationLine
        )

    def f_ray_d(self, pos):
        """Computes ray-ellipsoid distance ratios."""
        pos = self._check_pos(pos)
        return f_ray_shaped_ellipsoid(
            self['a'], self['b'], self['c'], self['sa'], self['sb'], self['sc'],
            pos, Ellipsoid_S.MaxIterationDist
        )

    @staticmethod
    def quick_call(a, eps_ab, eps_bc, sa, sb, sc, pos):
        """Quickly evaluates the ellipsoid function with given parameters."""
        b = a * (1 - eps_ab)
        c = b * (1 - eps_bc)
        return f_shaped_ellipsoid(a, b, c, sa, sb, sc, pos)

    @staticmethod
    def quick_f_ray_d(a, eps_ab, eps_bc, sa, sb, sc, pos):
        """Quickly computes ray-ellipsoid distance ratios with given parameters."""
        b = a * (1 - eps_ab)
        c = b * (1 - eps_bc)
        return f_ray_shaped_ellipsoid(
            a, b, c, sa, sb, sc, pos, Ellipsoid_S.MaxIterationDist
        )

    @staticmethod
    def quick_ray_dist(a, eps_ab, eps_bc, sa, sb, sc, pos):
        """Quickly computes ray distances with given parameters."""
        b = a * (1 - eps_ab)
        c = b * (1 - eps_bc)
        return IntersectRaysEllipsoid_S(
            float(a), b, c, sa, sb, sc, pos, Ellipsoid_S.MaxIterationDist
        )[1]

    @staticmethod
    def quick_line_intersect(a, eps_ab, eps_bc, sa, sb, sc, pos1, pos2):
        """Quickly computes line intersections with given parameters."""
        b = a * (1 - eps_ab)
        c = b * (1 - eps_bc)
        return IntersectLinesEllipsoid_S(
            float(a), float(b), float(c), float(sa), float(sb), float(sc),
            pos1, pos2, Ellipsoid_S.MaxIterationLine
        )

    @staticmethod
    def quick_jacobian(a, b, c, sa, sb, sc, pos):
        """Quickly computes the Jacobian with given parameters."""
        return f_shaped_ellipsoid_jacobian(
            float(a), float(b), float(c), float(sa), float(sb), float(sc), pos
        )
    
    @staticmethod
    def _check_pos(pos):
        """Ensure pos is a 2D numpy array of shape (N, 3)."""
        pos = np.asarray(pos, dtype=np.float64)
        if pos.ndim == 1:
            pos = pos[np.newaxis, :]
        if pos.shape[1] != 3:
            raise ValueError("Input position must have shape (N, 3)")
        return pos