# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from cython.parallel import prange
cimport cython



from gal3d.config import config

# Initialize numpy
np.import_array()

cdef int get_num_threads():
    return config.general.number_of_threads

# -------------------- Utility functions--------------------

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=1] vector_length3d(np.ndarray[np.float64_t, ndim=2] pos):
    """Calculate the length of 3D vectors."""
    cdef int n = pos.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros(n, dtype=np.float64)
    cdef double x, y, z
    cdef int i
    cdef int num_threads = get_num_threads()
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
    cdef int num_threads = get_num_threads()
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
cpdef tuple f_ellipsoid(double a, double b, double c, 
                                                np.ndarray[np.float64_t, ndim=2] pos):
    """Evaluate the ellipsoid function at the given positions."""
    cdef int n = pos.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] r = np.zeros(n, dtype=np.float64)
    cdef double x, y, z
    cdef double x2, y2, z2
    cdef int i
    cdef double inv_a2 = 1.0 / (a * a)
    cdef double inv_b2 = 1.0 / (b * b)
    cdef double inv_c2 = 1.0 / (c * c)
    cdef int num_threads = get_num_threads()
    for i in prange(n, nogil=True, schedule='static', num_threads=num_threads):
        x = pos[i, 0]
        y = pos[i, 1]
        z = pos[i, 2]
        x2 = x*x
        y2 = y*y
        z2 = z*z
        result[i] = x2*inv_a2 + y2*inv_b2 + z2*inv_c2
        r[i] = sqrt(x2 + y2 + z2)
    return result, r

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t, ndim=1] area_factor(double a, double b, double c, 
                                                np.ndarray[np.float64_t, ndim=2] pos):
    cdef int n = pos.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros(n, dtype=np.float64)
    cdef double x, y, z, xi, yi , zi
    cdef double L, L2, xi2, yi2, zi2
    cdef double cos_theta, sin_theta, cos_phi, sin_phi

    cdef double area_factor
    cdef double F_r, F_theta, F_phi, r_theta, r_phi
    cdef double coef_x, coef_y, coef_z


    cdef double a2_inv = 1. / (a * a)
    cdef double b2_inv = 1. / (b * b)
    cdef double c2_inv = 1. / (c * c)

    cdef int i
    cdef int num_threads = get_num_threads()
    for i in prange(n, nogil=True, schedule='static', num_threads=num_threads):
        x = pos[i, 0]
        y = pos[i, 1]
        z = pos[i, 2]
        L2 = x*x + y*y + z*z
        L = sqrt(L2)
        xi = x / L
        yi = y / L
        zi = z / L
        xi2 = xi * xi
        yi2 = yi * yi
        zi2 = zi * zi
        cos_theta = zi
        sin_theta = sqrt(1 - zi2)
        cos_phi = xi / sqrt(xi2 + yi2)
        sin_phi = yi / sqrt(xi2 + yi2)

        F_r = xi2 * a2_inv + yi2 * b2_inv + zi2 * c2_inv
        
        coef_x = a2_inv * xi
        coef_y = b2_inv * yi
        coef_z = c2_inv * zi

        F_theta = coef_x*cos_theta*cos_phi+coef_y*cos_theta*sin_phi+coef_z*(-sin_theta)
        F_phi = coef_x*(-sin_theta*sin_phi)+ coef_y*(sin_theta*cos_phi)

        r_theta = - F_theta/(F_r)
        r_phi = - F_phi/(F_r)
        area_factor = sqrt(1 + (r_theta*r_theta + r_phi*r_phi/(sin_theta*sin_theta)))
        result[i] = area_factor

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
    cdef int num_threads = get_num_threads()
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
    cdef int num_threads = get_num_threads()
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

    return tarpos, L - d, L

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef tuple f_ray_ellipsoid(double a, double b, double c, 
                                                    np.ndarray[np.float64_t, ndim=2] pos):
    """
    Computes the ray distance function for the ellipsoid.
    """
    cdef int n = pos.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] res = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] r = np.zeros(n, dtype=np.float64)
    cdef double x2, y2, z2
    cdef double x, y, z, xi, yi, zi, Li, di, denom
    cdef int i
    cdef int num_threads = get_num_threads()
    cdef double inv_a2 = 1.0 / (a * a)
    cdef double inv_b2 = 1.0 / (b * b)
    cdef double inv_c2 = 1.0 / (c * c)
    for i in prange(n, nogil=True, schedule='static', num_threads=num_threads):
            x = pos[i, 0]
            y = pos[i, 1]
            z = pos[i, 2]
            x2 = x*x
            y2 = y*y
            z2 = z*z
            res[i] = sqrt(x2*inv_a2 + y2*inv_b2 + z2*inv_c2)
            r[i] = sqrt(x2 + y2 + z2)

    return res, r

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
    cdef int num_threads = get_num_threads()
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
