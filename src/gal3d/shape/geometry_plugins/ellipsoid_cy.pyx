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
cpdef np.ndarray[np.float64_t, ndim=1] f_ellipsoid(double a, double b, double c, 
                                                np.ndarray[np.float64_t, ndim=2] pos):
    """Evaluate the ellipsoid function at the given positions."""
    cdef int n = pos.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros(n, dtype=np.float64)
    cdef double x, y, z
    cdef int i
    cdef int num_threads = get_num_threads()
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
    cdef int num_threads = get_num_threads()
    # Use sequential processing for small arrays
    if n < 1000:  # Higher threshold for small arrays
        for i in range(n):
            x = pos[i, 0]
            y = pos[i, 1]
            z = pos[i, 2]
            
            r[i] = sqrt(x*x/(a*a) + y*y/(b*b) + z*z/(c*c))
    else:
        
        for i in prange(n, nogil=True, schedule='static', num_threads=num_threads):
            x = pos[i, 0]
            y = pos[i, 1]
            z = pos[i, 2]

            r[i] = sqrt(x*x/(a*a) + y*y/(b*b) + z*z/(c*c))

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
