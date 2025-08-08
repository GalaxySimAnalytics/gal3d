"""
Module for computing centers, inertia tensors, and principal axes using Cython-accelerated routines.

Functions
---------
- centroid(pos): Returns the geometric center of positions.
- center_of_mass(pos, mass): Returns the mass-weighted center of mass.
- shrink_sphere_center(...): Iteratively computes the center using the shrinking sphere method.
- moment_of_inertia(pos, mass): Computes the inertia tensor.
- abc_vect(pos, mass): Returns eigenvalues and eigenvectors of the inertia tensor (i.e., principal axes).
"""

import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from cython.parallel import prange

from gal3d.config import config

ctypedef fused DTYPE_t:
    np.float32_t
    np.float64_t

cdef int get_num_threads():
    return config.general.number_of_threads

# Geometric centroid
cpdef np.ndarray centroid(np.ndarray[DTYPE_t, ndim=2] pos):
    cdef int i, n = pos.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] cenpos = np.zeros(3, dtype=pos.dtype)
    # Use prange for parallel summation
    cdef int num_threads = get_num_threads()
    for i in prange(n, nogil=True, num_threads=num_threads):
        cenpos[0] += pos[i, 0]
        cenpos[1] += pos[i, 1]
        cenpos[2] += pos[i, 2]
    cenpos[0] /= n
    cenpos[1] /= n
    cenpos[2] /= n
    return cenpos

# Shrinking sphere center
cpdef tuple shrink_sphere_center(
    np.ndarray[DTYPE_t, ndim=2] pos,
    np.ndarray[DTYPE_t, ndim=1] weight,
    int min_points,
    int particles_for_second_radius,
    double shrink_factor,
    double starting_rmax,
    int itermax
):
    cdef int npart_all = pos.shape[0]
    cdef int npart = npart_all
    cdef double r2 = 0.0
    cdef double tot_weight = 0.0
    cdef np.ndarray[DTYPE_t, ndim=1] com = np.zeros(3, dtype=pos.dtype)
    cdef np.ndarray[DTYPE_t, ndim=1] com_x = centroid(pos)
    cdef int iternum = 0
    cdef double current_rmax = np.inf
    cdef double second_radius = -1.0
    cdef int i
    cdef double pix, piy, piz, wi, cx, cy, cz, current_rmax2
    while True:
        offset_x = 0.0
        offset_y = 0.0
        offset_z = 0.0
        cx = com_x[0]
        cy = com_x[1]
        cz = com_x[2]
        current_rmax2 = current_rmax * current_rmax
        npart = 0
        tot_weight = 0.0
        for i in range(npart_all):
            pix = pos[i, 0] - cx
            piy = pos[i, 1] - cy
            piz = pos[i, 2] - cz
            r2 = pix * pix + piy * piy + piz * piz
            if r2 < current_rmax2:
                wi = weight[i]
                offset_x += pix * wi
                offset_y += piy * wi
                offset_z += piz * wi
                tot_weight += wi
                npart += 1

        if npart < particles_for_second_radius and second_radius < 0.0:
            second_radius = sqrt(current_rmax2)

        if npart < min_points:
            break

        com[0] = cx + offset_x / tot_weight
        com[1] = cy + offset_y / tot_weight
        com[2] = cz + offset_z / tot_weight

        com_x[:] = com
        com[:] = 0.0

        iternum += 1
        if iternum > 1:
            current_rmax *= shrink_factor
        else:
            current_rmax = starting_rmax

        if iternum > itermax:
            break

    return com_x, current_rmax, second_radius, iternum

# Center of mass
cpdef np.ndarray center_of_mass(np.ndarray[DTYPE_t, ndim=2] pos, np.ndarray[DTYPE_t, ndim=1] mass):
    """
    Compute the mass-weighted center of mass.
    """
    cdef int i, j, n = pos.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] cenpos = np.zeros(3, dtype=pos.dtype)
    cdef double massum = 0.0
    cdef double s0 = 0.0
    cdef double s1 = 0.0
    cdef double s2 = 0.0
    cdef int num_threads = get_num_threads()
    # Compute total mass
    for i in range(n):
        massum += mass[i]
    if massum == 0.0:
        raise ValueError("Total mass is zero.")

    # Compute weighted sum for each coordinate (parallel over particles)
    for i in prange(n, nogil=True, num_threads=num_threads):
        s0 += pos[i, 0] * mass[i]
        s1 += pos[i, 1] * mass[i]
        s2 += pos[i, 2] * mass[i]
    cenpos[0] = s0 / massum
    cenpos[1] = s1 / massum
    cenpos[2] = s2 / massum
    return cenpos

# Moment of inertia tensor
cpdef np.ndarray moment_of_inertia(np.ndarray[DTYPE_t, ndim=2] pos, np.ndarray[DTYPE_t, ndim=1] m):
    cdef np.ndarray[DTYPE_t, ndim=2] I = np.zeros((3, 3), dtype=pos.dtype)
    cdef int i, j, k, n = pos.shape[0]
    cdef double total_mass = 0.0
    cdef double s
    cdef int num_threads = get_num_threads()
    for k in range(n):
        total_mass += m[k]
    for i in range(3):
        for j in range(3):
            s = 0.0
            for k in prange(n, nogil=True, num_threads=num_threads):
                s += m[k] * pos[k, i] * pos[k, j]
            I[i, j] = s / total_mass
    return I

# Principal axes and eigenvalues
cpdef tuple abc_vect(np.ndarray[DTYPE_t, ndim=2] pos, np.ndarray[DTYPE_t, ndim=1] mass):
    """
    Compute principal axes and eigenvalues of the inertia tensor.
    Returns (sorted eigenvalues, corresponding eigenvectors).
    """
    cdef np.ndarray[DTYPE_t, ndim=2] I = moment_of_inertia(pos, mass)
    D = np.linalg.eigh(I)
    align = np.argsort(D[0])[::-1]
    abc = D[0][align]
    axes = D[1][:, align]
    return abc, axes