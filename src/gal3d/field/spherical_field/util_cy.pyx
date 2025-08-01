# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np

cimport numpy as np
from libc.math cimport cos, pi, sin, sqrt

import cython
from cython.parallel import prange

from gal3d import config

# Import dependencies
from ...point.util import abc_vect
from ...util.array_operate_cy import (
    Matmul,
    trans_to_Cartesian_coordinates,
    trans_to_Spherical_coordinates,
    unit_vector3d,
    vector_length3d,
)

# For OpenMP configuration


# Define numpy data types
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

__all__ = [
    'trans_to_Spherical_coordinates',
    'trans_to_Cartesian_coordinates',
    'fibonacci_sampling',
    'vector_length3d',
    'unit_vector3d',
    'Matmul',
]

def fibonacci_sampling(int Num_sampling=256):
    """
    Generate points using Fibonacci sphere sampling
    
    Parameters:
        Num_sampling: int, default 256,
            the number of points

    Return: [x,y,z],[r,phi,theta]
    """
    cdef:
        double Golden_Ratio = (sqrt(5) + 1) / 2 - 1
        np.ndarray[DTYPE_t, ndim=2] sampling_pos = np.zeros((Num_sampling, 3), dtype=DTYPE)
        int n, i
        double term, z
    cdef int num_threads = config['general']['number_of_threads']
    # Parallelize the point generation
    for n in prange(Num_sampling, nogil=True, num_threads=num_threads):
        i = n + 1
        z = (2.0 * i - 1.0) / Num_sampling - 1.0
        term = 1.0 - (z ** 2)
        sampling_pos[n, 0] = sqrt(term) * cos(2.0 * pi * i * Golden_Ratio)
        sampling_pos[n, 1] = sqrt(term) * sin(2.0 * pi * i * Golden_Ratio)
        sampling_pos[n, 2] = z
    
    # Convert to spherical coordinates
    cdef np.ndarray[DTYPE_t, ndim=2] sampling_sphere_coor = trans_to_Spherical_coordinates(sampling_pos)
    return sampling_pos, sampling_sphere_coor

def iso_profile_by_moi(
    np.ndarray[DTYPE_t, ndim=2] points,
    np.ndarray[DTYPE_t, ndim=2] pas,
    double res_b,
    double res_c
):
    """
    Calculate isophote profile using moment of inertia
    """
    cdef:
        double c_cos_max = res_c
        double c_cos_min = -res_c
        double b_cos_max = sqrt(1 - res_b**2)
        double b_cos_min = -b_cos_max
        np.ndarray[DTYPE_t, ndim=1] iso_pro_pa = np.zeros(pas.shape[1], dtype=DTYPE)
        int i
        tuple result
        np.ndarray[DTYPE_t, ndim=2] abc, rota, new_pos
        np.ndarray[np.uint8_t, ndim=1] sel1, sel2

    #cdef int num_threads = config['general']['number_of_threads']
    
    for i in range(pas.shape[1]):
        result = abc_vect(points, pas[:, i])
        abc = result[0]
        rota = result[1]
        
        new_pos = Matmul(rota.T, points.T).T
        sel1 = (c_cos_min <= new_pos[:, 2]) & (new_pos[:, 2] <= c_cos_max)
        sel2 = (b_cos_max <= new_pos[:, 0]) | (new_pos[:, 0] <= b_cos_min)
        iso_pro_pa[i] = np.mean(pas[:, i][(sel1) & (sel2)])
    
    return iso_pro_pa


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_t compute_max_mean(
    DTYPE_t[:, :] pas_mv,
    np.uint8_t[:, :] sel_mv,
    int m,
    int i
) nogil:
    cdef DTYPE_t max_val = -1e99
    cdef DTYPE_t mean_val
    cdef int cnt, j, k
    for j in range(m):
        mean_val = 0.0
        cnt = 0
        for k in range(m):
            if sel_mv[j, k]:
                mean_val += pas_mv[k, i]
                cnt += 1
        if cnt > 0:
            mean_val /= cnt
        else:
            mean_val = 0.0
        if mean_val > max_val:
            max_val = mean_val
    return max_val

def iso_profile_by_pair(
    np.ndarray[DTYPE_t, ndim=2] points,
    np.ndarray[DTYPE_t, ndim=2] pas,
    double res_b,
    double res_c
):
    """
    Calculate isophote profile using pair-wise comparisons (optimized)
    """
    cdef:
        int n = pas.shape[1]
        double angle_max = sqrt(1 - (res_b / 2 + res_c / 2) ** 2)
        np.ndarray[DTYPE_t, ndim=2] points_dist = Matmul(points, points.T)
        np.ndarray[np.uint8_t, ndim=2] sel = (points_dist < -angle_max) | (points_dist > angle_max)
        np.ndarray[DTYPE_t, ndim=1] iso_pro_pa = np.zeros(n, dtype=DTYPE)
        int i, m = sel.shape[0]
        DTYPE_t[:, :] pas_mv = pas
        np.uint8_t[:, :] sel_mv = sel

    cdef int num_threads = config['general']['number_of_threads']

    for i in prange(n, nogil=True, num_threads=num_threads):
        iso_pro_pa[i] = compute_max_mean(pas_mv, sel_mv, m, i)

    return np.asarray(iso_pro_pa)