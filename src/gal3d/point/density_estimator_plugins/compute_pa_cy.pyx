# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange

cdef double VOLUME_FACTOR = 4.0 / 3.0 * np.pi

cdef double calc_mean_pa(int i, int num_near, double[:, :] n_d,
                         int[:, :] n_index, double[:] mass, double distance_upper_bound) nogil:
    cdef double n_mass = 0.0
    cdef int valid_count = 0
    cdef int j
    for j in range(num_near):
        if n_d[i, j] < distance_upper_bound:
            n_mass += mass[n_index[i, j]]
            valid_count += 1
    return n_mass / valid_count if valid_count > 0 else 0.0

cdef double calc_volume_pa(int i, int num_near, double[:, :] n_d,
                          int[:, :] n_index, double[:] mass,
                          double distance_upper_bound) nogil:
    cdef double n_mass = 0.0
    cdef double n_d_max = 0.0
    cdef int j
    for j in range(num_near):
        if n_d[i, j] < distance_upper_bound:
            n_mass += mass[n_index[i, j]]
            if n_d[i, j] > n_d_max:
                n_d_max = n_d[i, j]
    if n_d_max == 0.0:
        n_d_max = distance_upper_bound
    return n_mass / (VOLUME_FACTOR * n_d_max ** 3)

def cal_pa(np.ndarray[double, ndim=2] n_d,
           np.ndarray[int, ndim=2] n_index,
           np.ndarray[double, ndim=1] mass,
           str pa_mode,
           double distance_upper_bound=np.inf) -> np.ndarray:
    cdef int m = n_d.shape[0]
    cdef int num_near = n_d.shape[1]
    cdef np.ndarray[double, ndim=1] fit_pa = np.zeros(m, dtype=np.float64)
    cdef int i

    # Use memoryviews for GIL-free access
    cdef double[:, :] n_d_mv = n_d
    cdef int[:, :] n_index_mv = n_index
    cdef double[:] mass_mv = mass

    if pa_mode == 'Mean':
        for i in prange(m, nogil=True):
            fit_pa[i] = calc_mean_pa(i, num_near, n_d_mv, n_index_mv, mass_mv, distance_upper_bound)
    else:
        for i in prange(m, nogil=True):
            fit_pa[i] = calc_volume_pa(i, num_near, n_d_mv, n_index_mv, mass_mv, distance_upper_bound)
    return fit_pa