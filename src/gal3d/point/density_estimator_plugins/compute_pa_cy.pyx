# distutils: language = c++
# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange


cdef extern from "sph_density.h":
    double calc_sph_density(
        int i, int num_near,
        const double* n_d,
        const int* n_index,
        const double* mass,
        const double* hsm
    ) nogil


def cal_pa(np.ndarray[double, ndim=2] n_d,
           np.ndarray[int, ndim=2] n_index,
           np.ndarray[double, ndim=1] mass,
           np.ndarray[double, ndim=1] hsm,) -> np.ndarray:

    cdef int n = n_d.shape[0]
    cdef int num_near = n_d.shape[1]
    cdef np.ndarray[double, ndim=1] fit_pa = np.zeros(n, dtype=np.float64)
    cdef int i

    cdef double* n_d_ptr = &n_d[0, 0]
    cdef int* n_index_ptr = &n_index[0, 0]
    cdef double* mass_ptr = &mass[0]
    cdef double* hsm_ptr = &hsm[0]

    for i in prange(n, nogil=True):
        fit_pa[i] = calc_sph_density(i, num_near, n_d_ptr, n_index_ptr, mass_ptr, hsm_ptr)

    return fit_pa