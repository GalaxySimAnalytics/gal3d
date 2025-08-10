# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange

from gal3d.config import config


cdef int get_num_threads():
    return config.general.number_of_threads

cdef extern from "sph_density.h":
    double calc_sph_density(
        int i, int num_near,
        const double* n_d,
        const int* n_index,
        const double* mass,
        const double* hsm
    ) nogil

    void calc_sph_gradient(
        int i, int num_near,
        const double* n_d,
        const int* n_index,
        const double* mass,
        const double* pos,
        const double* hsm,
        const double* target_pos,
        double* grad
    ) nogil


def sph_density(np.ndarray[double, ndim=2] n_d,
           np.ndarray[int, ndim=2] n_index,
           np.ndarray[double, ndim=1] mass,
           np.ndarray[double, ndim=1] hsm,) -> np.ndarray:

    cdef int n = n_d.shape[0]
    cdef int num_near = n_d.shape[1]
    cdef np.ndarray[double, ndim=1] density = np.zeros(n, dtype=np.float64)
    cdef int i

    cdef double* n_d_ptr = &n_d[0, 0]
    cdef int* n_index_ptr = &n_index[0, 0]
    cdef double* mass_ptr = &mass[0]
    cdef double* hsm_ptr = &hsm[0]
    cdef int num_threads = get_num_threads()
    for i in prange(n, nogil=True, num_threads=num_threads):
        density[i] = calc_sph_density(i, num_near, n_d_ptr, n_index_ptr, mass_ptr, hsm_ptr)

    return density


def sph_gradient(np.ndarray[double, ndim=2] n_d,
                 np.ndarray[int, ndim=2] n_index,
                 np.ndarray[double, ndim=1] mass,
                 np.ndarray[double, ndim=2] pos,
                 np.ndarray[double, ndim=1] hsm,
                 np.ndarray[double, ndim=2] target_pos) -> np.ndarray:

    cdef int n = n_d.shape[0]
    cdef int num_near = n_d.shape[1]
    cdef np.ndarray[double, ndim=2] gradient = np.zeros((n, 3), dtype=np.float64)
    cdef int i

    cdef double* n_d_ptr = &n_d[0, 0]
    cdef int* n_index_ptr = &n_index[0, 0]
    cdef double* mass_ptr = &mass[0]
    cdef double* pos_ptr = &pos[0, 0]
    cdef double* hsm_ptr = &hsm[0]
    cdef double* target_pos_ptr = &target_pos[0, 0]
    cdef double* gradient_ptr = &gradient[0, 0]

    cdef int num_threads = get_num_threads()
    for i in prange(n, nogil=True, num_threads=num_threads):
        calc_sph_gradient(i, num_near, n_d_ptr, n_index_ptr, mass_ptr, pos_ptr, hsm_ptr, target_pos_ptr, gradient_ptr)
    return gradient