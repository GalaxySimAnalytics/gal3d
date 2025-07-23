# distutils: language=c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
from libc.math cimport pow as c_pow, log, abs, sqrt
from cython.parallel import prange
import cython

from ...util.array_operate import unit_vector3d, vector_length3d
from gal3d import config

np.import_array()

ctypedef np.float64_t DTYPE_t


cdef extern from "ellipsoid_s.hpp":
    void f_shaped_ellipsoid_cpp(
        double a, double b, double c, double Sa, double Sb, double Sc,
        const double* pos, int n, double* result, int num_threads) nogil

def f_shaped_ellipsoid(double a, double b, double c, double Sa, double Sb, double Sc,
                             np.ndarray[DTYPE_t, ndim=2] pos):
    cdef int n = pos.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros(n, dtype=np.float64)
    cdef int num_threads = config['general']['number_of_threads']
    with nogil:
        f_shaped_ellipsoid_cpp(a, b, c, Sa, Sb, Sc,
                               &pos[0,0], n,
                               &result[0], num_threads)
    return result

cdef extern from "ellipsoid_s.hpp":
    void f_shaped_ellipsoid_jacobian_cpp(
        double a, double b, double c, double Sa, double Sb, double Sc,
        const double* pos, int n,
        double* da, double* db, double* dc,
        double* dSa, double* dSb, double* dSc,
        double* dx, double* dy, double* dz,
        int num_threads) nogil

def f_shaped_ellipsoid_jacobian(double a, double b, double c, double Sa, double Sb, double Sc,
                               np.ndarray[DTYPE_t, ndim=2] pos):
    cdef int n = pos.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] da = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] db = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] dc = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] dSa = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] dSb = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] dSc = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] dx = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] dy = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] dz = np.zeros(n, dtype=np.float64)
    cdef int num_threads = config['general']['number_of_threads']

    with nogil:
        f_shaped_ellipsoid_jacobian_cpp(a, b, c, Sa, Sb, Sc,
                                          &pos[0, 0], n,
                                          &da[0], &db[0], &dc[0],
                                          &dSa[0], &dSb[0], &dSc[0],
                                          &dx[0], &dy[0], &dz[0],
                                          num_threads)

    return (da, db, dc, dSa, dSb, dSc, dx, dy, dz)


cdef extern from "ellipsoid_s.hpp":
    void IntersectRaysEllipsoid_S_cpp(
        double a, double b, double c, double Sa, double Sb, double Sc,
        const double* pos, int n, int maxIterations,
        double* tarpos, double* result, int num_threads) nogil


def IntersectRaysEllipsoid_S(double a, double b, double c, double Sa, double Sb, double Sc,
                             np.ndarray[DTYPE_t, ndim=2] pos, int maxIterations):
    cdef int n = pos.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] tarpos = np.zeros((n, 3), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros(n, dtype=np.float64)
    cdef int num_threads = config['general']['number_of_threads']
    with nogil:
        IntersectRaysEllipsoid_S_cpp(a, b, c, Sa, Sb, Sc,
                                 &pos[0,0], n, maxIterations,
                                 &tarpos[0,0], &result[0], num_threads)
    return tarpos, result




cdef extern from "ellipsoid_s.hpp":
    void f_ray_shaped_ellipsoid_cpp(
        double a, double b, double c, double Sa, double Sb, double Sc,
        const double* pos, int n, int maxIterations,
        double* result, int num_threads) nogil

def f_ray_shaped_ellipsoid(double a, double b, double c, double Sa, double Sb, double Sc,
                                  np.ndarray[DTYPE_t, ndim=2] pos, int maxIterations):
    cdef int n = pos.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros(n, dtype=np.float64)
    cdef int num_threads = config['general']['number_of_threads']
    with nogil:
        f_ray_shaped_ellipsoid_cpp(a, b, c, Sa, Sb, Sc,
                               &pos[0,0], n, maxIterations,
                               &result[0], num_threads)
    return result

cdef extern from "ellipsoid_s.hpp":
    void IntersectLinesEllipsoid_S_cpp(
        double a, double b, double c, double Sa, double Sb, double Sc,
        const double* pos1, const double* pos2, int n, int maxIterations,
        double* ts, int num_threads) nogil

def IntersectLinesEllipsoid_S(double a, double b, double c, double Sa, double Sb, double Sc,
                             np.ndarray[DTYPE_t, ndim=2] pos1, 
                             np.ndarray[DTYPE_t, ndim=2] pos2,
                             int maxIteration):
    cdef int n = pos1.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] ts = np.zeros((n, 2), dtype=np.float64)
    cdef int num_threads = config['general']['number_of_threads']
    with nogil:
        IntersectLinesEllipsoid_S_cpp(a, b, c, Sa, Sb, Sc,
                                      &pos1[0,0], &pos2[0,0], n, maxIteration,
                                      &ts[0,0], num_threads)
    return ts