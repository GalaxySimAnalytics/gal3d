# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np

cimport numpy as np
from libc.math cimport abs, log
from libc.math cimport pow as c_pow
from libc.math cimport sqrt

import cython
from cython.parallel import prange

from gal3d.config import config

from ...util.array_operate import unit_vector3d, vector_length3d

np.import_array()

ctypedef np.float64_t DTYPE_t


cdef int get_num_threads():
    return config.general.number_of_threads

cdef int get_ray_method():
    return config.ellipsoid_s.DistIteration.value

cdef int get_line_method():
    return config.ellipsoid_s.LineIteration.value

def check_contiguous(pos):
    if not pos.flags['C_CONTIGUOUS']:
        pos = np.ascontiguousarray(pos)
    return pos


cdef extern from "ellipsoid_s.h":
    void f_shaped_ellipsoid_cpp(
        double a, double b, double c, double Sa, double Sb, double Sc,
        const double* pos, int n, double* result, double* r, int num_threads) nogil

def f_shaped_ellipsoid(double a, double b, double c, double Sa, double Sb, double Sc,
                             np.ndarray[DTYPE_t, ndim=2] pos):
    pos = check_contiguous(pos)
    cdef int n = pos.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] r = np.zeros(n, dtype=np.float64)
    cdef int num_threads = get_num_threads()
    with nogil:
        f_shaped_ellipsoid_cpp(a, b, c, Sa, Sb, Sc,
                               &pos[0,0], n,
                               &result[0], &r[0], num_threads)
    return result, r

cdef extern from "ellipsoid_s.h":
    void f_shaped_ellipsoid_jacobian_cpp(
        double a, double b, double c, double Sa, double Sb, double Sc,
        const double* pos, int n,
        double* da, double* db, double* dc,
        double* dSa, double* dSb, double* dSc,
        double* dx, double* dy, double* dz,
        int num_threads) nogil

def f_shaped_ellipsoid_jacobian(double a, double b, double c, double Sa, double Sb, double Sc,
                               np.ndarray[DTYPE_t, ndim=2] pos):
    pos = check_contiguous(pos)
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
    cdef int num_threads = get_num_threads()

    with nogil:
        f_shaped_ellipsoid_jacobian_cpp(a, b, c, Sa, Sb, Sc,
                                          &pos[0, 0], n,
                                          &da[0], &db[0], &dc[0],
                                          &dSa[0], &dSb[0], &dSc[0],
                                          &dx[0], &dy[0], &dz[0],
                                          num_threads)

    return (da, db, dc, dSa, dSb, dSc, dx, dy, dz)


cdef extern from "ellipsoid_s.h":
    void IntersectRaysEllipsoid_S_cpp(
        double a, double b, double c, double Sa, double Sb, double Sc,
        const double* pos, int n, int maxIterations,
        double* tarpos, double* result, double* r, int method, int num_threads) nogil


def IntersectRaysEllipsoid_S(double a, double b, double c, double Sa, double Sb, double Sc,
                             np.ndarray[DTYPE_t, ndim=2] pos, int maxIterations):
    pos = check_contiguous(pos)
    cdef int n = pos.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] tarpos = np.zeros((n, 3), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] r = np.zeros(n, dtype=np.float64)
    cdef int num_threads = get_num_threads()
    cdef int method = get_ray_method()
    with nogil:
        IntersectRaysEllipsoid_S_cpp(a, b, c, Sa, Sb, Sc,
                                 &pos[0,0], n, maxIterations,
                                 &tarpos[0,0], &result[0], &r[0], method, num_threads)
    return tarpos, result, r




cdef extern from "ellipsoid_s.h":
    void f_ray_shaped_ellipsoid_cpp(
        double a, double b, double c, double Sa, double Sb, double Sc,
        const double* pos, int n, int maxIterations,
        double* result, double* r, int method, int num_threads) nogil

def f_ray_shaped_ellipsoid(double a, double b, double c, double Sa, double Sb, double Sc,
                                  np.ndarray[DTYPE_t, ndim=2] pos, int maxIterations):
    pos = check_contiguous(pos)
    cdef int n = pos.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] r = np.zeros(n, dtype=np.float64)
    cdef int num_threads = get_num_threads()
    cdef int method = get_ray_method()
    with nogil:
        f_ray_shaped_ellipsoid_cpp(a, b, c, Sa, Sb, Sc,
                               &pos[0,0], n, maxIterations,
                               &result[0], &r[0], method, num_threads)
    return result, r

cdef extern from "ellipsoid_s.h":
    void IntersectLinesEllipsoid_S_cpp(
        double a, double b, double c, double Sa, double Sb, double Sc,
        const double* pos1, const double* pos2, int n, int maxIterations,
        double* ts, int num_threads) nogil

def IntersectLinesEllipsoid_S(double a, double b, double c, double Sa, double Sb, double Sc,
                             np.ndarray[DTYPE_t, ndim=2] pos1, 
                             np.ndarray[DTYPE_t, ndim=2] pos2,
                             int maxIteration):
    pos1 = check_contiguous(pos1)
    pos2 = check_contiguous(pos2)
    cdef int n = pos1.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] ts = np.zeros((n, 2), dtype=np.float64)
    cdef int num_threads = get_num_threads()
    with nogil:
        IntersectLinesEllipsoid_S_cpp(a, b, c, Sa, Sb, Sc,
                                      &pos1[0,0], &pos2[0,0], n, maxIteration,
                                      &ts[0,0], num_threads)
    return ts