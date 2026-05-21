# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np

cimport numpy as np
from libc.math cimport M_PI, acos, atan2, cos, isnan, sin, sqrt

from cython.parallel import prange

cimport cython

from gal3d.configuration import config

#openmp.omp_set_num_threads(num_threads)

np.import_array()

__all__ = [
    'Shift',
    'Rotate',
    'Matmul',
    'Hadamard',
    'Dot',
    'vector_length3d',
    'unit_vector3d',
    'trans_to_Spherical_coordinates',
    'trans_to_Cartesian_coordinates',
    'RobustLength2d',
    'RotateAndShift'
]

ctypedef fused DTYPE_t:
    np.float32_t
    np.float64_t

cdef int get_num_threads():
    return config.general.number_of_threads

@cython.cdivision(True)
cpdef double RobustLength2d(double v0, double v1):
    '''Avoiding floating-point overflow that could occur normally when computing'''
    v0 = v0 if v0 > 0 else -v0
    v1 = v1 if v1 > 0 else -v1
    if v0 > v1:
        return v0 * sqrt(1 + v1 * v1 / (v0 * v0))
    else:
        return v1 * sqrt(1 + v0 * v0 / (v1 * v1))


@cython.cdivision(True)
cdef inline double RobustLength3d(double v0, double v1, double v2) nogil:
    '''Avoiding floating-point overflow that could occur normally when computing'''
    cdef double av0 = abs(v0)
    cdef double av1 = abs(v1)
    cdef double av2 = abs(v2)
    if (av0 > av1) and (av0 > av2):
        return av0 * sqrt(1 + av1 * av1 / (av0 * av0) + av2 * av2 / (av0 * av0))
    elif (av2 > av0) and (av2 > av1):
        return av2 * sqrt(1 + av1 * av1 / (av2 * av2) + av0 * av0 / (av2 * av2))
    else:
        return av1 * sqrt(1 + av0 * av0 / (av1 * av1) + av2 * av2 / (av1 * av1))


def Shift(np.ndarray[DTYPE_t, ndim=2] pos, np.ndarray[DTYPE_t, ndim=1] cen):
    '''Shift positions by a center vector'''
    return pos - cen


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def Matmul(np.ndarray[DTYPE_t, ndim=2] v1, np.ndarray[DTYPE_t, ndim=2] v2):
    '''Matrix multiplication: nxm * mxc = nxc'''
    cdef:
        int n = v1.shape[0]
        int p = v1.shape[1]
        int m = v2.shape[1]
        int i, j, k
        double s
        np.ndarray[DTYPE_t, ndim=2] C = np.zeros((n, m), dtype=v1.dtype)
    cdef int num_threads = get_num_threads()
    for i in prange(n, nogil=True, num_threads=num_threads):
        for k in range(p):
            s = v1[i, k]
            for j in range(m):
                C[i, j] += s * v2[k, j]
    
    return C


@cython.boundscheck(False)
@cython.wraparound(False)
def Hadamard(np.ndarray[DTYPE_t, ndim=2] v1, np.ndarray[DTYPE_t, ndim=2] v2):
    '''Element-wise multiplication: nxm * nxm = nxm'''
    cdef:
        int n = v1.shape[0]
        int m = v1.shape[1]
        int i, j
        np.ndarray[DTYPE_t, ndim=2] C = np.zeros((n, m), dtype=v1.dtype)
    cdef int num_threads = get_num_threads()
    for i in prange(n, nogil=True, num_threads=num_threads):
        for j in range(m):
            C[i, j] = v1[i, j] * v2[i, j]
    
    return C


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double Dot(np.ndarray[DTYPE_t, ndim=1] v0, np.ndarray[DTYPE_t, ndim=1] v1):
    '''Dot product: n * n = 1'''
    cdef:
        int i, n = v0.shape[0]
        double dot = 0.0
    
    for i in range(n):
        dot += v0[i] * v1[i]
    
    return dot


def Rotate(np.ndarray[DTYPE_t, ndim=2] pos, np.ndarray[DTYPE_t, ndim=2] mat):
    '''Rotate positions using rotation matrix'''
    return np.matmul(pos, mat.T)

@cython.boundscheck(False)
@cython.wraparound(False)
def RotateAndShift(np.ndarray[DTYPE_t, ndim=2] pos, np.ndarray[DTYPE_t, ndim=2] matrix, 
                  np.ndarray[DTYPE_t, ndim=1] pc):
    """Perform rotation followed by translation in one operation."""
    cdef:
        int i, j, n = pos.shape[0] 
        np.ndarray[DTYPE_t, ndim=2] result = np.zeros((n, 3), dtype=pos.dtype)
    cdef int num_threads = get_num_threads()
    # Use num_threads parameter to control thread count
    for i in prange(n, nogil=True, num_threads=num_threads, schedule='static'):
        for j in range(3):
            # Inline the dot product calculation to avoid reduction variable
            result[i, j] = (pos[i, 0] * matrix[j, 0] + 
                           pos[i, 1] * matrix[j, 1] + 
                           pos[i, 2] * matrix[j, 2]) - pc[j]
    
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def vector_length3d(np.ndarray[DTYPE_t, ndim=2] pos):
    """Calculate lengths of multiple 3D vectors"""
    cdef:
        int i, n = pos.shape[0]
        np.ndarray[DTYPE_t, ndim=1] result = np.zeros(n, dtype=pos.dtype)
    cdef int num_threads = get_num_threads()
    # Use chunking for better cache utilization
    for i in prange(n, nogil=True, num_threads=num_threads, schedule='static'):
        result[i] = sqrt(pos[i, 0] * pos[i, 0] + pos[i, 1] * pos[i, 1] + pos[i, 2] * pos[i, 2])
    
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def unit_vector3d(np.ndarray[DTYPE_t, ndim=2] pos):
    """Normalize multiple 3D vectors"""
    cdef:
        int i, j, n = pos.shape[0]
        double r
        np.ndarray[DTYPE_t, ndim=2] result = np.zeros((n, 3), dtype=pos.dtype)
    cdef int num_threads = get_num_threads()
    for i in prange(n, nogil=True, num_threads=num_threads):
        r = sqrt(pos[i, 0] * pos[i, 0] + pos[i, 1] * pos[i, 1] + pos[i, 2] * pos[i, 2])
        if r > 0:
            for j in range(3):
                result[i, j] = pos[i, j] / r
    
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def trans_to_Spherical_coordinates(np.ndarray[DTYPE_t, ndim=2] pos_data):
    '''Convert Cartesian coordinates to spherical coordinates'''
    cdef:
        int i, nump = pos_data.shape[0]
        double epsilon = 1e-10
        np.ndarray[DTYPE_t, ndim=2] sphere_data = np.zeros((nump, 3), dtype=pos_data.dtype)
        np.ndarray[DTYPE_t, ndim=1] lengths = vector_length3d(pos_data)
    cdef int num_threads = get_num_threads()
    # Process everything in a single parallel loop
    for i in prange(nump, nogil=True, num_threads=num_threads):
        # Calculate r and ensure not too small
        sphere_data[i, 0] = lengths[i] if lengths[i] >= epsilon else epsilon
        
        # Calculate theta, avoiding NaN
        sphere_data[i, 1] = acos(pos_data[i, 2] / sphere_data[i, 0])
        if isnan(sphere_data[i, 1]):
            sphere_data[i, 1] = 0
        
        # Calculate phi
        sphere_data[i, 2] = atan2(pos_data[i, 0], pos_data[i, 1])
        if sphere_data[i, 2] < 0:
            sphere_data[i, 2] += 2.0 * M_PI
    
    return sphere_data


@cython.boundscheck(False)
@cython.wraparound(False)
def trans_to_Cartesian_coordinates(np.ndarray[DTYPE_t, ndim=2] sphere_coor):
    '''
    Convert spherical coordinates to Cartesian coordinates.

    Parameters
    ----------
    sphere_coor : ndarray of shape (N, 3)
        Input spherical coordinates arranged as ``[r, theta, phi]`` for each row.
        ``theta`` is expected to lie in ``[0, pi]`` and ``phi`` in ``[0, 2*pi)``.

    Returns
    -------
    pos_data : ndarray of shape (N, 3)
        Cartesian coordinates arranged as ``[x, y, z]`` for each row.
    '''
    cdef:
        int i, nump = sphere_coor.shape[0]
        np.ndarray[DTYPE_t, ndim=2] pos_data = np.zeros((nump, 3), dtype=sphere_coor.dtype)
    cdef int num_threads = get_num_threads()
    for i in prange(nump, num_threads=num_threads, nogil=True):
        pos_data[i, 0] = sphere_coor[i, 0] * sin(sphere_coor[i, 1]) * cos(sphere_coor[i, 2])
        pos_data[i, 1] = sphere_coor[i, 0] * sin(sphere_coor[i, 1]) * sin(sphere_coor[i, 2])
        pos_data[i, 2] = sphere_coor[i, 0] * cos(sphere_coor[i, 1])
    
    return pos_data