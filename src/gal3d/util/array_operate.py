from numba import (
    int32,
    deferred_type,
    optional,
    float64,
    boolean,
    int64,
    njit,
    jit,
    prange,
    types,
    cuda,
)

import numpy as np
import math

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
    'RobustLength3d',
]


@jit(float64(float64, float64), fastmath=True, cache=True)
def RobustLength2d(v0, v1):
    '''avoiding floating-point overflow that could occur normally when computing'''
    v0 = v0 if v0 > 0 else -v0
    v1 = v1 if v1 > 0 else -v1
    if v0 > v1:
        return v0 * math.sqrt(1 + v1 * v1 / v0 / v0)
    else:
        return v1 * math.sqrt(1 + v0 * v0 / v1 / v1)


@jit(float64(float64, float64, float64), fastmath=True, cache=True)
def RobustLength3d(v0, v1, v2):
    '''avoiding floating-point overflow that could occur normally when computing'''
    v0 = v0 if v0 > 0 else -v0
    v1 = v1 if v1 > 0 else -v1
    v2 = v2 if v2 > 0 else -v2
    if (v0 > v1) and (v0 > v2):
        return v0 * math.sqrt(1 + v1 * v1 / v0 / v0 + v2 * v2 / v0 / v0)
    elif (v2 > v0) and (v2 > v1):
        return v2 * math.sqrt(1 + v1 * v1 / v2 / v2 + v0 * v0 / v2 / v2)
    else:
        return v1 * math.sqrt(1 + v0 * v0 / v1 / v1 + v2 * v2 / v1 / v1)


#@jit(
#    float64[:, :](float64[:, :], float64[:]),
#    nogil=True,
#    parallel=True,
#    fastmath=True,
#    cache=True,
#)
#def Shift(pos, cen):
#    Np = len(pos)
#    for i in prange(Np):
#        
#        pos[i] = pos[i] - cen
#    return pos

def Shift(pos,cen):
    return pos - cen

@jit(
    float64[:, :](float64[:, :], float64[:, :]),
    nogil=True,
    parallel=True,
    fastmath=True,
    cache=True,
)
def Matmul(v1, v2):
    '''nxm * mxc = nxc'''
    n = len(v1)
    p = len(v1[0])
    m = len(v2[0])
    C = np.zeros((n, m), np.float64)
    for i in prange(n):
        for k in prange(p):
            s = v1[i, k]
            for j in prange(m):
                C[i, j] += s * v2[k, j]
    return C


@jit(
    float64[:, :](float64[:, :], float64[:, :]),
    nogil=True,
    parallel=True,
    fastmath=True,
    cache=True,
)
def Hadamard(v1, v2):
    '''nxm * nxm = nxm'''
    n = len(v1)
    m = len(v1[0])
    C = np.zeros((n, m))
    for i in prange(n):
        for j in prange(m):
            C[i, j] = v1[i, j] * v2[i, j]
    return C


@jit(float64(float64[:], float64[:]), fastmath=True, parallel=True, cache=True)
def Dot(v0, v1):
    '''n * n = 1'''
    dot = v0[0] * v1[0]
    for i in prange(1, len(v0)):
        dot += v0[i] * v1[i]
    return dot


#@jit(
#    float64[:, :](float64[:, :], float64[:, :]),
#    nogil=True,
#    fastmath=True,
#    cache=True,
#)
#def Rotate(pos, mat):
#    return Matmul(mat, pos.T).T

def Rotate(pos,mat):
    return np.matmul(pos,mat.T)



@jit(
    float64[:](float64[:, :]),
    nogil=True,
    parallel=True,
    fastmath=True,
    cache=True,
)
def vector_length3d(pos):
    pos_2 = pos[:, 0] ** 2 + pos[:, 1] ** 2 + pos[:, 2] ** 2
    return np.sqrt(pos_2)


@jit(
    float64[:, :](float64[:, :]),
    nogil=True,
    parallel=True,
    fastmath=True,
    cache=True,
)
def unit_vector3d(pos):
    r = vector_length3d(pos)
    return (pos.T / r).T


@jit(
    float64[:, :](float64[:, :]),
    nogil=True,
    parallel=True,
    fastmath=True,
    cache=True,
)
def trans_to_Spherical_coordinates(pos_data):
    '''
    input: pos_data shape: Nx3  [x,y,z]

    output: spherical_data shape: Nx3  [r,theta,phi]
        theta: 0~pi , phi: 0-2*pi
    '''
    nump = len(pos_data)
    epsilon = 1e-10
    sphere_data = np.zeros((nump, 3))
    r = vector_length3d(pos_data)
    phi = np.arctan2(pos_data.T[0], pos_data.T[1])
    for i in prange(nump):
        if phi[i] < 0:
            phi[i] = phi[i] + math.pi * 2
    r[r < epsilon] = epsilon
    theta = np.arccos(pos_data.T[2] / r)
    theta[np.isnan(theta)] = 0
    sphere_data[:, 0] = r
    sphere_data[:, 1] = theta
    sphere_data[:, 2] = phi
    return sphere_data


@jit(
    float64[:, :](float64[:, :]),
    nogil=True,
    parallel=True,
    fastmath=True,
    cache=True,
)
def trans_to_Cartesian_coordinates(sphere_coor):
    '''
    input: spherical_data shape: Nx3  [r,theta,phi]
        theta: 0~pi , phi: 0-2*pi

    output: pos_data shape: Nx3  [x,y,z]

    '''
    pos_data = np.zeros((len(sphere_coor), 3))
    pos_data[:, 0] = (
        sphere_coor[:, 0] * np.sin(sphere_coor[:, 1]) * np.cos(sphere_coor[:, 2])
    )
    pos_data[:, 1] = (
        sphere_coor[:, 0] * np.sin(sphere_coor[:, 1]) * np.sin(sphere_coor[:, 2])
    )
    pos_data[:, 2] = sphere_coor[:, 0] * np.cos(sphere_coor[:, 1])
    return pos_data
