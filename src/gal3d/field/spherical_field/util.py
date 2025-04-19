import math

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
)
import numpy as np


from ...point.util import abc_vect
from ...util.array_operate import (
    vector_length3d,
    unit_vector3d,
    trans_to_Spherical_coordinates,
    trans_to_Cartesian_coordinates,
    Matmul,
    Dot,
    Rotate,
)


__all__ = [
    'trans_to_Spherical_coordinates',
    'trans_to_Cartesian_coordinates',
    'fibonacci_sampling',
    'vector_length3d',
    'unit_vector3d',
    'Matmul',
]


@jit(
    types.Tuple((float64[:, :], float64[:, :]))(int64),
    nogil=True,
    parallel=True,
    fastmath=True,
    cache=True,
)
def fibonacci_sampling(Num_sampling: int = 256):
    '''
    Parameters:
        Num_sampling: int, default 256,
            the numble of points

    Return: [x,y,z],[r,phi,theta]
    '''
    Golden_Ratio = (math.sqrt(5) + 1) / 2 - 1
    sampling_pos = np.zeros((Num_sampling, 3))

    for n in prange(Num_sampling):
        i = n + 1
        sampling_pos[n][0] = math.sqrt(
            1 - ((2 * i - 1) / Num_sampling - 1) ** 2
        ) * math.cos(2 * math.pi * i * Golden_Ratio)
        sampling_pos[n][1] = math.sqrt(
            1 - ((2 * i - 1) / Num_sampling - 1) ** 2
        ) * math.sin(2 * math.pi * i * Golden_Ratio)
        sampling_pos[n][2] = (2 * i - 1) / Num_sampling - 1
    sampling_sphere_coor = trans_to_Spherical_coordinates(sampling_pos)
    return sampling_pos, sampling_sphere_coor


@njit(
    float64[:](float64[:, :], float64[:, :], float64, float64),
    fastmath=True,
    parallel=True,
    nogil=True,
)
def iso_profile_by_moi(points, pas, res_b, res_c):

    c_cos_max = res_c
    c_cos_min = -res_c

    b_cos_max = math.sqrt(1 - res_b**2)
    b_cos_min = -b_cos_max

    iso_pro_pa = np.zeros(len(pas[0]), dtype=np.float64)

    for i in prange(len(pas[0])):
        abc, rota = abc_vect(points, pas[:, i])
        new_pos = Rotate(points, rota.T)
        sel1 = (c_cos_min <= new_pos[:, 2]) & (new_pos[:, 2] <= c_cos_max)
        sel2 = (b_cos_max <= new_pos[:, 0]) | (new_pos[:, 0] <= b_cos_min)
        iso_pro_pa[i] = np.mean(pas[:, i][(sel1) & (sel2)])

    return iso_pro_pa


@njit(
    float64[:](float64[:, :], float64[:, :], float64, float64),
    fastmath=True,
    parallel=True,
    nogil=True,
)
def iso_profile_by_pair(points, pas, res_b, res_c):

    angle_max = math.sqrt(1 - (res_b / 2 + res_c / 2) ** 2)

    points_dist = Matmul(points, points.T)
    sel = (points_dist < -angle_max) | (points_dist > angle_max)

    iso_pro_pa = np.zeros(len(pas[0]), dtype=np.float64)
    for i in prange(len(pas[0])):
        pathis = np.zeros(len(sel))
        for j in prange(len(sel)):
            pathis[j] = np.mean(pas[:, i][sel[j]])
        iso_pro_pa[i] = np.max(pathis)
    return iso_pro_pa
