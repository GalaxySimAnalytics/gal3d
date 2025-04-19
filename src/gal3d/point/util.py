import math
import logging

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


@njit(float64[:](float64[:, :]), nogil=True, parallel=True, fastmath=True, cache=True)
def centroid(pos):
    cenpos = np.zeros(3, dtype=np.float64)
    for i in prange(3):
        cenpos[i] = np.mean(pos[:, i])
    return cenpos


# from https://github.com/pynbody/pynbody/blob/master/pynbody/analysis/_com.pyx, this is a numba version
@njit(
    types.Tuple((float64[:], float64, float64, int32))(
        float64[:, :], float64[:], int32, int32, float64, float64, int32
    ),
    nogil=True,
    parallel=True,
    fastmath=True,
    cache=True,
)
def shrink_sphere_center(
    pos,
    weight,
    min_points,
    particles_for_second_radius,
    shrink_factor,
    starting_rmax,
    itermax,
):

    npart = len(pos)
    npart_all = len(pos)
    r2 = 0.0
    tot_weight = 0.0
    com = np.zeros(3)

    com_x = centroid(pos)

    iternum = 0
    current_rmax = np.inf

    second_radius = -1.0

    while True:
        offset_x = 0.0
        offset_y = 0.0
        offset_z = 0.0
        cx = com_x[0]
        cy = com_x[1]
        cz = com_x[2]
        current_rmax2 = current_rmax * current_rmax
        npart = 0
        for i in prange(npart_all):
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
            second_radius = math.sqrt(current_rmax2)

        if npart < min_points:
            break

        # divide out total mass and shift
        com[0] = cx + offset_x / tot_weight
        com[1] = cy + offset_y / tot_weight
        com[2] = cz + offset_z / tot_weight

        # update for next cycle
        com_x[:] = com
        com[:] = 0.0
        tot_weight = 0.0

        iternum += 1
        if iternum > 1:
            current_rmax *= shrink_factor
        else:
            current_rmax = starting_rmax

        if iternum > itermax:
            break

    return com_x, current_rmax, second_radius, iternum


@njit(
    float64[:](float64[:, :], float64[:]),
    nogil=True,
    parallel=True,
    fastmath=True,
    cache=True,
)
def center_of_mass(pos, mass):
    cenpos = np.zeros(3, dtype=np.float64)
    massum = np.sum(mass)
    for i in prange(3):
        cenpos[i] = np.sum(pos[:, i] * mass) / massum
    return cenpos


@njit(
    float64[:, :](float64[:, :], float64[:]),
    fastmath=True,
    parallel=True,
    nogil=True,
)
def moment_of_inertia(pos, m):
    return np.array(
        [[np.sum(m * pos[:, i] * pos[:, j]) for j in prange(3)] for i in prange(3)]
    ) / np.sum(m)


@njit(types.Tuple((float64[:], float64[:, :]))(float64[:, :], float64[:]))
def abc_vect(pos, pa):
    D = np.linalg.eigh(moment_of_inertia(pos, pa))
    align = np.argsort(D[0])[::-1]
    abc = D[0][align]
    axis = np.eye(3)

    return abc, np.dot(
        D[1].T[align], axis
    )  # make np.eye(3), to abc, Rotate(np.eye(3),avc_vect.T)
