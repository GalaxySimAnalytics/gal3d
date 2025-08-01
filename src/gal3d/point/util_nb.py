"""
Module for computing centers, inertia tensors, and principal axes using Numba-accelerated routines.

Functions
---------
- centroid(pos): Returns the geometric center of positions.
- center_of_mass(pos, mass): Returns the mass-weighted center of mass.
- shrink_sphere_center(...): Iteratively computes the center using the shrinking sphere method.
- moment_of_inertia(pos, mass): Computes the inertia tensor.
- abc_vect(pos, mass): Returns eigenvalues and eigenvectors of the inertia tensor (i.e., principal axes).
"""

import logging
import math

import numpy as np
from numba import (
    boolean,
    cuda,
    deferred_type,
    float64,
    int32,
    int64,
    jit,
    njit,
    optional,
    prange,
    types,
)


@njit(float64[:](float64[:, :]), nogil=True, parallel=True, fastmath=True, cache=True)
def centroid(pos):
    """
    Compute the geometric center of positions.

    Parameters
    ----------
    pos : ndarray of shape (N, 3)
        3D positions of particles.

    Returns
    -------
    cenpos : ndarray of shape (3,)
        Geometric center position.
    """
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
    """
    Iteratively estimate the center of mass using the shrink-sphere method.

    Parameters
    ----------
    pos : ndarray of shape (N, 3)
        3D positions of particles.
    weight : ndarray of shape (N,)
        Weights of the particles, typically their masses.
    min_points : int
        Minimum number of particles required to continue iterations.
    particles_for_second_radius : int
        Number of particles used to record the second-radius value.
    shrink_factor : float
        Factor by which the sphere radius shrinks in each iteration.
    starting_rmax : float
        Initial radius to start the shrinking process.
    itermax : int
        Maximum number of iterations allowed.

    Returns
    -------
    com_x : ndarray of shape (3,)
        Final estimated center of mass.
    current_rmax : float
        Final radius used in the last iteration.
    second_radius : float
        Radius at which particle count dropped below `particles_for_second_radius`.
    iternum : int
        Number of iterations performed.
    """

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
    """
    Compute the center of mass using mass-weighted positions.

    Parameters
    ----------
    pos : ndarray of shape (N, 3)
        3D positions of particles.
    mass : ndarray of shape (N,)
        Masses of the particles.

    Returns
    -------
    cenpos : ndarray of shape (3,)
        Center of mass position.
    """
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
    """
    Compute the moment of inertia tensor.

    Parameters
    ----------
    pos : ndarray of shape (N, 3)
        3D positions of particles.
    mass : ndarray of shape (N,)
        Masses of the particles.

    Returns
    -------
    I : ndarray of shape (3, 3)
        Moment of inertia tensor.
    """
    return np.array(
        [[np.sum(m * pos[:, i] * pos[:, j]) for j in prange(3)] for i in prange(3)]
    ) / np.sum(m)


@njit(types.Tuple((float64[:], float64[:, :]))(float64[:, :], float64[:]))
def abc_vect(pos, mass):
    """
    Compute the principal axes and corresponding eigenvalues of the inertia tensor.

    Parameters
    ----------
    pos : ndarray of shape (N, 3)
        3D positions of particles.
    mass : ndarray of shape (N,)
        Masses of the particles.

    Returns
    -------
    abc : ndarray of shape (3,)
        Eigenvalues of the inertia tensor (sorted descendingly).
    rotation_matrix : ndarray of shape (3, 3)
        Corresponding eigenvectors as a rotation matrix.
    """
    D = np.linalg.eigh(moment_of_inertia(pos, mass))
    align = np.argsort(D[0])[::-1]
    abc = D[0][align]
    axis = np.eye(3)

    return abc, np.dot(
        D[1].T[align], axis
    )  # make np.eye(3), to abc, Rotate(np.eye(3),avc_vect.T)
