
import math

import numpy as np
from numpy.typing import ArrayLike
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

from ...util.array_operate_nb import unit_vector3d, RobustLength3d, Dot

__all__ = ['Ellipsoid']





# -------------------- Utility functions  --------------------
@jit(
    float64[:](float64, float64, float64, float64[:, :]),
    nogil=True,
    parallel=True,
    fastmath=True,
    cache=True,
)
def f_ellipsoid(a: float, b: float, c: float, pos):
    return (
        pos[:, 0] * pos[:, 0] / a / a
        + pos[:, 1] * pos[:, 1] / b / b
        + pos[:, 2] * pos[:, 2] / c / c
    )


@jit(
    types.Tuple(
        (float64[:], float64[:], float64[:], float64[:], float64[:], float64[:])
    )(float64, float64, float64, float64[:, :]),
    nogil=True,
    parallel=True,
    fastmath=True,
    cache=True,
)
def f_ellipsoid_jacobian(a: float, b: float, c: float, pos):
    '''
    jacobian of ellipsoid, d/da,d/db,d/dc,d/dx,d/dy,d/dz
    '''
    dx = 2 * pos[:, 0] / a / a
    dy = 2 * pos[:, 1] / b / b
    dz = 2 * pos[:, 2] / c / c
    da = -dx * pos[:, 0] / a
    db = -dy * pos[:, 1] / b
    dc = -dz * pos[:, 2] / c
    return (da, db, dc, dx, dy, dz)




@jit(
    float64[:, :](float64[:], float64[:], float64[:], float64[:, :]),
    nogil=True,
    parallel=True,
    fastmath=True,
    cache=True,
)
def f_ellipsoids(a, b, c, pos):
    res = np.zeros((len(a), len(pos)))
    for i in prange(len(a)):
        res[i] = f_ellipsoid(a[i], b[i], c[i], pos)
    return res


@jit(float64(float64[:], boolean), cache=True)
def Length(v, robust: bool = False):
    if robust:
        maxAbsComp = abs(v[0])
        for i in range(1, len(v)):
            absComp = abs(v[i])
            if absComp > maxAbsComp:
                maxAbsComp = absComp
        if maxAbsComp > 0:
            scaled = v / maxAbsComp
            length = maxAbsComp * math.sqrt(Dot(scaled, scaled))
        else:
            length = 0.0
        return length
    else:
        return math.sqrt(Dot(v, v))


@jit(float64(float64[:]), cache=True)
def RobustLength(v):
    '''avoiding floating-point overflow that could occur normally when computing'''
    return Length(v, robust=True)


@jit(
    types.Tuple((float64, float64, float64, float64, float64))(
        float64, float64, float64, float64, float64, float64
    ),
    fastmath=True,
)
def IntersectRayEllipsoid(a, b, c, x, y, z):
    # a >= b >= c > 0,
    L = RobustLength3d(x, y, z)  # x, y, z,
    xi = x / L
    yi = y / L
    zi = z / L

    d = math.sqrt(1 / ((xi / a) ** 2 + (yi / b) ** 2 + (zi / c) ** 2))
    return xi * d, yi * d, zi * d, d, L


@jit(
    types.Tuple((float64[:, :], float64[:]))(float64, float64, float64, float64[:, :]),
    nogil=True,
    parallel=True,
    fastmath=True,
    cache=True,
)
def IntersectRaysEllipsoid(a, b, c, pos):

    n = pos.shape[0]
    tarpos = np.zeros((n, 3))
    d = np.zeros(n)
    L = np.zeros(n)
    for i in prange(n):
        x = pos[i, 0]
        y = pos[i, 1]
        z = pos[i, 2]
        Li = RobustLength3d(x, y, z)
        if Li>0:
            xi = x / Li
            yi = y / Li
            zi = z / Li
            denom = (xi / a) ** 2 + (yi / b) ** 2 + (zi / c) ** 2
            di = np.sqrt(1.0 / denom)
            tarpos[i, 0] = xi * di
            tarpos[i, 1] = yi * di
            tarpos[i, 2] = zi * di
            d[i] = di
            L[i] = Li
        else:
            # To avoid division by zero
            tarpos[i, 0] = 0.0
            tarpos[i, 1] = 0.0
            tarpos[i, 2] = 0.0
            d[i] = 0.0
            L[i] = 0.0


    return tarpos, L - d


@jit(
    float64[:](float64, float64, float64, float64[:, :]),
    nogil=True,
    parallel=True,
    fastmath=True,
    cache=True,
)
def f_ray_ellipsoid(a, b, c, pos):

    tarpos = np.zeros((len(pos), 3))
    d = np.zeros(len(pos))
    L = np.zeros(len(pos))
    r = np.zeros(len(pos))
    for i in prange(len(pos)):
        tarpos[i, 0], tarpos[i, 1], tarpos[i, 2], d[i], L[i] = IntersectRayEllipsoid(
            a, b, c, pos[i, 0], pos[i, 1], pos[i, 2]
        )
    for i in prange(len(pos)):
        r[i] = L[i] / d[i]
    return r


@jit(
    float64[:](float64, float64, float64, float64[:], float64[:]),
    fastmath=True,
)
def IntersectLineEllipsoid(a, b, c, pos1, vect):

    ts = -np.ones(2, dtype=np.float64)
    a2, b2, c2 = a**2, b**2, c**2
    ell11 = vect[0] ** 2 / a2 + vect[1] ** 2 / b2 + vect[2] ** 2 / c2
    ell01 = vect[0] * pos1[0] / a2 + vect[1] * pos1[1] / b2 + vect[2] * pos1[2] / c2
    ell00 = pos1[0] ** 2 / a2 + pos1[1] ** 2 / b2 + pos1[2] ** 2 / c2

    D = ell01**2 - ell11 * (ell00 - 1)
    if D > 0:
        ts[0] = (-ell01 - math.sqrt(D)) / ell11
        ts[1] = (-ell01 + math.sqrt(D)) / ell11
        return ts
    if D == 0:
        ts[0] = (-ell01) / ell11
        ts[1] = ts[0]
        return ts

    return ts


@jit(
    float64[:, :](float64, float64, float64, float64[:, :], float64[:, :]),
    nogil=True,
    parallel=True,
    fastmath=True,
    cache=True,
)
def IntersectLinesEllipsoid(a, b, c, pos1, pos2):
    vects = unit_vector3d(pos2 - pos1)
    ts = np.ones((len(pos1), 2), dtype=np.float64)

    for i in prange(len(ts)):
        ts[i] = IntersectLineEllipsoid(a, b, c, pos1[i], vects[i])

    return ts

