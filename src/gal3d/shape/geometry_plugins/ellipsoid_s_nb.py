

import numpy as np
from numba import (
    boolean,
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

from ...util.array_operate import RobustLength3d, unit_vector3d, vector_length3d


@jit(
    float64[:](float64, float64, float64, float64, float64, float64, float64[:, :]),
    nogil=True,
    parallel=True,
    fastmath=True,
    cache=True,
)
def f_shaped_ellipsoid(a, b, c, Sa, Sb, Sc, pos):
    h1 = pos[:, 0] * pos[:, 0] / a / a
    h2 = pos[:, 1] * pos[:, 1] / b / b
    h3 = pos[:, 2] * pos[:, 2] / c / c
    return np.float_power(h1, Sa) + np.float_power(h2, Sb) + np.float_power(h3, Sc)


@jit(
    types.Tuple(
        (
            float64[:],
            float64[:],
            float64[:],
            float64[:],
            float64[:],
            float64[:],
            float64[:],
            float64[:],
            float64[:],
        )
    )(float64, float64, float64, float64, float64, float64, float64[:, :]),
    nogil=True,
    parallel=True,
    fastmath=True,
    cache=True,
)
def f_shaped_ellipsoid_jacobian(a, b, c, Sa, Sb, Sc, pos):
    cof0 = np.float_power(pos[:, 0] * pos[:, 0], Sa)
    cof1 = np.float_power(pos[:, 1] * pos[:, 1], Sb)
    cof2 = np.float_power(pos[:, 2] * pos[:, 2], Sc)
    dx = 2 * Sa * cof0 / pos[:, 0] / a ** (2 * Sa)
    dy = 2 * Sb * cof1 / pos[:, 1] / b ** (2 * Sb)
    dz = 2 * Sc * cof2 / pos[:, 2] / c ** (2 * Sc)
    da = -2 * Sa * cof0 / a ** (2 * Sa + 1)
    db = -2 * Sb * cof1 / b ** (2 * Sb + 1)
    dc = -2 * Sc * cof2 / c ** (2 * Sc + 1)
    dSa = 2 * cof0 * (np.log(np.abs(pos[:, 0])) - np.log(a)) / (a**2) ** Sa
    dSb = 2 * cof1 * (np.log(np.abs(pos[:, 1])) - np.log(b)) / (b**2) ** Sb
    dSc = 2 * cof2 * (np.log(np.abs(pos[:, 2])) - np.log(c)) / (c**2) ** Sc
    return (da, db, dc, dSa, dSb, dSc, dx, dy, dz)


@jit(
    float64[:, :](
        float64[:],
        float64[:],
        float64[:],
        float64[:],
        float64[:],
        float64[:],
        float64[:, :],
    ),
    nogil=True,
    parallel=True,
    fastmath=True,
    cache=True,
)
def f_shaped_ellipsoids(a, b, c, Sa, Sb, Sc, pos):
    res = np.zeros((len(a), len(pos)))
    for i in prange(len(a)):
        res[i] = f_shaped_ellipsoid(a[i], b[i], c[i], Sa[i], Sb[i], Sc[i], pos)
    return res


@jit(
    float64(float64, float64, float64, float64, float64, float64, float64),
    fastmath=True,
    cache=True,
)
def _iter_f_IntersectRayEllipsoid_S(d, Sa, Sb, Sc, Ex, Ey, Ez):
    # g= f*
    dd = d * d
    ExddSa = Ex * dd**Sa
    EyddSb = Ey * dd**Sb
    EzddSc = Ez * dd**Sc
    f = ExddSa + EyddSb + EzddSc - 1
    df = 2 * (Sa * ExddSa + Sb * EyddSb + Sc * EzddSc) / d
    return -f / df


@jit(
    types.Tuple((float64, float64, float64, float64, float64))(
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        int32,
    ),
    fastmath=True,
    cache=True,
)
def IntersectRayEllipsoid_S(a, b, c, Sa, Sb, Sc, x, y, z, maxIterations: int):
    # a >= b >= c > 0,
    L = RobustLength3d(x, y, z)  # x, y, z,
    xi = x / L
    yi = y / L
    zi = z / L
    # dmin = c, dmax = a
    Ex = ((xi / a) ** 2) ** Sa
    Ey = ((yi / b) ** 2) ** Sb
    Ez = ((zi / c) ** 2) ** Sc

    epsilon = 1e-9  # 1e-9 is ok ?
    i = 0
    d0 = (a + c) / 2
    while True:
        if i > maxIterations:
            break
        d1 = d0 + _iter_f_IntersectRayEllipsoid_S(d0, Sa, Sb, Sc, Ex, Ey, Ez)
        if abs(d1 - d0) < epsilon:
            break
        d0 = d1
        i = i + 1
    return d0 * xi, d0 * yi, d0 * zi, d0, L


@jit(
    types.Tuple((float64[:, :], float64[:]))(
        float64, float64, float64, float64, float64, float64, float64[:, :], int32
    ),
    nogil=True,
    parallel=True,
    fastmath=True,
    cache=True,
)
def IntersectRaysEllipsoid_S(a, b, c, Sa, Sb, Sc, pos, maxIterations):
    tarpos = np.zeros((len(pos), 3))
    d = np.zeros(len(pos))
    L = np.zeros(len(pos))
    for i in prange(len(pos)):
        tarpos[i, 0], tarpos[i, 1], tarpos[i, 2], d[i], L[i] = IntersectRayEllipsoid_S(
            a, b, c, Sa, Sb, Sc, pos[i, 0], pos[i, 1], pos[i, 2], maxIterations
        )
    return tarpos, L - d


@jit(
    float64[:](
        float64, float64, float64, float64, float64, float64, float64[:, :], int32
    ),
    nogil=True,
    parallel=True,
    fastmath=True,
    cache=True,
)
def f_ray_shaped_ellipsoid(a, b, c, Sa, Sb, Sc, pos, maxIterations: int):
    tarpos = np.zeros((len(pos), 3))
    d = np.zeros(len(pos))
    L = np.zeros(len(pos))
    r = np.zeros(len(pos))
    for i in prange(len(pos)):
        tarpos[i, 0], tarpos[i, 1], tarpos[i, 2], d[i], L[i] = IntersectRayEllipsoid_S(
            a, b, c, Sa, Sb, Sc, pos[i, 0], pos[i, 1], pos[i, 2], maxIterations
        )
    for i in prange(len(pos)):
        r[i] = L[i] / d[i]

    return r


@jit(
    types.Tuple((float64, float64))(
        float64[:],
        float64[:],
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        int32,
        float64,
        float64,
    ),
    fastmath=True,
    cache=True,
)
def _iter_IntersectLineEllipsoid_S(
    pos1, vect, t0, a, b, c, Sa, Sb, Sc, maxIteration, epsilon, delta_cut
):
    i = 0
    posi = pos1 + t0 * vect
    while True:
        Ex, Ey, Ez = (
            ((posi[0] / a) ** 2) ** Sa,
            ((posi[1] / b) ** 2) ** Sb,
            ((posi[2] / c) ** 2) ** Sc,
        )
        f = Ex + Ey + Ez - 1

        if abs(f) < epsilon:  # this position is already at the target positon
            break

        df_x = Ex * Sa * vect[0] / posi[0] if posi[0] != 0 else 0.0
        df_y = Ey * Sb * vect[1] / posi[1] if posi[1] != 0 else 0.0
        df_z = Ez * Sc * vect[2] / posi[2] if posi[2] != 0 else 0.0
        df = 4 * (df_x + df_y + df_z)

        delta = -f / df
        if f < 2.:                          # when near target pos
            delta = min(delta_cut, delta)  # avoid large update 
            delta = max(-delta_cut, delta)
        t0 = t0 + delta
        posi = pos1 + t0 * vect
        i = i + 1

        if abs(delta) < epsilon or i > maxIteration:
            break
    Ex, Ey, Ez = (
        ((posi[0] / a) ** 2) ** Sa,
        ((posi[1] / b) ** 2) ** Sb,
        ((posi[2] / c) ** 2) ** Sc,
    )
    f = Ex + Ey + Ez - 1
    return t0, f


@jit(
    float64[:](
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64[:],
        float64[:],
        float64,
        int32,
    ),
    fastmath=True,
)
def IntersectLineEllipsoid_S(a, b, c, Sa, Sb, Sc, pos1, vect, tmax, maxIteration):

    delta_cut = c/2.
    ts = -np.ones(2)
    epsilon = 1e-9  # 1e-9 is ok ?

    # From both ends of the line segment
    t0, f0 = _iter_IntersectLineEllipsoid_S(
        pos1, vect, epsilon, a, b, c, Sa, Sb, Sc, maxIteration, epsilon, delta_cut
    )
    t1, f1 = _iter_IntersectLineEllipsoid_S(
        pos1, vect, tmax, a, b, c, Sa, Sb, Sc, maxIteration, epsilon, delta_cut
    )

    if (
        abs(t0 - t1) <= 10 * epsilon
    ):  # converge at the same positon, maybe one zero position, or no zero position
        if (abs(f0) >= 10 * epsilon) and (abs(f1) >= 10 * epsilon):
            return ts
        else:
            ts[0] = t0
            ts[1] = t1
            return ts

    if (abs(f0) <= 10 * epsilon) and (
        abs(f1) <= 10 * epsilon
    ):  # converge at two positons, two zero positions
        ts[0] = t0
        ts[1] = t1
        return ts

    if (abs(f0) <= 10 * epsilon) or (
        abs(f1) <= 10 * epsilon
    ):  # one is not at the target position, will expore t0 -> t1
        ti_min = t0
        ti_max = t1
        delta_cut = c/4.
        while True:
            if abs(f0) > 10 * epsilon:
                t2, f2 = _iter_IntersectLineEllipsoid_S(
                    pos1,
                    vect,
                    ti_min + (ti_max - ti_min) / 3,
                    a,
                    b,
                    c,
                    Sa,
                    Sb,
                    Sc,
                    maxIteration,
                    epsilon,
                    delta_cut,
                )
            else:
                t2, f2 = _iter_IntersectLineEllipsoid_S(
                    pos1,
                    vect,
                    ti_min + 2 * (ti_max - ti_min) / 3,
                    a,
                    b,
                    c,
                    Sa,
                    Sb,
                    Sc,
                    maxIteration,
                    epsilon,
                    delta_cut,
                )

            if (
                (abs(t2 - ti_min) >= 10 * epsilon)
                and (abs(t2 - ti_max) >= 10 * epsilon)
                and (abs(f2) <= 10 * epsilon)
            ):
                break

            if abs(f0) > 10 * epsilon:
                ti_min = ti_min + (ti_max - ti_min) / 3
            else:
                ti_max = ti_min + 2 * (ti_max - ti_min) / 3

            if abs(ti_min - ti_max) < 10 * epsilon:
                break

        if abs(f0) <= 10 * epsilon:
            ts[0] = t0
            ts[1] = t2
            return ts
        ts[0] = t2
        ts[1] = t1
        return ts

    return ts


@jit(
    float64[:, :](
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64[:, :],
        float64[:, :],
        int32,
    ),
    nogil=True,
    parallel=True,
    fastmath=True,
    cache=True,
)
def IntersectLinesEllipsoid_S(a, b, c, Sa, Sb, Sc, pos1, pos2, maxIteration):
    vects = unit_vector3d(pos2 - pos1)
    tmax = vector_length3d(pos2 - pos1)

    ts = np.ones((len(pos1), 2), dtype=np.float64)
    for i in prange(len(ts)):
        ts[i] = IntersectLineEllipsoid_S(
            a, b, c, Sa, Sb, Sc, pos1[i], vects[i], tmax[i], maxIteration
        )

    return ts


