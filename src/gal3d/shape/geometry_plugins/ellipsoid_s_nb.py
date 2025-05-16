
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

from ..geometry import GeometryBase, Parameters
from ...util.array_operate import unit_vector3d, RobustLength3d, vector_length3d

__all__ = ['Ellipsoid_S']




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


class Ellipsoid_S(GeometryBase):

    # Using a tuple instead of a set to maintain order and allow duplicates if needed.
    PN = ('a', 'eps_ab', 'eps_bc', 'sa', 'sb', 'sc')
    LB = {'a': 0.1, 'eps_ab': 0.01, 'eps_bc': 0.01, 'sa': 0.2, 'sb': 0.2, 'sc': 0.2}
    UB = {'a': np.inf, 'eps_ab': 0.99, 'eps_bc': 0.99, 'sa': 2, 'sb': 2, 'sc': 2}

    MaxIterationDist = 100
    MaxIterationLine = 100
    def __init__(self, *args, **kwargs):
        """
        Initializes the Ellipsoid_S instance with given parameters.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the initializer (currently unused).
        **kwargs : dict
            A dictionary of parameters to initialize the ellipsoid.
        """
        self.parameters = self.init_parameters(**kwargs)
        self.parameters = self.init_parameters(**kwargs)

    @staticmethod
    def init_parameters(**kwargs):
        """
        Initializes and returns the parameters with derived values.

        Parameters
        ----------
        **kwargs : dict
            A dictionary of parameters to initialize the ellipsoid.

        Returns
        -------
        Parameters
            An instance of Parameters with initialized and derived values.
        """
        param = Parameters(**kwargs)
        param._derived['eps_ab'] = lambda d: 1.0 - d['b'] / d['a']
        param._derived['eps_bc'] = lambda d: 1.0 - d['c'] / d['b']
        param._derived['eps_ac'] = lambda d: 1.0 - d['c'] / d['a']
        param._derived['b'] = lambda d: (
            d['a'] * (1 - d['eps_ab']) if 'eps_ab' in d else d['c'] / (1 - d['eps_bc'])
        )
        param._derived['c'] = lambda d: (
            d['b'] * (1 - d['eps_bc']) if 'eps_bc' in d else d['a'] * (1 - d['eps_ac'])
        )
        param._derived['a'] = lambda d: (
            d['b'] / (1 - d['eps_ab']) if 'eps_ab' in d else d['c'] / (1 - d['eps_ac'])
        )

        parameters = Parameters(**{i: param[i] for i in Ellipsoid_S.PN})
        parameters._derived.update(param._derived)
        parameters.set_lb(**Ellipsoid_S.LB)
        parameters.set_ub(**Ellipsoid_S.UB)
        return parameters

    @staticmethod
    def get_parameters():
        """
        Returns a default set of parameters for the ellipsoid.

        Returns
        -------
        Parameters
            An instance of Parameters with default values.
        """
        return Ellipsoid_S.init_parameters(
            a=3.0, eps_ab=0.2, eps_bc=0.5, sa=1.0, sb=1.0, sc=1.0
        )

    def __call__(self, pos):
        """
        Evaluates the ellipsoid function at the given positions.

        Parameters
        ----------
        pos : array_like
            An array of positions where the ellipsoid function is evaluated.

        Returns
        -------
        array_like
            The evaluated values of the ellipsoid function at the given positions.
        """
        pos = self._check_pos(pos)
        return f_shaped_ellipsoid(
            self['a'], self['b'], self['c'], self['sa'], self['sb'], self['sc'], pos
        )

    def jacobian(self, pos: ArrayLike) -> tuple:
        """
        Computes the Jacobian of the ellipsoid function at the given positions.

        Parameters
        ----------
        pos : array_like
            An array of positions where the Jacobian is computed.

        Returns
        -------
        tuple
            The computed Jacobian values at the given positions.
        """
        pos = self._check_pos(pos)
        return f_shaped_ellipsoid_jacobian(
            self['a'], self['b'], self['c'], self['sa'], self['sb'], self['sc'], pos
        )

    def ray_intersect(self, pos: ArrayLike) -> tuple:
        """
        Computes the distance between points and ray points on the ellipsoid.

        Parameters
        ----------
        pos : array_like
            An array of positions where the distance is computed.

        Returns
        -------
        array_like
            The computed distances between points and ray points on the ellipsoid.
        """
        pos = self._check_pos(pos)
        return IntersectRaysEllipsoid_S(
            self['a'],
            self['b'],
            self['c'],
            self['sa'],
            self['sb'],
            self['sc'],
            pos,
            Ellipsoid_S.MaxIterationDist,
        )

    def line_intersect(self, pos1: ArrayLike, pos2: ArrayLike) -> np.ndarray:

        pos1 = self._check_pos(pos1)
        pos2 = self._check_pos(pos2)

        return IntersectLinesEllipsoid_S(
            self['a'],
            self['b'],
            self['c'],
            self['sa'],
            self['sb'],
            self['sc'],
            pos1,
            pos2,
            Ellipsoid_S.MaxIterationLine,
        )

    def f_ray_d(self, pos: ArrayLike) -> np.ndarray:
        """
        pos : array_like
            An array of positions where the derivative is evaluated.

        Returns
        -------
        array_like
            The evaluated distance values at the given positions.
        """
        pos = self._check_pos(pos)
        return f_ray_shaped_ellipsoid(
            self['a'],
            self['b'],
            self['c'],
            self['sa'],
            self['sb'],
            self['sc'],
            pos,
            Ellipsoid_S.MaxIterationDist,
        )

    @staticmethod
    def quick_call(a, eps_ab, eps_bc, sa, sb, sc, pos) -> np.ndarray:
        """
        Quickly evaluates the ellipsoid function with the given parameters and positions.

        This method is a simplified and faster version of the `__call__` method, designed for use in scenarios
        where performance is critical, such as optimization or error function evaluations. It computes the 
        ellipsoid function values directly using the provided parameters without relying on the instance's 
        internal state.

        Parameters
        ----------
        a : float
            The semi-major axis of the ellipsoid.
        eps_ab : float
            The eccentricity between the 'a' and 'b' axes.
        eps_bc : float
            The eccentricity between the 'b' and 'c' axes.
        sa : float
            Scaling factor along the 'a' axis.
        sb : float
            Scaling factor along the 'b' axis.
        sc : float
            Scaling factor along the 'c' axis.
        pos : array_like
            An array of positions where the ellipsoid function is evaluated.

        Returns
        -------
        np.ndarray
            The evaluated values of the ellipsoid function at the given positions.
        """
        b = a * (1 - eps_ab)
        c = b * (1 - eps_bc)
        return f_shaped_ellipsoid(a, b, c, sa, sb, sc, pos)

    @staticmethod
    def quick_f_ray_d(a, eps_ab, eps_bc, sa, sb, sc, pos) -> np.ndarray:
        """Quickly evaluates the distance fraction of the geometry function with given parameters and positions, useful in error function"""

        b = a * (1 - eps_ab)
        c = b * (1 - eps_bc)
        return f_ray_shaped_ellipsoid(
            a, b, c, sa, sb, sc, pos, Ellipsoid_S.MaxIterationDist
        )

    @staticmethod
    def quick_ray_dist(a, eps_ab, eps_bc, sa, sb, sc, pos) -> np.ndarray:
        """
        Quickly computes the distance between points and ray points on the ellipsoid.

        This method calculates the distance fraction of the geometry function using the provided parameters
        and positions, without relying on the instance's internal state.

        Parameters
        ----------
        a : float
            The semi-major axis of the ellipsoid.
        eps_ab : float
            The eccentricity between the 'a' and 'b' axes.
        eps_bc : float
            The eccentricity between the 'b' and 'c' axes.
        sa : float
            Scaling factor along the 'a' axis.
        sb : float
            Scaling factor along the 'b' axis.
        sc : float
            Scaling factor along the 'c' axis.
        pos : array_like
            An array of positions where the distance is computed.

        Returns
        -------
        np.ndarray
            The computed distances between points and ray points on the ellipsoid.
        """

        b = a * (1 - eps_ab)
        c = b * (1 - eps_bc)
        return IntersectRaysEllipsoid_S(
            float(a), b, c, sa, sb, sc, pos, Ellipsoid_S.MaxIterationDist
        )[1]

    @staticmethod
    def quick_line_intersect(a, eps_ab, eps_bc, sa, sb, sc, pos1, pos2) -> np.ndarray:
        b = a * (1 - eps_ab)
        c = b * (1 - eps_bc)
        return IntersectLinesEllipsoid_S(
            float(a),
            float(b),
            float(c),
            float(sa),
            float(sb),
            float(sc),
            pos1,
            pos2,
            Ellipsoid_S.MaxIterationLine,
        )

    @staticmethod
    def quick_jacobian(a, b, c, sa, sb, sc, pos) -> tuple:
        """
        Quickly computes the Jacobian of the ellipsoid function at the given positions.

        This method calculates the Jacobian using the provided parameters and positions
        without relying on the instance's internal state.

        Parameters
        ----------
        a : float
            The semi-major axis of the ellipsoid.
        b : float
            The semi-minor axis of the ellipsoid along the 'b' axis.
        c : float
            The semi-minor axis of the ellipsoid along the 'c' axis.
        sa : float
            Scaling factor along the 'a' axis.
        sb : float
            Scaling factor along the 'b' axis.
        sc : float
            Scaling factor along the 'c' axis.
        pos : array_like
            An array of positions where the Jacobian is computed.

        Returns
        -------
        tuple
            The computed Jacobian values at the given positions.
        """
        return f_shaped_ellipsoid_jacobian(
            float(a), float(b), float(c), float(sa), float(sb), float(sc), pos
        )
    
    @staticmethod
    def _check_pos(pos):
        """
        Ensure pos is a 2D numpy array of shape (N, 3).
        """
        pos = np.asarray(pos, dtype=np.float64)
        if pos.ndim == 1:
            pos = pos[np.newaxis, :]
        if pos.shape[1] != 3:
            raise ValueError("Input position must have shape (N, 3)")
        return pos
