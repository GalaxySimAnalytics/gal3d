import math

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

from .minimize_func import MinimizeFunc


@MinimizeFunc.fn_registry
@jit(
    float64(float64[:]),
    nogil=True,
    parallel=True,
    fastmath=True,
    cache=True,
)
def sums_dev(
    f_call: np.float64 | np.ndarray,
):
    h = 0.0
    for i in prange(len(f_call)):
        h = h + f_call[i] * f_call[i]
    h = h / len(f_call)
    return h


@MinimizeFunc.fn_registry
@jit(
    float64(float64[:], float64[:]),
    nogil=True,
    parallel=True,
    fastmath=True,
    cache=True,
)
def sums_dev_byw(f_call: np.float64 | np.ndarray, w: np.float64 | np.ndarray):
    h = 0.0
    for i in prange(len(f_call)):
        h = h + f_call[i] * f_call[i] * w[i]
    h = h / len(f_call)
    return h


@MinimizeFunc.fn_registry
@jit(
    float64(float64[:], float64[:]),
    nogil=True,
    parallel=True,
    fastmath=True,
    cache=True,
)
def sums_dev_rscale(f_call: np.float64 | np.ndarray, r: np.float64 | np.ndarray):
    h = 0.0
    for i in prange(len(f_call)):
        h = h + (f_call[i] * r[i]) * (f_call[i] * r[i])
    h = h / len(f_call)
    return h


@MinimizeFunc.fn_registry
@jit(
    float64(float64[:], float64[:], float64[:]),
    nogil=True,
    parallel=True,
    fastmath=True,
    cache=True,
)
def sums_dev_rscale_byw(
    f_call: np.float64 | np.ndarray,
    r: np.float64 | np.ndarray,
    w: np.float64 | np.ndarray,
):
    h = 0.0
    for i in prange(len(f_call)):
        h = h + (f_call[i] * r[i]) * (f_call[i] * r[i]) * w[i]
    h = h / len(f_call)
    return h
