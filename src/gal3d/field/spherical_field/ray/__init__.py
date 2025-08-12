import logging
from collections.abc import Callable
from typing import Any, overload

import numpy as np

from .lu_mono_cy import LU_Mono

logger = logging.getLogger("gal3d.field.ray.MonotonRay")


def judge_monoton(x: np.ndarray, is_decreasing: bool = True) -> bool:
    """
    Judge whether the array `x` is monotonically decreasing or increasing.

    Parameters
    ----------
    x : array_like
        The input array to be checked for monotonicity.
    is_decreasing : bool, optional
        If True, checks for monotonically decreasing. If False, checks for monotonically increasing.
        Default is True.

    Returns
    -------
    bool
        True if `x` is monotonic (either decreasing or increasing based on `is_decreasing`), False otherwise.
    """

    if is_decreasing:
        judge_mono = all(np.diff(x) < 0)
    else:
        judge_mono = all(np.diff(x) > 0)

    return judge_mono

class MonotonRay:
    _interpolator_method: dict[str, Callable | type] = {}

    def __init__(
        self,
        r: np.ndarray,
        f: np.ndarray,
        is_decreasing: bool = True,
        interpolator_method: str = "LU",
        interpolator_kwargs: dict | None = None,
        **kwargs: Any,
    ):
        """
        Parameters
        ----------
        r : 1-D array
            A 1-D array of monotonically increasing real values.

        f : 1-D array
            A 1-D array of real values, must be of the same length as `r`.

        is_decreasing : bool, optional
            If True, the function f(r) is assumed to be monotonically decreasing. Default is True.

        interpolator_method : {'LU'}, optional
            Determines the method used to smooth f(r).
            - 'LU': Uses the median of lower and upper values.
            Default is 'LU'.

        interpolator_kwargs : dict, optional
            Additional keyword arguments to pass to the interpolator. Default is an empty dictionary.

        smoothlog : bool, optional
            If True, smooth f in log scale. Default is False.

        mono_de : bool, optional
            If True, the function f(r) is assumed to be monotonically decreasing. Default is True.

        extrapolate : bool, optional
            If True, extrapolate to out-of-bounds points based on the first and last intervals. Default is True.

        throw_point : bool, optional
            If True, when using 'SG' smoothing, some bad points will be thrown out. Default is True.

        Notes
        -----
        - If using `smooth_mode='LU'`, the function first calculates (r_upper, f_upper) and (r_lower, f_lower),
        then interpolates them. The median value `f_median = (f_upper + f_lower)/2` is used as the smoothed
        function, and the error at each point `r` can be obtained from `f_lower` and `f_upper`.

        Raises
        ------
        ValueError
            If `r` is not a strictly increasing sequence.
        """

        # r must be increasing
        if interpolator_kwargs is None:
            interpolator_kwargs = {}
        if not judge_monoton(r, is_decreasing=False):
            raise ValueError("'r' must be strictly increasing sequence.")
        interpolator = self._interpolator_method[interpolator_method]

        self._interpolator: LU_Mono = interpolator(r, f, is_decreasing=is_decreasing, **interpolator_kwargs)

    @overload
    def __call__(self, value: float, inv: bool = False) -> float: ...
    @overload
    def __call__(self, value: np.ndarray, inv: bool = False) -> np.ndarray: ...
    def __call__(self, value: float | np.ndarray, inv: bool =False) -> float | np.ndarray:
        """
        Evaluate the interpolated function at a given value.

        Parameters
        ----------
        value : float or array-like
            The value(s) at which to evaluate the function.
        inv : bool, optional
            If True, evaluate the inverse function r(f). Default is False.

        Returns
        -------
        float or array-like
            The interpolated value(s) of f(r) or r(f).
        """
        if inv:
            return self._interpolator.inv_f_value(value)
        return self._interpolator.f_value(value)

    @overload
    def lower(self, value: float, inv: bool = False) -> float: ...
    @overload
    def lower(self, value: np.ndarray, inv: bool = False) -> np.ndarray: ...
    def lower(self, value: float | np.ndarray, inv: bool =False) -> float | np.ndarray:
        """
        Evaluate the lower bound of the interpolated function at a given value.

        Parameters
        ----------
        value : float or array-like
            The value(s) at which to evaluate the lower bound.
        inv : bool, optional
            If True, evaluate the lower bound of the inverse function r(f). Default is False.

        Returns
        -------
        float or array-like
            The lower bound value(s) of f(r) or r(f).
        """
        if inv:
            return self._interpolator.inv_f_lower(value)
        return self._interpolator.f_lower(value)

    @overload
    def upper(self, value: float, inv: bool = False) -> float: ...
    @overload
    def upper(self, value: np.ndarray, inv: bool = False) -> np.ndarray: ...
    def upper(self, value: float | np.ndarray, inv: bool =False) -> float | np.ndarray:
        """
        Evaluate the upper bound of the interpolated function at a given value.

        Parameters
        ----------
        value : float or array-like
            The value(s) at which to evaluate the upper bound.
        inv : bool, optional
            If True, evaluate the upper bound of the inverse function r(f). Default is False.

        Returns
        -------
        float or array-like
            The upper bound value(s) of f(r) or r(f).
        """
        if inv:
            return self._interpolator.inv_f_upper(value)
        return self._interpolator.f_upper(value)

    @staticmethod
    def interpolator_registry(fn: str | Callable[..., Any] | type) -> Callable[..., Any] | type:
        if callable(fn):
            MonotonRay._interpolator_method[fn.__name__] = fn
            return fn

        fn_name = fn

        def decorator(fn: Callable) -> Callable:
            if callable(fn):
                MonotonRay._interpolator_method[fn_name] = fn
                return fn
            raise TypeError(f"try register {fn} as {fn_name}, but {fn} is not callable")

        return decorator



MonotonRay.interpolator_registry("LU")(LU_Mono)
