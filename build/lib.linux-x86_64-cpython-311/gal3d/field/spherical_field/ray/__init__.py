import logging
from typing import Callable

import numpy as np
from scipy.signal import savgol_filter

from .monotonic_profile import SG_Mono, LU_Mono, judge_monoton


logger = logging.getLogger('gal3d.preprocessing.ray.ray_profile')


class MonotonRay:
    _interpolator_method = {'LU': LU_Mono, 'SG': SG_Mono}

    def __init__(
        self,
        r,
        f,
        f_de=True,
        interpolator_method: str = 'LU',
        interpolator_kwargs: dict = dict(),
        **kwargs,
    ):
        '''
        Interpolate f(r) and r(f) using specified methods.

        Parameters
        ----------
        r : 1-D array
            A 1-D array of monotonically increasing real values.
        f : 1-D array
            A 1-D array of real values, must be of the same length as `r`.
        f_de : bool, optional
            If True, the function f(r) is assumed to be monotonically decreasing. Default is True.
        interpolator_method : {'LU', 'SG'}, optional
            Determines the method used to smooth f(r).
            - 'LU': Uses the median of lower and upper values.
            - 'SG': Uses the Savitzky-Golay filter.
            Default is 'LU'.
        interpolator_kwargs : dict, optional
            Additional keyword arguments to pass to the interpolator. Default is an empty dictionary.
        **kwargs : dict, optional
            Additional keyword arguments:
            - smoothlog : bool, default is False
                If True, smooth f in log scale.
            - mono_de : bool, default is True
                If True, the function f(r) is assumed to be monotonically decreasing.
            - extrapolate : bool, default is True
                If True, extrapolate to out-of-bounds points based on the first and last intervals.
                If False, return NaNs for out-of-bounds points.
            - throw_point : bool, default is True
                If True, when using 'SG' smoothing, some bad points will be thrown out.

        Notes
        -----
        - If using `smooth_mode='LU'`, the function first calculates (r_upper, f_upper) and (r_lower, f_lower),
          then interpolates them. The median value `f_median = (f_upper + f_lower)/2` is used as the smoothed
          function, and the error at each point `r` can be obtained from `f_lower` and `f_upper`.
        - If using `smooth_mode='SG'`, the function first applies the Savitzky-Golay filter to make `f` monotonic,
          then interpolates. The smoothed function `f_smooth` is used, and the error at each point `r` can be
          obtained from the ratio `f/f_smooth`.

        Raises
        ------
        ValueError
            If `r` is not a strictly increasing sequence.
        '''

        # r must be increasing
        if not judge_monoton(r, mono_de=False):
            raise ValueError(f"'r' must be strictly increasing sequence.")
        interpolator = self._interpolator_method[interpolator_method]

        self._interpolator = interpolator(r, f, y_de=f_de, **interpolator_kwargs)

    def __call__(self, value, inv=False):
        '''
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
        '''
        if inv:
            return self._interpolator.inv_f_value(value)
        return self._interpolator.f_value(value)

    def lower(self, value, inv=False):
        '''
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
        '''
        if inv:
            return self._interpolator.inv_f_lower(value)
        return self._interpolator.f_lower(value)

    def upper(self, value, inv=False):
        '''
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
        '''
        if inv:
            return self._interpolator.inv_f_upper(value)
        return self._interpolator.f_upper(value)

    @staticmethod
    def interpolator_registry(fn: str | Callable) -> Callable:
        if callable(fn):
            MonotonRay._interpolator_method[fn] = fn
            return fn

        fn_name = fn

        def decorator(fn: Callable) -> Callable:
            if callable(fn):
                MonotonRay._interpolator_method[fn_name] = fn
                return fn
            raise TypeError(f"try register {fn} as {fn_name}, but {fn} is not callable")

        return decorator


from .monotonic_profile import SG_Mono, LU_Mono

MonotonRay.interpolator_registry('SG')(SG_Mono)
MonotonRay.interpolator_registry('LU')(LU_Mono)
