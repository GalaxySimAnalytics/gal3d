import logging


import numpy as np


from .lu_mono import MyPchipInterpolator

from .util import (
    savgol_filter,
    sg_smooth,
    sg_smooth_tomono,
    judge_monoton,
    sg_smooth_throw,
    #MyPchipInterpolator,
    MyAkima1DInterpolator,
    resample_1D,
)
from ....util.func_signature import func_optional_key, update_dict_value


logger = logging.getLogger("gal3d.field.ray.monotonic_profile")

__all__ = ['SG_Mono', 'LU_Mono']


def inverse_interpolate(y, x, y_de, interpolate, extrapolate=False):
    """
    Perform inverse interpolation on the given data.

    Parameters
    ----------
    y : array-like
        The dependent variable values.
    x : array-like
        The independent variable values.
    y_de : bool
        If True, the y values are in decreasing order; otherwise, they are in increasing order.
    interpolate : callable
        The interpolation function to use.
    extrapolate : bool, optional
        If True, allow extrapolation beyond the range of the data. Default is False.

    Returns
    -------
    callable
        A function that performs the inverse interpolation.
    """
    if y_de:
        y_temp = y[::-1]
        x_temp = x[::-1]
    else:
        y_temp = y
        x_temp = x
    sel = np.diff(y_temp, prepend=-np.inf) > 0

    return interpolate(y_temp[sel], x_temp[sel], extrapolate=extrapolate)


class SG_Mono:
    """
    A class for smoothing and interpolating profiles using Savitzky-Golay filter.

    Parameters
    ----------
    x : array-like
        The independent variable values.
    y : array-like
        The dependent variable values.
    smooth_log : bool, optional
        If True, apply smoothing in log space. Default is True.
    window_length_max_frac : float, optional
        The maximum fraction of the data length to use as the window length for smoothing. Default is 0.1.
    y_de : bool, optional
        If True, the y values are in decreasing order; otherwise, they are in increasing order. Default is True.
    throw : bool, optional
        If True, throw away points that do not meet the monotonicity condition. Default is True.
    interpolate_mode : str, optional
        The interpolation mode to use. Options are 'Pchip' and 'Akima'. Default is 'Pchip'.
    polyorder : int, optional
        The order of the polynomial used in the Savitzky-Golay filter. Default is 1.
    mode : str, optional
        The mode parameter for the Savitzky-Golay filter. Default is 'nearest'.
    **kwargs : dict
        Additional keyword arguments passed to the Savitzky-Golay filter.

    Attributes
    ----------
    f_value : callable
        The main interpolation function.
    inv_f : callable
        The inverse interpolation function.
    f_lower : callable
        The lower bound interpolation function.
    f_upper : callable
        The upper bound interpolation function.
    inv_f_value : callable
        The inverse interpolation function for the main values.
    inv_f_lower : callable
        The inverse interpolation function for the lower bound.
    inv_f_upper : callable
        The inverse interpolation function for the upper bound.
    """

    savgol_filter_options = func_optional_key(savgol_filter)

    def __init__(
        self,
        x,
        y,
        smooth_log=True,
        window_length_max_frac=0.1,
        y_de=True,
        throw=True,
        interpolate_mode='Pchip',
        polyorder: int = 1,
        mode: str = 'nearest',
        **kwargs,
    ):
        INTERPOLATE = {'Pchip': MyPchipInterpolator, 'Akima': MyAkima1DInterpolator}
        interpolate = INTERPOLATE[interpolate_mode]

        if throw:
            y_sm, ind = sg_smooth_throw(
                y,
                smooth_log=smooth_log,
                window_length_max=min(int(window_length_max_frac * len(x)), len(x) - 1),
                mono_de=y_de,
                polyorder=polyorder,
                mode=mode,
            )
            x = x[ind]
            y = y[ind]
        else:
            y_sm = sg_smooth_tomono(y, smooth_log, polyorder, mode, y_de)

        self.f_value = interpolate(x, y_sm, extrapolate=False)

        if y_de:
            self.inv_f = interpolate(y_sm[::-1], x[::-1], extrapolate=False)
        else:
            self.inv_f = interpolate(y_sm, x, extrapolate=False)

        vars = np.abs(y_sm / y - 1)
        var = np.percentile(vars, 84)
        self.f_lower = interpolate(x, y_sm * (1 - var), extrapolate=False)
        self.f_upper = interpolate(x, y_sm * (1 + var), extrapolate=False)

        self.inv_f_value = inverse_interpolate(
            y_sm, x, y_de, interpolate, extrapolate=False
        )
        self.inv_f_lower = inverse_interpolate(
            y_sm * (1 - var), x, y_de, interpolate, extrapolate=False
        )
        self.inv_f_upper = inverse_interpolate(
            y_sm * (1 + var), x, y_de, interpolate, extrapolate=False
        )


class LU_Mono:
    """
    A class for smoothing and interpolating profiles using lower and upper bounds.

    Parameters
    ----------
    x : array-like
        The independent variable values.
    y : array-like
        The dependent variable values.
    is_decreasing : bool, optional
        If True, the y values are in decreasing order; otherwise, they are in increasing order. Default is True.
    interpolate_mode : str, optional
        The interpolation mode to use. Options are 'Pchip' and 'Akima'. Default is 'Pchip'.
    re_sample_ord : int, optional
        The order of resampling. If less than 1, no resampling is performed. Default is 0.

    Attributes
    ----------
    f_lower : callable
        The lower bound interpolation function.
    f_upper : callable
        The upper bound interpolation function.
    f_value : callable
        The main interpolation function.
    inv_f_value : callable
        The inverse interpolation function for the main values.
    inv_f_lower : callable
        The inverse interpolation function for the lower bound.
    inv_f_upper : callable
        The inverse interpolation function for the upper bound.
    """

    def __init__(self, x, y, is_decreasing=True, interpolate_mode='Pchip', re_sample_ord=0):
        INTERPOLATE = {'Pchip': MyPchipInterpolator, 'Akima': MyAkima1DInterpolator}
        interpolate = INTERPOLATE[interpolate_mode]

        # Use boundary calculation
        lo, up = self.profile_boundary(y, is_decreasing=is_decreasing)
        
        # Create interpolators with pre-filtered data
        x_lo, y_lo = x[lo], y[lo]
        x_up, y_up = x[up], y[up]
        fupper = interpolate(x_up, y_up, extrapolate='const')
        flower = interpolate(x_lo, y_lo, extrapolate='const')

        # Calculate shared nodes
        rnodes = x[lo & up]
        
        # Pre-compute interpolated values
        upper_vals = fupper(x)
        lower_vals = flower(x)
        midpoints = (upper_vals + lower_vals) / 2
        
        # Efficient binning
        bin_indices = np.searchsorted(rnodes, x, side='right')
        above_mid = y > midpoints
        
        # Count bins once
        rbins = np.bincount(bin_indices)
        
        # Use vectorized operations for counts
        max_bin = max(bin_indices.max() if len(bin_indices) else 0, 1)
        upper = np.bincount(bin_indices[above_mid], minlength=max_bin+1)
        lower = np.bincount(bin_indices[~above_mid], minlength=max_bin+1)

        # Process new x points
        new_x = x if re_sample_ord < 1 else resample_1D(x, re_sample_ord)
        new_x_ind = np.searchsorted(rnodes, new_x, side='right')
        
        # Pre-allocate array
        new_y_upper_ratio = np.zeros_like(new_x, dtype=float)
        
        # Handle valid indices only
        valid_indices = new_x_ind < len(rbins)
        safe_indices = new_x_ind[valid_indices]
        
        # Calculate ratios efficiently
        if np.sum(lower) > np.sum(upper):
            ratio_calc = 1 - ((1 + lower[safe_indices]) / (2 + rbins[safe_indices]))
        else:
            ratio_calc = (1 + upper[safe_indices]) / (2 + rbins[safe_indices])

        new_y_upper_ratio[valid_indices] = ratio_calc
        
        # Calculate interpolated values for new points
        new_upper = fupper(new_x)
        new_lower = flower(new_x)
        new_y = new_y_upper_ratio * new_upper + (1 - new_y_upper_ratio) * new_lower
        
        # Store the results
        self.f_lower = flower
        self.f_upper = fupper
        self.f_value = interpolate(new_x, new_y, extrapolate=False)
        
        # Compute inverse functions
        self.inv_f_value = inverse_interpolate(new_y, new_x, is_decreasing, interpolate, extrapolate=False)
        self.inv_f_lower = inverse_interpolate(y_lo, x_lo, is_decreasing, interpolate, extrapolate=False)
        self.inv_f_upper = inverse_interpolate(y_up, x_up, is_decreasing, interpolate, extrapolate=False)


    @staticmethod
    def profile_boundary(y, is_decreasing=True):
        """
        Determine the lower and upper boundaries of the profile.

        Parameters
        ----------
        y : array-like
            The dependent variable values.
        is_decreasing : bool, optional
            If True, the y values are in decreasing order; otherwise, they are in increasing order. Default is True.

        Returns
        -------
        tuple
            A tuple containing the lower and upper boundary indices.
        """

        def sel_lower(arr):
            sel = np.zeros(len(arr), dtype=bool)
            vamax = arr[0]
            for idx, va in enumerate(arr):
                if va <= vamax:
                    vamax = va
                    sel[idx] = True
            return sel

        def sel_upper(arr):
            sel = np.zeros(len(arr), dtype=bool)
            vamin = arr[-1]
            for idx, va in enumerate(arr[::-1]):
                if va >= vamin:
                    vamin = va
                    sel[idx] = True
            return sel[::-1]

        if is_decreasing:
            data = y
            upper = sel_upper(data)
            lower = sel_lower(data)
            return lower, upper

        data = y[::-1]
        upper = sel_upper(data)
        lower = sel_lower(data)
        return lower[::-1], upper[::-1]
