from libcpp.vector cimport vector
cimport numpy as np
import numpy as np


cdef extern from "pchip.hpp":
    cdef cppclass PchipInterpolator:
        PchipInterpolator(const vector[double]& x, const vector[double]& y)
        double interpolate(double xval, int nu) const
        vector[double] interpolate(const vector[double]& xvals, int nu) const
        double get_x_min() const
        double get_x_max() const

cdef vector[double] from_numpy_to_vector(np.ndarray[np.float64_t] arr):
    cdef vector[double] v
    cdef Py_ssize_t i, n = arr.shape[0]
    v.reserve(n)
    for i in range(n):
        v.push_back(arr[i])
    return v

cdef class MyPchipInterpolator:
    """
    A Cython wrapper for a C++ PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) interpolator.

    This class provides 1D monotonic interpolation with several extrapolation modes.
    It wraps a C++ implementation for efficient computation and supports both scalar and array inputs.

    Parameters
    ----------
    x : array-like
        The independent variable values (must be 1D and sorted).
    y : array-like
        The dependent variable values (must be 1D).
    extrapolate : {'const', 'periodic', True, False}, optional
        Extrapolation mode:
            - 'const': use constant boundary value for out-of-bounds input (default)
            - 'periodic': treat the data as periodic
            - True: same as 'periodic'
            - False: return NaN for out-of-bounds input

    Methods
    -------
    __call__(x, nu=0, extrapolate=None)
        Interpolate at given x (scalar or array). Optionally specify derivative order (nu) and extrapolation mode.

    Example
    -------
    >>> interp = MyPchipInterpolator([0, 1, 2], [1, 2, 0])
    >>> interp(1.5)
    1.0
    """
    cdef PchipInterpolator* cpp_interp

    cdef object extrapolate_mode

    def __cinit__(self, x, y, extrapolate ='const'):
        """
        Initialize the PCHIP interpolator.

        Parameters
        ----------
        x : array-like
            The independent variable values (must be 1D and sorted).
        y : array-like
            The dependent variable values (must be 1D).
        extrapolate : {'const', 'periodic', True, False}, optional
            Extrapolation mode:
                - 'const': use constant boundary value for out-of-bounds input (default)
                - 'periodic': treat the data as periodic
                - True: same as 'periodic'
                - False: return NaN for out-of-bounds input
        """

        x_arr = np.ascontiguousarray(np.asarray(x).ravel(), dtype=np.float64)
        y_arr = np.ascontiguousarray(np.asarray(y).ravel(), dtype=np.float64)
        cdef vector[double] vx = from_numpy_to_vector(x_arr)
        cdef vector[double] vy = from_numpy_to_vector(y_arr)
        self.cpp_interp = new PchipInterpolator(vx, vy)
        self.extrapolate_mode = extrapolate

    def __dealloc__(self):
        """
        Release the underlying C++ interpolator resources.
        """
        del self.cpp_interp

    def __call__(self, x, int nu=0, extrapolate=None):
        """
        Interpolate at the given x values.

        Parameters
        ----------
        x : float or array-like
            Points at which to interpolate.
        nu : int, optional
            Derivative order (default is 0).
        extrapolate : {'const', 'periodic', True, False}, optional
            Extrapolation mode to override the default.

        Returns
        -------
        float or np.ndarray
            Interpolated values at the input locations.
        """
        mode = extrapolate if extrapolate is not None else self.extrapolate_mode

        # 优先判断是否为标量
        if np.isscalar(x) or (np.shape(x) == ()):
            xval = float(x)
            x_min = self.cpp_interp.get_x_min()
            x_max = self.cpp_interp.get_x_max()
            if mode is True or mode == 'periodic':
                if mode == 'periodic':
                    xval = x_min + (xval - x_min) % (x_max - x_min)
                return self.cpp_interp.interpolate(xval, nu)
            elif mode == 'const':
                xval = min(max(xval, x_min), x_max)
                return self.cpp_interp.interpolate(xval, nu)
            elif mode is False:
                if xval < x_min or xval > x_max:
                    return float('nan')
                else:
                    return self.cpp_interp.interpolate(xval, nu)
            else:
                raise ValueError("Unknown extrapolate mode: %s" % mode)

        cdef vector[double] vx
        cdef vector[double] vy
        cdef np.ndarray result

        x_np = np.ascontiguousarray(np.asarray(x).ravel(), dtype=np.float64)
        x_min = self.cpp_interp.get_x_min()
        x_max = self.cpp_interp.get_x_max()

        if mode is True or mode == 'periodic':
            if mode == 'periodic':
                x_np = x_min + (x_np - x_min) % (x_max - x_min)
            vx = from_numpy_to_vector(x_np)
            vy = self.cpp_interp.interpolate(vx, nu)
            result = np.array(vy)
        elif mode == 'const':
            x_clip = np.clip(x_np, x_min, x_max)
            vx = from_numpy_to_vector(x_clip)
            vy = self.cpp_interp.interpolate(vx, nu)
            result = np.array(vy)
        elif mode is False:
            mask = (x_np >= x_min) & (x_np <= x_max)
            result = np.full(x_np.shape, np.nan)
            if np.any(mask):
                vx = from_numpy_to_vector(x_np[mask])
                vy = self.cpp_interp.interpolate(vx, nu)
                result[mask] = np.array(vy)
        else:
            raise ValueError("Unknown extrapolate mode: %s" % mode)

        # if input is scalar
        if np.isscalar(x) or (np.shape(x) == ()):
            return float(result[0])
        return result

cdef class LU_Mono:
    """
    A class for smoothing and interpolating 1D profiles using lower and upper monotonic bounds.

    This class constructs monotonic lower and upper envelopes for a given 1D profile,
    and provides smooth interpolation between them. It also supports resampling and
    inverse interpolation.

    Parameters
    ----------
    x : array-like
        The independent variable values.
    y : array-like
        The dependent variable values.
    is_decreasing : bool, optional
        If True, the y values are in decreasing order; otherwise, they are in increasing order. Default is True.
    resample_order : int, optional
        The order of resampling. If less than 1, no resampling is performed. Default is 0.

    Attributes
    ----------
    f_lower : MyPchipInterpolator
        The lower bound interpolation function.
    f_upper : MyPchipInterpolator
        The upper bound interpolation function.
    f_value : MyPchipInterpolator
        The main interpolation function.
    inv_f_value : MyPchipInterpolator
        The inverse interpolation function for the main values.
    inv_f_lower : MyPchipInterpolator
        The inverse interpolation function for the lower bound.
    inv_f_upper : MyPchipInterpolator
        The inverse interpolation function for the upper bound.
    """
    cdef MyPchipInterpolator interp_lower
    cdef MyPchipInterpolator interp_upper
    cdef MyPchipInterpolator interp_value
    cdef MyPchipInterpolator interp_value_inv
    cdef MyPchipInterpolator interp_lower_inv
    cdef MyPchipInterpolator interp_upper_inv

    def __init__(self, x, y, is_decreasing=True, resample_order=0, **kwargs):
        """
        Initialize the LU_Mono interpolator.

        Parameters
        ----------
        x : array-like
            The independent variable values.
        y : array-like
            The dependent variable values.
        is_decreasing : bool, optional
            If True, the y values are in decreasing order; otherwise, they are in increasing order. Default is True.
        resample_order : int, optional
            The order of resampling. If less than 1, no resampling is performed. Default is 0.
        """
        
        cdef np.ndarray[np.float64_t] x_arr = np.ascontiguousarray(np.asarray(x), dtype=np.float64)
        cdef np.ndarray[np.float64_t] y_arr = np.ascontiguousarray(np.asarray(y), dtype=np.float64)

        is_decreasing = <bint>is_decreasing
        resample_order = <int>resample_order

        # 只用 MyPchipInterpolator
        interpolate = MyPchipInterpolator

        # 边界索引
        lower_mask, upper_mask = LU_Mono.profile_boundary(y_arr, is_decreasing)

        x_lower = x_arr[lower_mask]
        y_lower = y_arr[lower_mask]
        x_upper = x_arr[upper_mask]
        y_upper = y_arr[upper_mask]

        if len(x_lower) < 2 or len(x_upper) < 2:
            raise ValueError("x_lower and x_upper must have at least 2 points for interpolation.")

        interp_upper = interpolate(x_upper, y_upper, extrapolate='const')
        interp_lower = interpolate(x_lower, y_lower, extrapolate='const')

        # 共享节点
        shared_nodes = x_arr[lower_mask & upper_mask]

        # 插值
        upper_values = interp_upper(x_arr)
        lower_values = interp_lower(x_arr)
        mid_values = (upper_values + lower_values) / 2.0

        # bin 分配
        bin_indices = np.searchsorted(shared_nodes, x_arr, side='right')

        is_above_mid = y_arr > mid_values
        bin_counts = np.bincount(bin_indices)
        max_bin = max(bin_indices.max() if len(bin_indices) else 0, 1)
        upper_bin_counts = np.bincount(bin_indices[is_above_mid], minlength=max_bin+1)
        lower_bin_counts = np.bincount(bin_indices[~is_above_mid], minlength=max_bin+1)

        # 新采样
        if resample_order < 1:
            x_resampled = x_arr
        else:
            x_resampled = LU_Mono.resample_1D(x_arr, resample_order)
        resampled_bin_indices = np.searchsorted(shared_nodes, x_resampled, side='right')
        upper_ratio_resampled = np.zeros_like(x_resampled, dtype=np.float64)
        valid_mask = resampled_bin_indices < len(bin_counts)
        valid_indices = resampled_bin_indices[valid_mask]

        if np.sum(lower_bin_counts) > np.sum(upper_bin_counts):
            ratio_calc = 1 - ((1 + lower_bin_counts[valid_indices]) / (2 + bin_counts[valid_indices]))
        else:
            ratio_calc = (1 + upper_bin_counts[valid_indices]) / (2 + bin_counts[valid_indices])
        upper_ratio_resampled[valid_mask] = ratio_calc

        upper_resampled = interp_upper(x_resampled)
        lower_resampled = interp_lower(x_resampled)
        y_resampled = upper_ratio_resampled * upper_resampled + (1 - upper_ratio_resampled) * lower_resampled

        self.interp_lower = interp_lower
        self.interp_upper = interp_upper
        self.interp_value = interpolate(x_resampled, y_resampled, extrapolate=False)
        self.interp_value_inv = LU_Mono.inverse_interpolate(y_resampled, x_resampled, is_decreasing, interpolate)
        self.interp_lower_inv = LU_Mono.inverse_interpolate(y_lower, x_lower, is_decreasing, interpolate)
        self.interp_upper_inv = LU_Mono.inverse_interpolate(y_upper, x_upper, is_decreasing, interpolate)

    @staticmethod
    cdef tuple profile_boundary(np.ndarray[np.float64_t] y, bint is_decreasing=True):
        """
        Compute the lower and upper monotonic boundaries of a 1D profile.

        Parameters
        ----------
        y : np.ndarray
            The dependent variable values.
        is_decreasing : bool, optional
            Whether the profile is decreasing.

        Returns
        -------
        lower_mask : np.ndarray
            Boolean mask for the lower boundary.
        upper_mask : np.ndarray
            Boolean mask for the upper boundary.
        """
        cdef int n = y.shape[0]
        lower_mask = np.zeros(n, dtype=bool)
        upper_mask = np.zeros(n, dtype=bool)

        cdef double max_val, min_val
        cdef int i
        cdef double[:] y_view = y

        # lower
        max_val = y_view[0]
        for i in range(n):
            if y_view[i] <= max_val:
                max_val = y_view[i]
                lower_mask[i] = True

        # upper
        min_val = y[n-1]
        for i in range(n):
            if y_view[n-1-i] >= min_val:
                min_val = y_view[n-1-i]
                upper_mask[n-1-i] = True

        if is_decreasing:
            return lower_mask, upper_mask
        else:
            return lower_mask[::-1], upper_mask[::-1]

    @staticmethod
    cdef np.ndarray[np.float64_t] resample_1D(np.ndarray[np.float64_t] x, int order):
        """
        Resample a 1D array with a given order.

        Parameters
        ----------
        x : np.ndarray
            The input array.
        order : int
            The resampling order.

        Returns
        -------
        x_resampled : np.ndarray
            The resampled array.
        """
        cdef int n = x.shape[0]
        cdef int new_n = n + order * (n - 1)
        cdef np.ndarray[np.float64_t] x_resampled = np.zeros(new_n, dtype=np.float64)
        cdef np.ndarray[np.float64_t] incre = np.diff(x)
        cdef int i, j, idx = 0
        for i in range(n-1):
            x_resampled[idx] = x[i]
            idx += 1
            for j in range(order):
                x_resampled[idx] = x[i] + (j+1) * incre[i] / (order+1)
                idx += 1
        x_resampled[idx] = x[n-1]
        return x_resampled

    @staticmethod
    cdef object inverse_interpolate(np.ndarray[np.float64_t] y, np.ndarray[np.float64_t] x, bint is_decreasing, object interpolate):
        """
        Construct an inverse interpolation function.

        Parameters
        ----------
        y : np.ndarray
            The dependent variable values.
        x : np.ndarray
            The independent variable values.
        is_decreasing : bool
            Whether the profile is decreasing.
        interpolate : callable
            The interpolation class or function.

        Returns
        -------
        inv_interp : object
            The inverse interpolation function.
        """
        if is_decreasing:
            y_temp = y[::-1]
            x_temp = x[::-1]
        else:
            y_temp = y
            x_temp = x
        sel = np.diff(y_temp, prepend=-np.inf) > 0
        return interpolate(y_temp[sel], x_temp[sel], extrapolate=False)

    @property
    def f_lower(self):
        """The lower bound interpolation function."""
        return self.interp_lower

    @property
    def f_upper(self):
        """The upper bound interpolation function."""
        return self.interp_upper

    @property
    def f_value(self):
        """The main interpolation function."""
        return self.interp_value

    @property
    def inv_f_value(self):
        """The inverse interpolation function for the main values."""
        return self.interp_value_inv

    @property
    def inv_f_lower(self):
        """The inverse interpolation function for the lower bound."""
        return self.interp_lower_inv

    @property
    def inv_f_upper(self):
        """The inverse interpolation function for the upper bound."""
        return self.interp_upper_inv