# lu_mono_cy.pyi

import numpy as np
from typing import Any, Optional, Union, List,overload

class MyPchipInterpolator:
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
    """

    def __init__(self, x: Any, y: Any, extrapolate: Union[str, bool] = 'const') -> None: ...
    @overload
    def __call__(self, x: float, nu: int = 0, extrapolate: Optional[Union[str, bool]] = None) -> float: ...
    @overload
    def __call__(self, x: np.ndarray, nu: int = 0, extrapolate: Optional[Union[str, bool]] = None) -> np.ndarray: ...
    @overload
    def solve(self, y: float, discontinuity: bool = True, extrapolate: bool = True) -> Union[float, np.ndarray]: ... 
    @overload
    def solve(self, y: np.ndarray, discontinuity: bool = True, extrapolate: bool = True) -> List[Union[float, np.ndarray]]: ...
    def __dealloc__(self) -> None: ...

class LU_Mono:
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

    def __init__(self, x: Any, y: Any, is_decreasing: bool = True, resample_order: int = 0, **kwargs) -> None: ...
    @property
    def f_lower(self) -> MyPchipInterpolator: ...
    @property
    def f_upper(self) -> MyPchipInterpolator: ...
    @property
    def f_value(self) -> MyPchipInterpolator: ...
    @property
    def inv_f_value(self) -> MyPchipInterpolator: ...
    @property
    def inv_f_lower(self) -> MyPchipInterpolator: ...
    @property
    def inv_f_upper(self) -> MyPchipInterpolator: ...