
import logging


import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import PchipInterpolator,Akima1DInterpolator

logger = logging.getLogger('gal3d.preprocessing.ray.util')

def judge_monoton(x,mono_de: bool = True) -> bool:
    '''
    Judge whether the array `x` is monotonically decreasing or increasing.

    Parameters
    ----------
    x : array_like
        The input array to be checked for monotonicity.
    mono_de : bool, optional
        If True, checks for monotonically decreasing. If False, checks for monotonically increasing.
        Default is True.

    Returns
    -------
    bool
        True if `x` is monotonic (either decreasing or increasing based on `mono_de`), False otherwise.
    '''
    
    if mono_de:
        judge_mono = all(np.diff(x) < 0)
    else:
        judge_mono =all(np.diff(x) > 0)
        
    return judge_mono



def sg_smooth(x,smooth_log,window_length,polyorder: int = 1, mode:str = 'nearest',**kwargs):
    '''
    Smooth the input array `x` using the Savitzky-Golay filter.

    Parameters
    ----------
    x : array_like
        Input array to be smoothed.
    smooth_log : bool
        If True, apply the filter to the logarithm of `x` and then exponentiate the result.
    window_length : int
        The length of the filter window (i.e., the number of coefficients).
    polyorder : int, optional
        The order of the polynomial used to fit the samples. Default is 1.
    mode : str, optional
        The mode parameter for the Savitzky-Golay filter. Default is 'nearest'.
    **kwargs : dict
        Additional keyword arguments passed to `savgol_filter`.

    Returns
    -------
    array_like
        The smoothed array.
    '''
    if smooth_log:
        new_x = 10**savgol_filter(np.log10(x), window_length = window_length, polyorder=polyorder, mode = mode,**kwargs)
    else:
        new_x = savgol_filter(x, window_length = window_length, polyorder=polyorder, mode = mode,**kwargs)
    return new_x



def sg_smooth_tomono(x,smooth_log, polyorder:int =1, mode:str = 'nearest',mono_de:bool = True):
    '''
    Smooth the input array `x` using the Savitzky-Golay filter until it becomes monotonic.

    Parameters
    ----------
    x : array_like
        Input array to be smoothed.
    smooth_log : bool
        If True, apply the filter to the logarithm of `x` and then exponentiate the result.
    polyorder : int, optional
        The order of the polynomial used to fit the samples. Default is 1.
    mode : str, optional
        The mode parameter for the Savitzky-Golay filter. Default is 'nearest'.
    mono_de : bool, optional
        If True, ensures the smoothed array is monotonically decreasing. If False, ensures it is monotonically increasing.
        Default is True.

    Returns
    -------
    tuple
        A tuple containing the smoothed array and the final window size used for smoothing.
    '''
    windowmin = 2
    windowi = len(x)
    maxwindowsize = len(x)
    smoothflag = True
    
    while(smoothflag):
        new_x = sg_smooth(x,smooth_log=smooth_log,window_length=windowi,polyorder=polyorder, mode = mode)
        
        judge_mono = judge_monoton(new_x,mono_de)
        
        if judge_mono:
            windowmax = windowi
            if (windowi == windowmin) or (windowmax == (windowmin + 1)):
                smoothflag = False
            windowi = int((windowi + windowmin)/2)
        else:
            if (windowi >= maxwindowsize):
                smoothflag = False
            windowmin = windowi
            windowi = max(int((windowi+ windowmax)/2),windowmin+1)
    return new_x,windowi


def sg_smooth_throw(x,smooth_log,window_length_max,mono_de,polyorder: int = 1, mode:str = 'nearest',):
    '''
    Smooth the input array `x` using the Savitzky-Golay filter, removing non-monotonic points iteratively.

    Parameters
    ----------
    x : array_like
        Input array to be smoothed.
    smooth_log : bool
        If True, apply the filter to the logarithm of `x` and then exponentiate the result.
    window_length_max : int
        The maximum window length for the Savitzky-Golay filter.
    mono_de : bool
        If True, ensures the smoothed array is monotonically decreasing. If False, ensures it is monotonically increasing.
    polyorder : int, optional
        The order of the polynomial used to fit the samples. Default is 1.
    mode : str, optional
        The mode parameter for the Savitzky-Golay filter. Default is 'nearest'.

    Returns
    -------
    tuple
        A tuple containing the smoothed array and the indices of the remaining points.
    '''
    sel = np.ones(len(x),dtype=bool)
    ind = np.arange(len(x))
    
    flag = False
    x_smooth = np.array(x)
    
    iter_num = 0
    while (flag is False):
        x_smooth = x_smooth[sel]
        ind = ind[sel]
        
        sel = sel[sel]
        smoothed = sg_smooth(x_smooth,smooth_log=smooth_log,window_length = window_length_max,polyorder=polyorder,mode=mode)
        
        flag = judge_monoton(smoothed,mono_de)
        if flag:
            break
        sel[np.diff(smoothed,append=0)>0] = False
        sel[np.diff(smoothed,prepend=smoothed[0]+1)>0] = False
        sel[0] = True
        sel[-1] = True
        iter_num = iter_num+1
    return smoothed,ind

def resample_1D(x,order:int = 1):
    '''
    Resample the input array `x` by inserting additional points between existing points.

    Parameters
    ----------
    x : array_like
        Input array to be resampled.
    order : int, optional
        The number of additional points to insert between each pair of existing points. Default is 1.

    Returns
    -------
    array_like
        The resampled array.
    '''
    new_x = np.zeros(len(x)+order*(len(x)-1))
    incre = np.diff(x)
    xappend = [x[:-1]+ (i+1)*incre/(order+1) for i in range(order)]
    new_x[:-1] = np.array([x[:-1]]+xappend).T.flatten()
    new_x[-1] = x[-1]
    return new_x

class MyPchipInterpolator(PchipInterpolator):
    '''
    A custom PCHIP interpolator with additional extrapolation options.

    Parameters
    ----------
    *args : tuple
        Arguments passed to the base `PchipInterpolator` class.
    extrapolate : str, optional
        The extrapolation mode. If 'const', extrapolation is constant outside the bounds.
        Default is 'const'.
    **kwargs : dict
        Additional keyword arguments passed to the base `PchipInterpolator` class.
    '''
    
    def __init__(self, *args, extrapolate='const',**kwargs):
        PchipInterpolator.__init__(self,*args,**kwargs)
        self.extrapolate = extrapolate
    

    def __call__(self, x, nu=0, extrapolate=None):
        '''
        Evaluate the interpolator at the given points.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the interpolator.
        nu : int, optional
            The order of the derivative to evaluate. Default is 0.
        extrapolate : str, optional
            The extrapolation mode. If None, uses the mode set during initialization.
            Default is None.

        Returns
        -------
        array_like
            The interpolated values at the points `x`.
        '''
        x = np.asarray(x)
       # x = np.ascontiguousarray(x.ravel(), dtype=np.float64)

        extrapolate = extrapolate or self.extrapolate
        if extrapolate == 'const':
            x = np.asarray(x.clip(self.x[0],self.x[-1]))
            return super().__call__(x,nu,extrapolate)
        
        return super().__call__(x,nu,extrapolate)
        
        # TODO
        if extrapolate == 'linear':
            xnew = np.asarray(x.clip(self.x[0],self.x[-1]))
            y = super().__call__(x,nu,extrapolate)
            dy1 = super().__call__(self.x[0],1)
            dy2 = super().__call__(self.x[-1],1)
            dx = np.asarray(xnew - x)
            dx[dx>0] = dx[dx>0]*dy1
            dx[dx<0] = dx[dx<0]*dy2
            return y-dx
        
class MyAkima1DInterpolator(Akima1DInterpolator):
    '''
    A custom Akima 1D interpolator with additional extrapolation options.

    Parameters
    ----------
    *args : tuple
        Arguments passed to the base `Akima1DInterpolator` class.
    extrapolate : str, optional
        The extrapolation mode. If 'const', extrapolation is constant outside the bounds.
        Default is 'const'.
    **kwargs : dict
        Additional keyword arguments passed to the base `Akima1DInterpolator` class.
    '''
    
    def __init__(self, *args, extrapolate='const',**kwargs):
        PchipInterpolator.__init__(self,*args,**kwargs)
        self.extrapolate = extrapolate
    

    def __call__(self, x, nu=0, extrapolate=None):
        '''
        Evaluate the interpolator at the given points.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the interpolator.
        nu : int, optional
            The order of the derivative to evaluate. Default is 0.
        extrapolate : str, optional
            The extrapolation mode. If None, uses the mode set during initialization.
            Default is None.

        Returns
        -------
        array_like
            The interpolated values at the points `x`.
        '''
        x = np.asarray(x)
       # x = np.ascontiguousarray(x.ravel(), dtype=np.float64)

        extrapolate = extrapolate or self.extrapolate
        if extrapolate == 'const':
            x = np.asarray(x.clip(self.x[0],self.x[-1]))
            return super().__call__(x,nu,extrapolate)
        
        return super().__call__(x,nu,extrapolate)
        
        # TODO
        if extrapolate == 'linear':
            xnew = np.asarray(x.clip(self.x[0],self.x[-1]))
            y = super().__call__(x,nu,extrapolate)
            dy1 = super().__call__(self.x[0],1)
            dy2 = super().__call__(self.x[-1],1)
            dx = np.asarray(xnew - x)
            dx[dx>0] = dx[dx>0]*dy1
            dx[dx<0] = dx[dx<0]*dy2
            return y-dx