import logging


import numpy as np


from .util import savgol_filter,sg_smooth,sg_smooth_tomono,judge_monoton,sg_smooth_throw,MyPchipInterpolator,MyAkima1DInterpolator,resample_1D
from ...util.func_signature import func_optional_key,update_dict_value


logger = logging.getLogger("gal3d.preprocessing.ray.monotonic_profile")




def inverse_interpolate(y,x,y_de,interpolate,extrapolate=False):
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
    sel = np.diff(y_temp,prepend=-np.inf)>0
    
    return interpolate(y_temp[sel],x_temp[sel],extrapolate=extrapolate)

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
    
    def __init__(self,x,y,smooth_log = True,window_length_max_frac = 0.1,y_de = True,throw=True,interpolate_mode='Pchip',polyorder: int = 1, mode:str = 'nearest',**kwargs):
        INTERPOLATE = {'Pchip': MyPchipInterpolator,'Akima':MyAkima1DInterpolator}
        interpolate = INTERPOLATE[interpolate_mode]
        
        if throw:
            y_sm,ind = sg_smooth_throw(y,smooth_log=smooth_log,window_length_max=min(int(window_length_max_frac*len(x)),len(x)-1),
                            mono_de=y_de,polyorder =polyorder, mode=mode)
            x = x[ind]
            y = y[ind]
        else:
            y_sm = sg_smooth_tomono(y,smooth_log,polyorder,mode,y_de)
            
        self.f_value = interpolate(x,y_sm,extrapolate=False)
        
        if y_de:
            self.inv_f = interpolate(y_sm[::-1],x[::-1],extrapolate=False)
        else:
            self.inv_f = interpolate(y_sm,x,extrapolate=False)
        
        vars = np.abs(y_sm/y-1)
        var = np.percentile(vars,84)
        self.f_lower = interpolate(x,y_sm*(1-var),extrapolate=False)
        self.f_upper = interpolate(x,y_sm*(1+var),extrapolate=False)
        
        
        self.inv_f_value = inverse_interpolate(y_sm,x,y_de,interpolate,extrapolate=False)
        self.inv_f_lower = inverse_interpolate(y_sm*(1-var),x,y_de,interpolate,extrapolate=False)
        self.inv_f_upper = inverse_interpolate(y_sm*(1+var),x,y_de,interpolate,extrapolate=False)
    
    


class LU_Mono:
    """
    A class for smoothing and interpolating profiles using lower and upper bounds.

    Parameters
    ----------
    x : array-like
        The independent variable values.
    y : array-like
        The dependent variable values.
    y_de : bool, optional
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
    def __init__(self,x, y, y_de = True,interpolate_mode='Pchip', re_sample_ord = 0):
        INTERPOLATE = {'Pchip': MyPchipInterpolator,'Akima':MyAkima1DInterpolator}
        interpolate = INTERPOLATE[interpolate_mode]
        
        
        lo,up = self.profile_boundary(y,mono_de=y_de)
        fupper = interpolate(x[up],y[up],extrapolate='const')
        flower = interpolate(x[lo],y[lo],extrapolate='const')
    
        rnodes = x[lo&up]
        
        new_x = x if re_sample_ord<1 else resample_1D(x,re_sample_ord)
        
        
        rbins = np.bincount(np.searchsorted(rnodes, x,side='right'))
        upper = np.bincount(np.searchsorted(rnodes, x[y>((fupper(x)+flower(x))/2)],side='right'))
        lower = np.bincount(np.searchsorted(rnodes, x[y<=((fupper(x)+flower(x))/2)],side='right'))
        
        new_x_ind = np.searchsorted(rnodes, new_x,side='right')
        
        if len(lower)>len(upper):
            new_y_upper_ratio = 1- np.array([(1+lower[i])/(2+rbins[i]) for i in new_x_ind])
        else:
            new_y_upper_ratio = np.array([(1+upper[i])/(2+rbins[i]) for i in new_x_ind])
            
        new_y = new_y_upper_ratio*fupper(new_x)+(1-new_y_upper_ratio)*flower(new_x)
        
        ftarget = interpolate(new_x,new_y,extrapolate=False)
    
        self.f_lower = flower
        self.f_upper = fupper
        self.f_value = ftarget
        
        
        self.inv_f_value = inverse_interpolate(new_y,new_x,y_de,interpolate,extrapolate=False)
        self.inv_f_lower = inverse_interpolate(y[lo],x[lo],y_de,interpolate,extrapolate=False)
        self.inv_f_upper = inverse_interpolate(y[up],x[up],y_de,interpolate,extrapolate=False)


            
        
    @staticmethod
    def profile_boundary(y,mono_de = True):
        """
        Determine the lower and upper boundaries of the profile.

        Parameters
        ----------
        y : array-like
            The dependent variable values.
        mono_de : bool, optional
            If True, the y values are in decreasing order; otherwise, they are in increasing order. Default is True.

        Returns
        -------
        tuple
            A tuple containing the lower and upper boundary indices.
        """
        def sel_lower(arr):
            sel = np.zeros(len(arr),dtype=bool)
            vamax = arr[0]
            for idx,va in enumerate(arr):
                if va<=vamax:
                    vamax = va
                    sel[idx] = True
            return sel

        def sel_upper(arr):
            sel = np.zeros(len(arr),dtype=bool)
            vamin = arr[-1]
            for idx,va in enumerate(arr[::-1]):
                if va>=vamin:
                    vamin = va
                    sel[idx] = True
            return sel[::-1]
        
        if mono_de:
            data = y
            upper = sel_upper(data)
            lower = sel_lower(data)
            return lower,upper
        
        data = y[::-1]
        upper = sel_upper(data)
        lower = sel_lower(data)
        return lower[::-1],upper[::-1]
    