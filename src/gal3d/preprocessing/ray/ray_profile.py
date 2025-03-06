

import logging


import numpy as np
from scipy.signal import savgol_filter

from .monotonic_profile import SG_Mono,LU_Mono,judge_monoton


logger = logging.getLogger('gal3d.preprocessing.ray.ray_profile')




class Ray:
    def __init__(self,r,f,f_de = True, interpolator_method:str = 'LU', interpolator_kwargs: dict = dict(),**kwargs):
        '''
        Interplote f(r), and r(f)
        
        Input:
            r, 1-D array, 
                A 1-D array of monotonically increasing real values.
            f, 1-D array,
                must be equal to the length of r.
            smooth_mode: {'SG', 'LU'}, optional,
                determine how to smooth f.  if 'SG', use the savgol_filter. Default to use 'LU', use the median of lower and upper values.
                
            interpolate_mode: {'Pchip', 'Akima', 'Spline'}, optional,
                determine how to interpolate f(r) and r(f), default 'Pchip'        
            
        kwargs:
            smoothlog: bool, default is False, smooth f in logscale
            mono_de: bool, default is True, the f(r) function is a monotonically decreasing function
            extrapolate: bool, default is True, Whether to extrapolate to out-of-bounds points based on first and last intervals, or to return NaNs
            throw_point: bool, default is True, when using SG smooth, will throw some bad points
            
        notes:
            if using smooth_mode = 'LU', we first cal (r_upper,f_upper),(r_lower,f_lower), then interpolate them
                then f_median = (f_upper+f_lower)/2, is what we want, and the error at r, can be get from f_lower and f_upper
                
            if using smooth_mode = 'SG', we first cal using sg_filter let f to be monotonic, then interpolate,
                f_smooth, is what we want, and the error at r, can get from (f/f_smooth)
        
        '''
        
        # r must be increasing
        if not judge_monoton(r,mono_de=False):
            raise ValueError(f"'r' must be strictly increasing sequence.")
        
        INTERPOLATOR = {'LU': LU_Mono,'SG':SG_Mono}
        interpolator = INTERPOLATOR[interpolator_method]
        
        
        self._interpolator = interpolator(r,f,y_de = f_de,**interpolator_kwargs)
        
        
    def __call__(self,value,inv = False):
        if inv:
            return self._interpolator.inv_f_value(value)
        return self._interpolator.f_value(value)
        
    def lower(self,value,inv = False):
        if inv:
            return self._interpolator.inv_f_lower(value)
        return self._interpolator.f_lower(value)
    
    def upper(self,value,inv = False):
        if inv:
            return self._interpolator.inv_f_upper(value)
        return self._interpolator.f_upper(value)
    