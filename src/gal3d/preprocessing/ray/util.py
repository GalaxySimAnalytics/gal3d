
import logging


import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import PchipInterpolator,Akima1DInterpolator

logger = logging.getLogger('gal3d.preprocessing.ray.util')

def judge_monoton(x,mono_de: bool = True) -> bool:
    '''
    judge whether x is monotonic decreasing or increasing
    '''
    
    if mono_de:
        judge_mono = all(np.diff(x) < 0)
    else:
        judge_mono =all(np.diff(x) > 0)
        
    return judge_mono



def sg_smooth(x,smooth_log,window_length,polyorder: int = 1, mode:str = 'nearest',**kwargs):
    
    if smooth_log:
        new_x = 10**savgol_filter(np.log10(x), window_length = window_length, polyorder=polyorder, mode = mode,**kwargs)
    else:
        new_x = savgol_filter(x, window_length = window_length, polyorder=polyorder, mode = mode,**kwargs)
    return new_x



def sg_smooth_tomono(x,smooth_log, polyorder:int =1, mode:str = 'nearest',mono_de:bool = True):
    '''
    using sgfilter to smooth x to be monotonic with minimum window size
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
    new_x = np.zeros(len(x)+order*(len(x)-1))
    incre = np.diff(x)
    xappend = [x[:-1]+ (i+1)*incre/(order+1) for i in range(order)]
    new_x[:-1] = np.array([x[:-1]]+xappend).T.flatten()
    new_x[-1] = x[-1]
    return new_x

class MyPchipInterpolator(PchipInterpolator):

    
    def __init__(self, *args, extrapolate='const',**kwargs):
        PchipInterpolator.__init__(self,*args,**kwargs)
        self.extrapolate = extrapolate
    

    def __call__(self, x, nu=0, extrapolate=None):
        x = np.asarray(x)
       # x = np.ascontiguousarray(x.ravel(), dtype=np.float64)
        if extrapolate is None:
            extrapolate = self.extrapolate
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

    
    def __init__(self, *args, extrapolate='const',**kwargs):
        PchipInterpolator.__init__(self,*args,**kwargs)
        self.extrapolate = extrapolate
    

    def __call__(self, x, nu=0, extrapolate=None):
        x = np.asarray(x)
       # x = np.ascontiguousarray(x.ravel(), dtype=np.float64)
        if extrapolate is None:
            extrapolate = self.extrapolate
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