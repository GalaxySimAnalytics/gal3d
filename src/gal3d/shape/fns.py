
import math

import numpy as np
from .minimize_func import MinimizeFunc
from numba import (
    int32,
    deferred_type,
    optional,
    float64,
    boolean,
    int64,
    njit,
    jit,
    prange,
    types,
)


@MinimizeFunc.fn_registry
@jit(float64(float64[:]),nogil=True,parallel=True,fastmath=True,cache=True,)
def sums_dev(f_call: np.float64 | np.ndarray,):
    h = 0. 
    for i in prange(len(f_call)):
        h = h + f_call[i]*f_call[i]
    h = h/len(f_call)
    return h

@MinimizeFunc.fn_registry
@jit(float64(float64[:],float64[:]),nogil=True,parallel=True,fastmath=True,cache=True,)
def sums_dev_byw(f_call: np.float64 | np.ndarray,w: np.float64 | np.ndarray):
    h = 0. 
    for i in prange(len(f_call)):
        h = h + f_call[i]*f_call[i]*w[i]
    h = h/len(f_call)
    return h


@MinimizeFunc.fn_registry
@jit(float64(float64[:], float64[:]),nogil=True,parallel=True,fastmath=True,cache=True,)
def sums_dev_rscale(f_call: np.float64 | np.ndarray, r: np.float64 | np.ndarray):
    h = 0.
    for i in prange(len(f_call)):
        h = h + (f_call[i]*r[i])*(f_call[i]*r[i])
    h = h/len(f_call)
    return h

@MinimizeFunc.fn_registry
@jit(float64(float64[:], float64[:], float64[:]),nogil=True,parallel=True,fastmath=True,cache=True,)
def sums_dev_rscale_byw(f_call: np.float64 | np.ndarray, r: np.float64 | np.ndarray, w: np.float64 | np.ndarray):
    h = 0.
    for i in prange(len(f_call)):
        h = h + (f_call[i]*r[i])*(f_call[i]*r[i])*w[i]
    h = h/len(f_call)
    return h





@MinimizeFunc.fn_registry
@jit(float64(float64[:], float64[:], float64[:]),nogil=True,parallel=True,fastmath=True,cache=True,)
def shell_sums_padev(f_call1,f_call2,parameter):
    tarpa = parameter[(f_call1>0)&(f_call2<0)]
    mean = np.mean(tarpa)
    std = np.std(tarpa)
    h = 0.
    for i in prange(len(tarpa)):
        h = h + abs(tarpa[i]-mean)/std
    h = h/len(tarpa)
    return h


# sum (v[i]/V * abs(pa[i]-mean(pa)) / std(pa))
@MinimizeFunc.fn_registry
@jit(float64(float64[:], float64[:], float64[:],float64[:]),nogil=True,parallel=True,fastmath=True,cache=True,)
def grid_sums_padev(f_call1,f_call2,parameter,volumn):
    sel = (f_call1>0)&(f_call2<0)
    
    tarpa = parameter[sel]   # log ??
    tarvolumn = volumn[sel]
    
    mean = 0.
    sumv = 0.
    std = 0.
    for i in prange(len(tarpa)):
        mean = mean + tarvolumn[i]*tarpa[i]
        sumv = sumv + tarvolumn[i]
    mean = mean/sumv
    for i in prange(len(tarpa)):
        std = std + math.sqrt(tarvolumn[i]/sumv*(tarpa[i]-mean)**2)
    
    h = 0.
    for i in prange(len(tarpa)):
       # der = abs(tarpa[i]-mean)/std
        #if der > 3:                 # 3sigma clip ??
        #    continue            
        h = h + tarvolumn[i]/sumv*abs(tarpa[i]-mean)/std
        
    return h


# Gaussian MLE statistic using data pixel values for σ (“Neyman’s χ2”)
@MinimizeFunc.fn_registry
@jit(float64(float64[:], float64[:], float64[:],float64[:]),nogil=True,parallel=True,fastmath=True,cache=True,)
def grid_sums_NeymanChi(f_call1,f_call2,parameter,volumn):
    sel = (f_call1>0)&(f_call2<0)
    
    tarpa = parameter[sel]   
    tarvolumn = volumn[sel]
    
    mean = 0.
    sumv = 0.
    for i in prange(len(tarpa)):
        mean = mean + tarvolumn[i]*tarpa[i]
        sumv = sumv + tarvolumn[i]
    mean = mean/sumv    
    h = 0.
    for i in prange(len(tarpa)):
        h = h + tarvolumn[i]/sumv*abs(tarpa[i]-mean)/math.sqrt(tarpa[i])
    return h

#Gaussian MLE statistic using model pixel values for σ (“Pearson’s χ2”)
@MinimizeFunc.fn_registry
@jit(float64(float64[:], float64[:], float64[:],float64[:]),nogil=True,parallel=True,fastmath=True,cache=True,)
def grid_sums_PearsonChi(f_call1,f_call2,parameter,volumn):
    sel = (f_call1>0)&(f_call2<0)
    
    tarpa = parameter[sel]   
    tarvolumn = volumn[sel]
    
    mean = 0.
    sumv = 0.
    for i in prange(len(tarpa)):
        mean = mean + tarvolumn[i]*tarpa[i]
        sumv = sumv + tarvolumn[i]
    mean = mean/sumv 
    std = math.sqrt(mean)   
    h = 0.
    for i in prange(len(tarpa)):
        h = h + tarvolumn[i]/sumv*abs(tarpa[i]-mean)/std
    return h