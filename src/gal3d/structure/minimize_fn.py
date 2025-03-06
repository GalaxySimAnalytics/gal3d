
import math

import numpy as np
from .structure_main import Structure_3D_fitter
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


@Structure_3D_fitter.minimize_func_fcall
@jit(float64(float64[:]),nogil=True,parallel=True,fastmath=True,cache=True,)
def isodensity_sums_fdev(f_call: np.float64 | np.ndarray,):
    h = 0. 
    for i in prange(len(f_call)):
        h = h + (f_call[i] - 1)*(f_call[i] - 1)
    h = h/len(f_call)
    return h

@Structure_3D_fitter.minimize_func_fcall
@jit(float64(float64[:],float64[:]),nogil=True,parallel=True,fastmath=True,cache=True,)
def isodensity_sums_fdev_byw(f_call: np.float64 | np.ndarray,w: np.float64 | np.ndarray):
    h = 0. 
    for i in prange(len(f_call)):
        h = h + (f_call[i] - 1)*(f_call[i] - 1)*w[i]
    h = h/len(f_call)
    return h


@Structure_3D_fitter.minimize_func_fcall
@jit(float64(float64[:], float64[:]),nogil=True,parallel=True,fastmath=True,cache=True,)
def isodensity_sums_fdev_rscale(f_call: np.float64 | np.ndarray, r: np.float64 | np.ndarray):
    h = 0.
    for i in prange(len(f_call)):
        h = h + ((f_call[i] - 1)*r[i])*((f_call[i] - 1)*r[i])
    h = h/len(f_call)
    return h

@Structure_3D_fitter.minimize_func_fcall
@jit(float64(float64[:], float64[:], float64[:]),nogil=True,parallel=True,fastmath=True,cache=True,)
def isodensity_sums_fdev_rscale_byw(f_call: np.float64 | np.ndarray, r: np.float64 | np.ndarray, w: np.float64 | np.ndarray):
    h = 0.
    for i in prange(len(f_call)):
        h = h + ((f_call[i] - 1)*r[i])*((f_call[i] - 1)*r[i])*w[i]
    h = h/len(f_call)
    return h

@Structure_3D_fitter.minimize_func_fcall
@jit(float64(float64[:]),nogil=True,parallel=True,fastmath=True,cache=True,)
def isodensity_sums_ddev(d_call: np.float64 | np.ndarray,):
    h = 0. 
    for i in prange(len(d_call)):
        h = h + (d_call[i] - 1)*(d_call[i] - 1)
    h = h/len(d_call)
    return h

@Structure_3D_fitter.minimize_func_fcall
@jit(float64(float64[:],float64[:]),nogil=True,parallel=True,fastmath=True,cache=True,)
def isodensity_sums_ddev_byw(d_call: np.float64 | np.ndarray,w: np.float64 | np.ndarray):
    h = 0. 
    for i in prange(len(d_call)):
        h = h + (d_call[i] - 1)*(d_call[i] - 1)*w[i]
    h = h/len(d_call)
    return h


@Structure_3D_fitter.minimize_func_fcall
@jit(float64(float64[:], float64[:]),nogil=True,parallel=True,fastmath=True,cache=True,)
def isodensity_sums_ddev_rscale(d_call: np.float64 | np.ndarray, r: np.float64 | np.ndarray):
    h = 0.
    for i in prange(len(d_call)):
        h = h + ((d_call[i] - 1)*r[i])*((d_call[i] - 1)*r[i])
    h = h/len(d_call)
    return h

@Structure_3D_fitter.minimize_func_fcall
@jit(float64(float64[:], float64[:], float64[:]),nogil=True,parallel=True,fastmath=True,cache=True,)
def isodensity_sums_ddev_rscale_byw(d_call: np.float64 | np.ndarray, r: np.float64 | np.ndarray, w: np.float64 | np.ndarray):
    h = 0.
    for i in prange(len(d_call)):
        h = h + ((d_call[i] - 1)*r[i])*((d_call[i] - 1)*r[i])*w[i]
    h = h/len(d_call)
    return h

@Structure_3D_fitter.minimize_func_fcall
@jit(float64(float64[:]),nogil=True,parallel=True,fastmath=True,cache=True,)
def isodensity_sums_dist(d_call):
    h = 0. 
    for i in prange(len(d_call)):
        h = h + d_call[i]*d_call[i]
    h = h/len(d_call)
    return h

@Structure_3D_fitter.minimize_func_fcall
@jit(float64(float64[:], float64[:]),nogil=True,parallel=True,fastmath=True,cache=True,)
def isodensity_sums_dist_rscale(d_call: np.float64 | np.ndarray, r: np.float64 | np.ndarray):
    h = 0.
    for i in prange(len(d_call)):
        h = h + d_call[i]*d_call[i]*r[i]*r[i]
    h = h/len(d_call)
    return h

@Structure_3D_fitter.minimize_func_fcall
@jit(float64(float64[:], float64[:], float64[:]),nogil=True,parallel=True,fastmath=True,cache=True,)
def shell_sums_padev(f_call1,f_call2,parameter):
    tarpa = parameter[(f_call1>1)&(f_call2<1)]
    mean = np.mean(tarpa)
    std = np.std(tarpa)
    h = 0.
    for i in prange(len(tarpa)):
        h = h + abs(tarpa[i]-mean)/std
    h = h/len(tarpa)
    return h


# sum (v[i]/V * abs(pa[i]-mean(pa)) / std(pa))
@Structure_3D_fitter.minimize_func_fcall
@jit(float64(float64[:], float64[:], float64[:],float64[:]),nogil=True,parallel=True,fastmath=True,cache=True,)
def grid_sums_padev(f_call1,f_call2,parameter,volumn):
    sel = (f_call1>1)&(f_call2<1)
    
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
@Structure_3D_fitter.minimize_func_fcall
@jit(float64(float64[:], float64[:], float64[:],float64[:]),nogil=True,parallel=True,fastmath=True,cache=True,)
def grid_sums_NeymanChi(f_call1,f_call2,parameter,volumn):
    sel = (f_call1>1)&(f_call2<1)
    
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
@Structure_3D_fitter.minimize_func_fcall
@jit(float64(float64[:], float64[:], float64[:],float64[:]),nogil=True,parallel=True,fastmath=True,cache=True,)
def grid_sums_PearsonChi(f_call1,f_call2,parameter,volumn):
    sel = (f_call1>1)&(f_call2<1)
    
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