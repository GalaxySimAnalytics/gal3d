# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
import numpy as np

cimport numpy as np
from libc.math cimport fabs, sqrt

from cython.parallel import prange

from gal3d.configuration import config

# Import registry mechanism
from .minimize_func import MinimizeFunc

# Initialize numpy
np.import_array()

# Register functions after definition
def _register_all():
    """Register all functions with MinimizeFunc"""
    MinimizeFunc.fn_registry(sums_dev)
    MinimizeFunc.fn_registry(sums_dev_byw)
    MinimizeFunc.fn_registry(sums_dev_rscale)
    MinimizeFunc.fn_registry(sums_dev_rscale_byw)
    MinimizeFunc.fn_registry(shell_sums_padev)
    MinimizeFunc.fn_registry(grid_sums_padev)
    MinimizeFunc.fn_registry(grid_sums_NeymanChi)
    MinimizeFunc.fn_registry(grid_sums_PearsonChi)

cpdef double sums_dev(double[:] f_call):
    """
    Sum of squared function values.
    
    Parameters
    ----------
    f_call : array
        Function values
        
    Returns
    -------
    double
        Mean of squared values
    """
    cdef int i
    cdef int n = f_call.shape[0]
    cdef double h = 0.0
    
    # Use reduction for h in parallel loop
    cdef int num_threads = config['general']['number_of_threads']
    for i in prange(n, nogil=True,num_threads=num_threads, schedule='static'):
        h += f_call[i] * f_call[i]
    
    return h / n

cpdef double sums_dev_byw(double[:] f_call, double[:] w):
    """
    Weighted sum of squared function values.
    
    Parameters
    ----------
    f_call : array
        Function values
    w : array
        Weights
        
    Returns
    -------
    double
        Weighted mean of squared values
    """
    cdef int i
    cdef int n = f_call.shape[0]
    cdef double h = 0.0
    cdef int num_threads = config['general']['number_of_threads']
    for i in prange(n, nogil=True, num_threads=num_threads, schedule='static'):
        h += f_call[i] * f_call[i] * w[i]
    
    return h / n

cpdef double sums_dev_rscale(double[:] f_call, double[:] r):
    """
    Sum of squared function values scaled by r.
    
    Parameters
    ----------
    f_call : array
        Function values
    r : array
        Scale factors
        
    Returns
    -------
    double
        Mean of squared scaled values
    """
    cdef int i
    cdef int n = f_call.shape[0]
    cdef double h = 0.0
    cdef double scaled
    cdef int num_threads = config['general']['number_of_threads']
    for i in prange(n, nogil=True, num_threads=num_threads, schedule='static'):
        scaled = f_call[i] * r[i]
        h += scaled * scaled
    
    return h / n

cpdef double sums_dev_rscale_byw(double[:] f_call, double[:] r, double[:] w):
    """
    Weighted sum of squared function values scaled by r.
    
    Parameters
    ----------
    f_call : array
        Function values
    r : array
        Scale factors
    w : array
        Weights
        
    Returns
    -------
    double
        Weighted mean of squared scaled values
    """
    cdef int i
    cdef int n = f_call.shape[0]
    cdef double h = 0.0
    cdef double scaled
    cdef int num_threads = config['general']['number_of_threads']
    for i in prange(n, nogil=True, num_threads=num_threads, schedule='static'):
        scaled = f_call[i] * r[i]
        h += scaled * scaled * w[i]
    
    return h / n

cpdef double shell_sums_padev(double[:] f_call1, double[:] f_call2, double[:] parameter):
    """
    Statistical deviation of parameter values within shell.
    
    Parameters
    ----------
    f_call1 : array
        First function values
    f_call2 : array
        Second function values
    parameter : array
        Parameter values
        
    Returns
    -------
    double
        Normalized deviation measure
    """
    cdef int i, j, count = 0
    cdef int n = f_call1.shape[0]
    cdef double h = 0.0, mean = 0.0, std = 0.0
    cdef int num_threads = config['general']['number_of_threads']
    # First pass: create mask and compute mean
    # Note: we can't use numpy indexing with nogil, so we'll do this manually
    
    # Count valid elements
    for i in range(n):
        if f_call1[i] > 0 and f_call2[i] < 0:
            count += 1
    
    if count == 0:
        return 0.0
    
    # Allocate temporary arrays
    cdef double[:] tarpa = np.zeros(count, dtype=np.float64)
    
    # Fill temporary array
    j = 0
    for i in range(n):
        if f_call1[i] > 0 and f_call2[i] < 0:
            tarpa[j] = parameter[i]
            mean += tarpa[j]
            j += 1
    
    mean = mean / count
    
    # Compute standard deviation
    for i in range(count):
        std += (tarpa[i] - mean) * (tarpa[i] - mean)
    
    std = sqrt(std / count)
    
    if std == 0.0:
        return 0.0
    # Calculate final result - with private reduction variable
    h = 0.0
    for i in prange(count, nogil=True, num_threads=num_threads, schedule='static'):
        h += fabs(tarpa[i] - mean) / std
    
    return h / count

cpdef double grid_sums_padev(double[:] f_call1, double[:] f_call2, double[:] parameter, double[:] volumn):
    """
    Volume-weighted statistical deviation within grid selection.
    
    Parameters
    ----------
    f_call1 : array
        First function values
    f_call2 : array
        Second function values
    parameter : array
        Parameter values
    volumn : array
        Volume weights
        
    Returns
    -------
    double
        Volume-weighted normalized deviation
    """
    cdef int i, j, count = 0
    cdef int n = f_call1.shape[0]
    cdef double h = 0.0, mean = 0.0, sumv = 0.0, std = 0.0
    cdef int num_threads = config['general']['number_of_threads']
    # Count valid elements
    for i in range(n):
        if f_call1[i] > 0 and f_call2[i] < 0:
            count += 1
    
    if count == 0:
        return 0.0
    
    # Allocate temporary arrays
    cdef double[:] tarpa = np.zeros(count, dtype=np.float64)
    cdef double[:] tarvolumn = np.zeros(count, dtype=np.float64)
    
    # Fill temporary arrays
    j = 0
    for i in range(n):
        if f_call1[i] > 0 and f_call2[i] < 0:
            tarpa[j] = parameter[i]
            tarvolumn[j] = volumn[i]
            j += 1
    
    # Compute volume-weighted mean
    for i in range(count):
        mean += tarvolumn[i] * tarpa[i]
        sumv += tarvolumn[i]
    
    if sumv == 0.0:
        return 0.0
        
    mean = mean / sumv
    
    # Compute volume-weighted standard deviation
    for i in range(count):
        std += sqrt(tarvolumn[i] / sumv * (tarpa[i] - mean) * (tarpa[i] - mean))
    
    if std == 0.0:
        return 0.0
    # Calculate final result - reset h for parallel reduction
    h = 0.0
    for i in prange(count, nogil=True, num_threads=num_threads, schedule='static'):
        h += tarvolumn[i] / sumv * fabs(tarpa[i] - mean) / std
    
    return h

cpdef double grid_sums_NeymanChi(double[:] f_call1, double[:] f_call2, double[:] parameter, double[:] volumn):
    """
    Neyman's Chi-squared statistic.
    
    Parameters
    ----------
    f_call1 : array
        First function values
    f_call2 : array
        Second function values
    parameter : array
        Parameter values
    volumn : array
        Volume weights
        
    Returns
    -------
    double
        Neyman's Chi-squared statistic
    """
    cdef int i, j, count = 0
    cdef int n = f_call1.shape[0]
    cdef double h = 0.0, mean = 0.0, sumv = 0.0
    cdef int num_threads = config['general']['number_of_threads']
    
    # Count valid elements
    for i in range(n):
        if f_call1[i] > 0 and f_call2[i] < 0:
            count += 1
    
    if count == 0:
        return 0.0
    
    # Allocate temporary arrays
    cdef double[:] tarpa = np.zeros(count, dtype=np.float64)
    cdef double[:] tarvolumn = np.zeros(count, dtype=np.float64)
    
    # Fill temporary arrays
    j = 0
    for i in range(n):
        if f_call1[i] > 0 and f_call2[i] < 0:
            tarpa[j] = parameter[i]
            tarvolumn[j] = volumn[i]
            j += 1
    
    # Compute volume-weighted mean
    for i in range(count):
        mean += tarvolumn[i] * tarpa[i]
        sumv += tarvolumn[i]
    
    if sumv == 0.0:
        return 0.0
        
    mean = mean / sumv
    # Calculate Neyman's chi-squared - reset h for parallel reduction
    h = 0.0
    for i in prange(count, nogil=True, num_threads=num_threads, schedule='static'):
        if tarpa[i] > 0:  # Avoid division by zero
            h += tarvolumn[i] / sumv * fabs(tarpa[i] - mean) / sqrt(tarpa[i])
    
    return h

cpdef double grid_sums_PearsonChi(double[:] f_call1, double[:] f_call2, double[:] parameter, double[:] volumn):
    """
    Pearson's Chi-squared statistic.
    
    Parameters
    ----------
    f_call1 : array
        First function values
    f_call2 : array
        Second function values
    parameter : array
        Parameter values
    volumn : array
        Volume weights
        
    Returns
    -------
    double
        Pearson's Chi-squared statistic
    """
    cdef int i, j, count = 0
    cdef int n = f_call1.shape[0]
    cdef double h = 0.0, mean = 0.0, sumv = 0.0, std = 0.0
    cdef int num_threads = config['general']['number_of_threads']
    # Count valid elements
    for i in range(n):
        if f_call1[i] > 0 and f_call2[i] < 0:
            count += 1
    
    if count == 0:
        return 0.0
    
    # Allocate temporary arrays
    cdef double[:] tarpa = np.zeros(count, dtype=np.float64)
    cdef double[:] tarvolumn = np.zeros(count, dtype=np.float64)
    
    # Fill temporary arrays
    j = 0
    for i in range(n):
        if f_call1[i] > 0 and f_call2[i] < 0:
            tarpa[j] = parameter[i]
            tarvolumn[j] = volumn[i]
            j += 1
    
    # Compute volume-weighted mean
    for i in range(count):
        mean += tarvolumn[i] * tarpa[i]
        sumv += tarvolumn[i]
    
    if sumv == 0.0:
        return 0.0
        
    mean = mean / sumv
    std = sqrt(mean)  # Model-based standard deviation
    
    if std == 0.0:
        return 0.0
    
    # Calculate Pearson's chi-squared - reset h for parallel reduction
    h = 0.0
    for i in prange(count, nogil=True, num_threads=num_threads, schedule='static'):
        h += tarvolumn[i] / sumv * fabs(tarpa[i] - mean) / std
    
    return h

# Register all functions at import time
_register_all()