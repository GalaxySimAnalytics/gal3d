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

_register_all()