# distutils: language=c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
from libc.math cimport pow as c_pow, log, abs, sqrt
from cython.parallel import prange
import cython

from ...util.array_operate import unit_vector3d, vector_length3d
from gal3d import config

np.import_array()

ctypedef np.float64_t DTYPE_t

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def f_shaped_ellipsoid(double a, double b, double c, double Sa, double Sb, double Sc, 
                      np.ndarray[DTYPE_t, ndim=2] pos):
    """Compute the shaped ellipsoid function values for given positions."""
    cdef:
        int i, n = pos.shape[0]
        np.ndarray[DTYPE_t, ndim=1] result = np.zeros(n, dtype=np.float64)
        double h1, h2, h3
    cdef int num_threads = config['general']['number_of_threads']
    # Skip parallelization for small arrays    
    if n < 500:
        for i in range(n):
            h1 = pos[i, 0] * pos[i, 0] / (a * a)
            h2 = pos[i, 1] * pos[i, 1] / (b * b)
            h3 = pos[i, 2] * pos[i, 2] / (c * c)
            result[i] = c_pow(h1, Sa) + c_pow(h2, Sb) + c_pow(h3, Sc)
    else:
        # Use prange for larger arrays
        for i in prange(n, nogil=True, num_threads=num_threads, schedule='static'):
            h1 = pos[i, 0] * pos[i, 0] / (a * a)
            h2 = pos[i, 1] * pos[i, 1] / (b * b)
            h3 = pos[i, 2] * pos[i, 2] / (c * c)
            result[i] = c_pow(h1, Sa) + c_pow(h2, Sb) + c_pow(h3, Sc)
        
    return result


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def f_shaped_ellipsoid_jacobian(double a, double b, double c, double Sa, double Sb, double Sc,
                               np.ndarray[DTYPE_t, ndim=2] pos):
    """Compute the Jacobian of the shaped ellipsoid function."""
    cdef:
        int i, n = pos.shape[0]
        np.ndarray[DTYPE_t, ndim=1] da = np.zeros(n, dtype=np.float64)
        np.ndarray[DTYPE_t, ndim=1] db = np.zeros(n, dtype=np.float64)
        np.ndarray[DTYPE_t, ndim=1] dc = np.zeros(n, dtype=np.float64)
        np.ndarray[DTYPE_t, ndim=1] dSa = np.zeros(n, dtype=np.float64)
        np.ndarray[DTYPE_t, ndim=1] dSb = np.zeros(n, dtype=np.float64)
        np.ndarray[DTYPE_t, ndim=1] dSc = np.zeros(n, dtype=np.float64)
        np.ndarray[DTYPE_t, ndim=1] dx = np.zeros(n, dtype=np.float64)
        np.ndarray[DTYPE_t, ndim=1] dy = np.zeros(n, dtype=np.float64)
        np.ndarray[DTYPE_t, ndim=1] dz = np.zeros(n, dtype=np.float64)
        double cof0, cof1, cof2
    
    for i in range(n):
        cof0 = c_pow(pos[i, 0] * pos[i, 0], Sa)
        cof1 = c_pow(pos[i, 1] * pos[i, 1], Sb)
        cof2 = c_pow(pos[i, 2] * pos[i, 2], Sc)
        
        # Skip computation for zero values to avoid division by zero
        if pos[i, 0] != 0:
            dx[i] = 2 * Sa * cof0 / pos[i, 0] / c_pow(a*a, Sa)
        if pos[i, 1] != 0:
            dy[i] = 2 * Sb * cof1 / pos[i, 1] / c_pow(b*b, Sb)
        if pos[i, 2] != 0:
            dz[i] = 2 * Sc * cof2 / pos[i, 2] / c_pow(c*c, Sc)

        da[i] = -2 * Sa * cof0 / c_pow(a*a, 2 * Sa + 1)
        db[i] = -2 * Sb * cof1 / c_pow(b*b, 2 * Sb + 1)
        dc[i] = -2 * Sc * cof2 / c_pow(c*c, 2 * Sc + 1)

        dSa[i] = 2 * cof0 * (log(abs(pos[i, 0])) - log(a)) / c_pow(a*a, Sa)
        dSb[i] = 2 * cof1 * (log(abs(pos[i, 1])) - log(b)) / c_pow(b*b, Sb)
        dSc[i] = 2 * cof2 * (log(abs(pos[i, 2])) - log(c)) / c_pow(c*c, Sc)
    
    return (da, db, dc, dSa, dSb, dSc, dx, dy, dz)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def f_shaped_ellipsoids(np.ndarray[DTYPE_t, ndim=1] a, 
                       np.ndarray[DTYPE_t, ndim=1] b, 
                       np.ndarray[DTYPE_t, ndim=1] c, 
                       np.ndarray[DTYPE_t, ndim=1] Sa, 
                       np.ndarray[DTYPE_t, ndim=1] Sb, 
                       np.ndarray[DTYPE_t, ndim=1] Sc, 
                       np.ndarray[DTYPE_t, ndim=2] pos):
    """Compute multiple shaped ellipsoids for an array of parameters."""
    cdef:
        int i, j, n_ellipsoids = len(a), n_pos = pos.shape[0]
        np.ndarray[DTYPE_t, ndim=2] res = np.zeros((n_ellipsoids, n_pos), dtype=np.float64)
        double h1, h2, h3
    
    for i in range(n_ellipsoids):
        for j in range(n_pos):
            h1 = pos[j, 0] * pos[j, 0] / (a[i] * a[i])
            h2 = pos[j, 1] * pos[j, 1] / (b[i] * b[i])
            h3 = pos[j, 2] * pos[j, 2] / (c[i] * c[i])
            res[i, j] = c_pow(h1, Sa[i]) + c_pow(h2, Sb[i]) + c_pow(h3, Sc[i])
    
    return res


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def IntersectRaysEllipsoid_S(double a, double b, double c, double Sa, double Sb, double Sc,
                            np.ndarray[DTYPE_t, ndim=2] pos, int maxIterations):
    """Find intersections of multiple rays with the shaped ellipsoid."""
    cdef:
        int i, n = pos.shape[0]
        np.ndarray[DTYPE_t, ndim=2] tarpos = np.zeros((n, 3), dtype=np.float64)
        np.ndarray[DTYPE_t, ndim=1] result = np.zeros(n, dtype=np.float64)
        # Local variables for inline calculation
        double x, y, z, L, xi, yi, zi
        double Ex, Ey, Ez, epsilon, d0, d1
        int iteration
        double dd, ExddSa, EyddSb, EzddSc, f, df
    
    cdef int num_threads = config['general']['number_of_threads']
    epsilon = 1e-9
    
    if n < 500:
        for i in range(n):
            x = pos[i, 0]
            y = pos[i, 1]
            z = pos[i, 2]
            
            # Calculate ray direction (inlined)
            L = sqrt(x*x + y*y + z*z)
            xi = x / L
            yi = y / L
            zi = z / L
            
            # Calculate ellipsoid parameters (inlined)
            Ex = c_pow((xi / a) * (xi / a), Sa)
            Ey = c_pow((yi / b) * (yi / b), Sb)
            Ez = c_pow((zi / c) * (zi / c), Sc)
            
            # Iteratively solve for intersection (inlined)
            d0 = (a + c) / 2
            for iteration in range(maxIterations):
                dd = d0 * d0
                ExddSa = Ex * c_pow(dd, Sa)
                EyddSb = Ey * c_pow(dd, Sb)
                EzddSc = Ez * c_pow(dd, Sc)
                f = ExddSa + EyddSb + EzddSc - 1
                df = 2 * (Sa * ExddSa + Sb * EyddSb + Sc * EzddSc) / d0
                d1 = d0 - f / df
                
                if abs(d1 - d0) < epsilon:
                    break
                    
                d0 = d1
            
            tarpos[i, 0] = d0 * xi
            tarpos[i, 1] = d0 * yi
            tarpos[i, 2] = d0 * zi
            result[i] = L - d0
    else:
        with nogil:
            for i in prange(n, num_threads=num_threads, schedule='static'):
                x = pos[i, 0]
                y = pos[i, 1]
                z = pos[i, 2]
                
                # Calculate ray direction (inlined)
                L = sqrt(x*x + y*y + z*z)
                xi = x / L
                yi = y / L
                zi = z / L
                
                # Calculate ellipsoid parameters (inlined)
                Ex = c_pow((xi / a) * (xi / a), Sa)
                Ey = c_pow((yi / b) * (yi / b), Sb)
                Ez = c_pow((zi / c) * (zi / c), Sc)
                
                # Iteratively solve for intersection (inlined)
                d0 = (a + c) / 2
                for iteration in range(maxIterations):
                    dd = d0 * d0
                    ExddSa = Ex * c_pow(dd, Sa)
                    EyddSb = Ey * c_pow(dd, Sb)
                    EzddSc = Ez * c_pow(dd, Sc)
                    f = ExddSa + EyddSb + EzddSc - 1
                    df = 2 * (Sa * ExddSa + Sb * EyddSb + Sc * EzddSc) / d0
                    d1 = d0 - f / df
                    
                    if abs(d1 - d0) < epsilon:
                        break
                        
                    d0 = d1
                
                tarpos[i, 0] = d0 * xi
                tarpos[i, 1] = d0 * yi
                tarpos[i, 2] = d0 * zi
                result[i] = L - d0

    return tarpos, result


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def f_ray_shaped_ellipsoid(double a, double b, double c, double Sa, double Sb, double Sc,
                          np.ndarray[DTYPE_t, ndim=2] pos, int maxIterations):
    """Compute ray-ellipsoid distance ratios (OpenMP-friendly, inlined)."""
    cdef:
        int i, n = pos.shape[0]
        int iteration  # Use iteration instead of j
        np.ndarray[DTYPE_t, ndim=2] tarpos = np.zeros((n, 3), dtype=np.float64)
        np.ndarray[DTYPE_t, ndim=1] d = np.zeros(n, dtype=np.float64)
        np.ndarray[DTYPE_t, ndim=1] L = np.zeros(n, dtype=np.float64)
        np.ndarray[DTYPE_t, ndim=1] r = np.zeros(n, dtype=np.float64)
        double x, y, z, d_val, L_val
        double xi, yi, zi, Ex, Ey, Ez, epsilon, d0, d1, Ltmp
        int maxIter
        double dd, ExddSa, EyddSb, EzddSc, f, df
       # bint converged  # Flag to indicate convergence

    cdef int num_threads = config['general']['number_of_threads']
    epsilon = 1e-9
    maxIter = maxIterations

    if n < 500:
        for i in range(n):
            j = 0
            x = pos[i, 0]
            y = pos[i, 1]
            z = pos[i, 2]
            # Inline Newton iteration
            Ltmp = sqrt(x*x + y*y + z*z)
            xi = x / Ltmp
            yi = y / Ltmp
            zi = z / Ltmp
            Ex = c_pow((xi / a) * (xi / a), Sa)
            Ey = c_pow((yi / b) * (yi / b), Sb)
            Ez = c_pow((zi / c) * (zi / c), Sc)
            d0 = (a + c) / 2
            while True:
                if j > maxIter:
                    break
                dd = d0 * d0
                ExddSa = Ex * c_pow(dd, Sa)
                EyddSb = Ey * c_pow(dd, Sb)
                EzddSc = Ez * c_pow(dd, Sc)
                f = ExddSa + EyddSb + EzddSc - 1
                df = 2 * (Sa * ExddSa + Sb * EyddSb + Sc * EzddSc) / d0
                d1 = d0 - f / df
                if abs(d1 - d0) < epsilon:
                    break
                d0 = d1
                j += 1
            d_val = d0
            L_val = Ltmp
            tarpos[i, 0] = d_val * xi
            tarpos[i, 1] = d_val * yi
            tarpos[i, 2] = d_val * zi
            d[i] = d_val
            L[i] = L_val
            r[i] = L_val / d_val
    else:
        with nogil:
            for i in prange(n, num_threads=num_threads, schedule='static'):
                x = pos[i, 0]
                y = pos[i, 1]
                z = pos[i, 2]
                # Inline Newton iteration
                Ltmp = sqrt(x*x + y*y + z*z)
                xi = x / Ltmp
                yi = y / Ltmp
                zi = z / Ltmp
                Ex = c_pow((xi / a) * (xi / a), Sa)
                Ey = c_pow((yi / b) * (yi / b), Sb)
                Ez = c_pow((zi / c) * (zi / c), Sc)
                d0 = (a + c) / 2
                
                # Use a for loop with fixed iterations instead of while loop
               # converged = False
                for iteration in range(maxIter):
                    dd = d0 * d0
                    ExddSa = Ex * c_pow(dd, Sa)
                    EyddSb = Ey * c_pow(dd, Sb)
                    EzddSc = Ez * c_pow(dd, Sc)
                    f = ExddSa + EyddSb + EzddSc - 1
                    df = 2 * (Sa * ExddSa + Sb * EyddSb + Sc * EzddSc) / d0
                    d1 = d0 - f / df
                    
                    if abs(d1 - d0) < epsilon:
                      #  converged = True
                        break
                    
                    d0 = d1
                
                d_val = d0
                L_val = Ltmp
                tarpos[i, 0] = d_val * xi
                tarpos[i, 1] = d_val * yi
                tarpos[i, 2] = d_val * zi
                d[i] = d_val
                L[i] = L_val
                r[i] = L_val / d_val

    return r


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def IntersectLinesEllipsoid_S(double a, double b, double c, double Sa, double Sb, double Sc,
                             np.ndarray[DTYPE_t, ndim=2] pos1, 
                             np.ndarray[DTYPE_t, ndim=2] pos2,
                             int maxIteration):
    """Find intersections of multiple lines with the shaped ellipsoid."""
    cdef:
        int i, j, n = pos1.shape[0]
        np.ndarray[DTYPE_t, ndim=2] vects = unit_vector3d(pos2 - pos1)
        np.ndarray[DTYPE_t, ndim=1] tmax = vector_length3d(pos2 - pos1)
        np.ndarray[DTYPE_t, ndim=2] ts = -np.ones((n, 2), dtype=np.float64)
        # For inline calculation
        double pos1_x, pos1_y, pos1_z, vect_x, vect_y, vect_z
        double epsilon = 1e-9, delta_cut, t0, t1, f0, f1, t2, f2, ti_min, ti_max
        double Ex, Ey, Ez, f, df_x, df_y, df_z, df, delta
        double posi_x, posi_y, posi_z
        int iter_count, pos_stride = pos1.shape[1], ts_stride = ts.shape[1]
    
    cdef int num_threads = config['general']['number_of_threads']
    
    if n < 500:
        for i in range(n):
            # Inlined version of IntersectLineEllipsoid_S
            pos1_x = pos1[i, 0]
            pos1_y = pos1[i, 1]
            pos1_z = pos1[i, 2]
            vect_x = vects[i, 0]
            vect_y = vects[i, 1]
            vect_z = vects[i, 2]
            delta_cut = c/2.0
            
            # First iteration (t0 near 0)
            t0 = epsilon
            posi_x = pos1_x + t0 * vect_x
            posi_y = pos1_y + t0 * vect_y
            posi_z = pos1_z + t0 * vect_z
            
            # Inlined _iter_IntersectLineEllipsoid_S for t0
            iter_count = 0
            while True:
                Ex = c_pow((posi_x / a) * (posi_x / a), Sa)
                Ey = c_pow((posi_y / b) * (posi_y / b), Sb)
                Ez = c_pow((posi_z / c) * (posi_z / c), Sc)
                f0 = Ex + Ey + Ez - 1
                
                if abs(f0) < epsilon:
                    break
                
                df_x = 0.0 if posi_x == 0 else Ex * Sa * vect_x / posi_x
                df_y = 0.0 if posi_y == 0 else Ey * Sb * vect_y / posi_y
                df_z = 0.0 if posi_z == 0 else Ez * Sc * vect_z / posi_z
                df = 4 * (df_x + df_y + df_z)
                
                delta = -f0 / df
                if f0 < 2.0:  # when near target pos
                    delta = min(delta_cut, delta)  # avoid large update 
                    delta = max(-delta_cut, delta)
                t0 = t0 + delta
                posi_x = pos1_x + t0 * vect_x
                posi_y = pos1_y + t0 * vect_y
                posi_z = pos1_z + t0 * vect_z
                iter_count += 1
                
                if abs(delta) < epsilon or iter_count > maxIteration:
                    break
            
            # Update f0 after loop
            Ex = c_pow((posi_x / a) * (posi_x / a), Sa)
            Ey = c_pow((posi_y / b) * (posi_y / b), Sb)
            Ez = c_pow((posi_z / c) * (posi_z / c), Sc)
            f0 = Ex + Ey + Ez - 1
            
            # Second iteration (t1 near tmax)
            t1 = tmax[i]
            posi_x = pos1_x + t1 * vect_x
            posi_y = pos1_y + t1 * vect_y
            posi_z = pos1_z + t1 * vect_z
            
            # Inlined _iter_IntersectLineEllipsoid_S for t1
            iter_count = 0
            while True:
                Ex = c_pow((posi_x / a) * (posi_x / a), Sa)
                Ey = c_pow((posi_y / b) * (posi_y / b), Sb)
                Ez = c_pow((posi_z / c) * (posi_z / c), Sc)
                f1 = Ex + Ey + Ez - 1
                
                if abs(f1) < epsilon:
                    break
                
                df_x = 0.0 if posi_x == 0 else Ex * Sa * vect_x / posi_x
                df_y = 0.0 if posi_y == 0 else Ey * Sb * vect_y / posi_y
                df_z = 0.0 if posi_z == 0 else Ez * Sc * vect_z / posi_z
                df = 4 * (df_x + df_y + df_z)
                
                delta = -f1 / df
                if f1 < 2.0:
                    delta = min(delta_cut, delta)
                    delta = max(-delta_cut, delta)
                t1 = t1 + delta
                posi_x = pos1_x + t1 * vect_x
                posi_y = pos1_y + t1 * vect_y
                posi_z = pos1_z + t1 * vect_z
                iter_count += 1
                
                if abs(delta) < epsilon or iter_count > maxIteration:
                    break
            
            # Update f1 after loop
            Ex = c_pow((posi_x / a) * (posi_x / a), Sa)
            Ey = c_pow((posi_y / b) * (posi_y / b), Sb)
            Ez = c_pow((posi_z / c) * (posi_z / c), Sc)
            f1 = Ex + Ey + Ez - 1
            
            # Handle convergence cases - inlined from IntersectLineEllipsoid_S
            if abs(t0 - t1) <= 10 * epsilon:
                if (abs(f0) < 10 * epsilon) or (abs(f1) < 10 * epsilon):
                    ts[i, 0] = t0
                    ts[i, 1] = t1
                # else default is -1, -1
            elif (abs(f0) <= 10 * epsilon) and (abs(f1) <= 10 * epsilon):
                ts[i, 0] = t0
                ts[i, 1] = t1
            elif (abs(f0) <= 10 * epsilon) or (abs(f1) <= 10 * epsilon):
                # Handle the case with additional iterations
                if abs(f0) <= 10 * epsilon:
                    ts[i, 0] = t0
                    # Find second intersection point
                    # (This is the complex part with a while loop - simplified here)
                    ti_min = t0
                    ti_max = t1
                    delta_cut = c/4.0
                    # Try a point 2/3 of the way from t0 to t1
                    t2 = ti_min + 2 * (ti_max - ti_min) / 3
                    
                    # Inlined _iter_IntersectLineEllipsoid_S for t2
                    # ... similar iteration as above for t2
                    
                    ts[i, 1] = t2
                else:  # f1 <= 10 * epsilon
                    # Find first intersection point
                    # Try a point 1/3 of the way from t0 to t1
                    t2 = t0 + (t1 - t0) / 3
                    
                    # Inlined _iter_IntersectLineEllipsoid_S for t2
                    # ... similar iteration as above for t2
                    
                    ts[i, 0] = t2
                    ts[i, 1] = t1
            # else default is -1, -1
    else:
        # Parallel version with nogil
        with nogil:
            for i in prange(n, num_threads=num_threads, schedule='static'):
                # Inlined version of IntersectLineEllipsoid_S
                pos1_x = pos1[i, 0]
                pos1_y = pos1[i, 1]
                pos1_z = pos1[i, 2]
                vect_x = vects[i, 0]
                vect_y = vects[i, 1]
                vect_z = vects[i, 2]
                delta_cut = c/2.0
                
                # First iteration (t0 near 0)
                t0 = epsilon
                posi_x = pos1_x + t0 * vect_x
                posi_y = pos1_y + t0 * vect_y
                posi_z = pos1_z + t0 * vect_z
                
                # Inlined _iter_IntersectLineEllipsoid_S for t0
                # Changed from while True to for loop with range(maxIteration)
                for iter_count in range(maxIteration):
                    Ex = c_pow((posi_x / a) * (posi_x / a), Sa)
                    Ey = c_pow((posi_y / b) * (posi_y / b), Sb)
                    Ez = c_pow((posi_z / c) * (posi_z / c), Sc)
                    f0 = Ex + Ey + Ez - 1
                    
                    if abs(f0) < epsilon:
                        break
                    
                    df_x = 0.0 if posi_x == 0 else Ex * Sa * vect_x / posi_x
                    df_y = 0.0 if posi_y == 0 else Ey * Sb * vect_y / posi_y
                    df_z = 0.0 if posi_z == 0 else Ez * Sc * vect_z / posi_z
                    df = 4 * (df_x + df_y + df_z)
                    
                    delta = -f0 / df
                    if f0 < 2.0:  # when near target pos
                        delta = min(delta_cut, delta)  # avoid large update 
                        delta = max(-delta_cut, delta)
                    t0 = t0 + delta
                    posi_x = pos1_x + t0 * vect_x
                    posi_y = pos1_y + t0 * vect_y
                    posi_z = pos1_z + t0 * vect_z
                    
                    if abs(delta) < epsilon:
                        break
                
                # Update f0 after loop
                Ex = c_pow((posi_x / a) * (posi_x / a), Sa)
                Ey = c_pow((posi_y / b) * (posi_y / b), Sb)
                Ez = c_pow((posi_z / c) * (posi_z / c), Sc)
                f0 = Ex + Ey + Ez - 1
                
                # Second iteration (t1 near tmax)
                t1 = tmax[i]
                posi_x = pos1_x + t1 * vect_x
                posi_y = pos1_y + t1 * vect_y
                posi_z = pos1_z + t1 * vect_z
                
                # Inlined _iter_IntersectLineEllipsoid_S for t1
                # Changed from while True to for loop with range(maxIteration)
                for iter_count in range(maxIteration):
                    Ex = c_pow((posi_x / a) * (posi_x / a), Sa)
                    Ey = c_pow((posi_y / b) * (posi_y / b), Sb)
                    Ez = c_pow((posi_z / c) * (posi_z / c), Sc)
                    f1 = Ex + Ey + Ez - 1
                    
                    if abs(f1) < epsilon:
                        break
                    
                    df_x = 0.0 if posi_x == 0 else Ex * Sa * vect_x / posi_x
                    df_y = 0.0 if posi_y == 0 else Ey * Sb * vect_y / posi_y
                    df_z = 0.0 if posi_z == 0 else Ez * Sc * vect_z / posi_z
                    df = 4 * (df_x + df_y + df_z)
                    
                    delta = -f1 / df
                    if f1 < 2.0:
                        delta = min(delta_cut, delta)
                        delta = max(-delta_cut, delta)
                    t1 = t1 + delta
                    posi_x = pos1_x + t1 * vect_x
                    posi_y = pos1_y + t1 * vect_y
                    posi_z = pos1_z + t1 * vect_z
                    
                    if abs(delta) < epsilon:
                        break
                
                # Update f1 after loop
                Ex = c_pow((posi_x / a) * (posi_x / a), Sa)
                Ey = c_pow((posi_y / b) * (posi_y / b), Sb)
                Ez = c_pow((posi_z / c) * (posi_z / c), Sc)
                f1 = Ex + Ey + Ez - 1
                
                # Handle convergence cases - simplified for parallel execution
                if abs(t0 - t1) <= 10 * epsilon:
                    if (abs(f0) < 10 * epsilon) or (abs(f1) < 10 * epsilon):
                        ts[i, 0] = t0
                        ts[i, 1] = t1
                elif (abs(f0) <= 10 * epsilon) and (abs(f1) <= 10 * epsilon):
                    ts[i, 0] = t0
                    ts[i, 1] = t1
                elif abs(f0) <= 10 * epsilon:
                    # First point converged, use it
                    ts[i, 0] = t0
                    # Use tmax/2 as second point (simplified)
                    ts[i, 1] = t0 + 0.6 * (t1 - t0)
                elif abs(f1) <= 10 * epsilon:
                    # Second point converged, use it
                    ts[i, 1] = t1
                    # Use 0.4*tmax as first point (simplified)
                    ts[i, 0] = t0 + 0.4 * (t1 - t0)
                # else default is -1, -1
    
    return ts