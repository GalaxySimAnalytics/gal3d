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
        np.ndarray[DTYPE_t, ndim=1] result_arr = np.zeros(n, dtype=np.float64)
        DTYPE_t[::1] result = result_arr  # Memoryview of result array
        
        # Precompute constants once
        double inv_a_squared = 1.0 / (a * a)
        double inv_b_squared = 1.0 / (b * b)
        double inv_c_squared = 1.0 / (c * c)
        double x, y, z, h1, h2, h3
    cdef int num_threads = config['general']['number_of_threads']
    # Skip parallelization for small arrays    
    if n < 500:
        for i in range(n):
            x = pos[i, 0]
            y = pos[i, 1]
            z = pos[i, 2]
            h1 = x * x * inv_a_squared
            h2 = y * y * inv_b_squared
            h3 = z * z * inv_c_squared
            result[i] = c_pow(h1, Sa) + c_pow(h2, Sb) + c_pow(h3, Sc)
    else:
        # Use prange for larger arrays
        for i in prange(n, nogil=True, num_threads=num_threads, schedule='static'):
            x = pos[i, 0]
            y = pos[i, 1]
            z = pos[i, 2]
            h1 = x * x * inv_a_squared
            h2 = y * y * inv_b_squared
            h3 = z * z * inv_c_squared
            result[i] = c_pow(h1, Sa) + c_pow(h2, Sb) + c_pow(h3, Sc)

    return result_arr


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

        # Precomputed constants
        double inv_a_squared = 1.0 / a / a
        double inv_b_squared = 1.0 / b / b
        double inv_c_squared = 1.0 / c / c
        double epsilon = 1e-9
        double initial_guess = (a + c) / 2.0

        # Local variables for inline calculation
        double x, y, z, L, xi, yi, zi
        double Ex, Ey, Ez, d0, d1
        int iteration
        double dd, ExddSa, EyddSb, EzddSc, f, df
    
    cdef int num_threads = config['general']['number_of_threads']
    
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
            
            # Precompute these values once
            Ex = c_pow((xi * xi) * inv_a_squared, Sa)
            Ey = c_pow((yi * yi) * inv_b_squared, Sb)
            Ez = c_pow((zi * zi) * inv_c_squared, Sc)
            
            # Iteratively solve for intersection (inlined)
            d0 = initial_guess
            for iteration in range(maxIterations):
                dd = d0 * d0
                ExddSa = Ex * c_pow(dd, Sa)
                EyddSb = Ey * c_pow(dd, Sb)
                EzddSc = Ez * c_pow(dd, Sc)
                f = ExddSa + EyddSb + EzddSc - 1.0
                df = 2.0 * (Sa * ExddSa + Sb * EyddSb + Sc * EzddSc) / d0
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
                
                # Precompute these values once
                Ex = c_pow((xi * xi) * inv_a_squared, Sa)
                Ey = c_pow((yi * yi) * inv_b_squared, Sb)
                Ez = c_pow((zi * zi) * inv_c_squared, Sc)
                
                # Iteratively solve for intersection (inlined)
                d0 = initial_guess
                for iteration in range(maxIterations):
                    dd = d0 * d0
                    ExddSa = Ex * c_pow(dd, Sa)
                    EyddSb = Ey * c_pow(dd, Sb)
                    EzddSc = Ez * c_pow(dd, Sc)
                    f = ExddSa + EyddSb + EzddSc - 1.0
                    df = 2.0 * (Sa * ExddSa + Sb * EyddSb + Sc * EzddSc) / d0
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
        int iteration
        np.ndarray[DTYPE_t, ndim=1] r = np.zeros(n, dtype=np.float64)

        # Precomputed constants
        double inv_a_squared = 1.0 / a / a
        double inv_b_squared = 1.0 / b / b
        double inv_c_squared = 1.0 / c / c
        double epsilon = 1e-9
        double initial_guess = (a + c) / 2.0
        
        double x, y, z
        double xi, yi, zi, Ex, Ey, Ez, d0, d1, Ltmp
        double dd, ExddSa, EyddSb, EzddSc, f, df
        
    cdef int num_threads = config['general']['number_of_threads']

    
    if n < 500:
        for i in range(n):
            x = pos[i, 0]
            y = pos[i, 1]
            z = pos[i, 2]
            
            # Calculate ray direction and length
            Ltmp = sqrt(x*x + y*y + z*z)
            xi = x / Ltmp
            yi = y / Ltmp
            zi = z / Ltmp
            
            # Calculate ellipsoid parameters
            Ex = c_pow(xi * xi * inv_a_squared, Sa)
            Ey = c_pow(yi * yi * inv_b_squared, Sb)
            Ez = c_pow(zi * zi * inv_c_squared, Sc)

            # Iteratively solve for intersection
            d0 = initial_guess
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
            
            # Store the result ratio directly
            r[i] = Ltmp / d0
    else:
        with nogil:
            for i in prange(n, num_threads=num_threads, schedule='static'):
                x = pos[i, 0]
                y = pos[i, 1]
                z = pos[i, 2]
                
                # Calculate ray direction and length
                Ltmp = sqrt(x*x + y*y + z*z)
                xi = x / Ltmp
                yi = y / Ltmp
                zi = z / Ltmp
                
                # Calculate ellipsoid parameters
                Ex = xi * xi * inv_a_squared
                Ey = yi * yi * inv_b_squared
                Ez = zi * zi * inv_c_squared

                # Iteratively solve for intersection
                d0 = initial_guess
                for iteration in range(maxIterations):
                    dd = d0 * d0
                    ExddSa = c_pow(Ex * dd, Sa) 
                    EyddSb = c_pow(Ey * dd, Sb) 
                    EzddSc = c_pow(Ez * dd, Sc) 
                    f = ExddSa + EyddSb + EzddSc - 1
                    df = 2 * (Sa * ExddSa + Sb * EyddSb + Sc * EzddSc) / d0
                    d1 = d0 - f / df
                    
                    if abs(d1 - d0) < epsilon:
                        break
                    
                    d0 = d1
                
                # Store the result ratio directly
                r[i] = Ltmp / d0

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
        int i, n = pos1.shape[0]
        np.ndarray[DTYPE_t, ndim=2] vects = unit_vector3d(pos2 - pos1)
        np.ndarray[DTYPE_t, ndim=1] tmax = vector_length3d(pos2 - pos1)
        np.ndarray[DTYPE_t, ndim=2] ts = -np.ones((n, 2), dtype=np.float64)

        # Precomputed constants
        double inv_a_squared = 1.0 / a / a
        double inv_b_squared = 1.0 / b / b
        double inv_c_squared = 1.0 / c / c
        double delta_cut = c/2.0
        double epsilon = 1e-9
        double ten_epsilon = 1e-8


        # Working variables
        double pos1_x, pos1_y, pos1_z, vect_x, vect_y, vect_z
        double posi_x, posi_y, posi_z, t0, t1, delta
        double Ex, Ey, Ez, f0, f1, df_x, df_y, df_z, df

        int iter_count
    
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

            # First iteration (t0 near 0)
            t0 = epsilon

            # Calculate position at current t
            posi_x = pos1_x + t0 * vect_x
            posi_y = pos1_y + t0 * vect_y
            posi_z = pos1_z + t0 * vect_z
            
            # Newton-Raphson iteration for first intersection
            iter_count = 0
            while True:
                # Calculate ellipsoid function terms
                Ex = c_pow((posi_x / a) * (posi_x / a), Sa)
                Ey = c_pow((posi_y / b) * (posi_y / b), Sb)
                Ez = c_pow((posi_z / c) * (posi_z / c), Sc)
                f0 = Ex + Ey + Ez - 1.0
                
                # Check if we've converged
                if abs(f0) < epsilon:
                    break
                
                # Calculate gradient components with robust handling of zero values
                df_x = 0.0 if posi_x == 0 else Ex * Sa * vect_x / posi_x
                df_y = 0.0 if posi_y == 0 else Ey * Sb * vect_y / posi_y
                df_z = 0.0 if posi_z == 0 else Ez * Sc * vect_z / posi_z
                df = 4.0 * (df_x + df_y + df_z)
                
                # Calculate step and apply adaptive limit
                delta = -f0 / df

                # Apply adaptive step size limit
                if f0 < 2.0:  # when near target pos
                    delta = min(delta_cut, delta)  # avoid large update 
                    delta = max(-delta_cut, delta)

                t0 = t0 + delta

                posi_x = pos1_x + t0 * vect_x
                posi_y = pos1_y + t0 * vect_y
                posi_z = pos1_z + t0 * vect_z
                iter_count += 1
                
                # Check if we've converged based on step size
                if abs(delta) < epsilon or iter_count > maxIteration:
                    break
            
            # Update f0 after loop
            Ex = c_pow(posi_x * posi_x * inv_a_squared, Sa)
            Ey = c_pow(posi_y * posi_y * inv_b_squared, Sb)
            Ez = c_pow(posi_z * posi_z * inv_c_squared, Sc)
            f0 = Ex + Ey + Ez - 1.0
            
            # Second iteration (t1 near tmax)
            t1 = tmax[i]

            # Calculate position at current t
            posi_x = pos1_x + t1 * vect_x
            posi_y = pos1_y + t1 * vect_y
            posi_z = pos1_z + t1 * vect_z
            
            # Newton-Raphson iteration for second intersection
            iter_count = 0
            while True:

                # Calculate ellipsoid function terms
                Ex = c_pow(posi_x * posi_x * inv_a_squared, Sa)
                Ey = c_pow(posi_y * posi_y * inv_b_squared, Sb)
                Ez = c_pow(posi_z * posi_z * inv_c_squared, Sc)
                f1 = Ex + Ey + Ez - 1.0
                
                # Check if we've converged
                if abs(f1) < epsilon:
                    break
                
                # Calculate gradient components with robust handling of zero values
                df_x = 0.0 if posi_x == 0 else Ex * Sa * vect_x / posi_x
                df_y = 0.0 if posi_y == 0 else Ey * Sb * vect_y / posi_y
                df_z = 0.0 if posi_z == 0 else Ez * Sc * vect_z / posi_z
                df = 4.0 * (df_x + df_y + df_z)

                # Calculate step and apply adaptive limit
                delta = -f1 / df

                # Apply adaptive step size limit
                if f1 < 2.0:    # When near the surface
                    delta = min(delta_cut, delta)
                    delta = max(-delta_cut, delta)

                t1 = t1 + delta

                posi_x = pos1_x + t1 * vect_x
                posi_y = pos1_y + t1 * vect_y
                posi_z = pos1_z + t1 * vect_z
                iter_count += 1
                
                # Check if we've converged based on step size
                if abs(delta) < epsilon or iter_count > maxIteration:
                    break
            
            # Update f1 after loop
            Ex = c_pow(posi_x * posi_x * inv_a_squared, Sa)
            Ey = c_pow(posi_y * posi_y * inv_b_squared, Sb)
            Ez = c_pow(posi_z * posi_z * inv_c_squared, Sc)
            f1 = Ex + Ey + Ez - 1.0
            
            # Determine intersection results
            if abs(t0 - t1) <= ten_epsilon:
                # Single intersection point (tangent)
                if (abs(f0) < ten_epsilon) or (abs(f1) < ten_epsilon):
                    ts[i, 0] = t0
                    ts[i, 1] = t1
                # else default is -1, -1
            elif (abs(f0) <= ten_epsilon) and (abs(f1) <= ten_epsilon):
                # Two distinct intersection points
                ts[i, 0] = t0
                ts[i, 1] = t1
            elif (abs(f0) <= ten_epsilon) or (abs(f1) <= ten_epsilon):
                # First point converged, second didn't
                if abs(f0) <= ten_epsilon:
                    ts[i, 0] = t0

                    # Try a point 2/3 of the way from t0 to t1
                    t2 = t0 + 0.66 * (t1 - t0)
                    ts[i, 1] = t2
                else:  # f1 <= 10 * epsilon
                    # Second point converged, first didn't
                    t2 = t0 + 0.33 * (t1 - t0)
                    ts[i, 0] = t2
                    ts[i, 1] = t1
            # else default is -1, -1
    else:
        # Parallel version with nogil
        with nogil:
            for i in prange(n, num_threads=num_threads, schedule='static'):
                pos1_x = pos1[i, 0]
                pos1_y = pos1[i, 1]
                pos1_z = pos1[i, 2]
                vect_x = vects[i, 0]
                vect_y = vects[i, 1]
                vect_z = vects[i, 2]

                # First iteration (t0 near 0)
                t0 = epsilon

                # Calculate position at current t
                posi_x = pos1_x + t0 * vect_x
                posi_y = pos1_y + t0 * vect_y
                posi_z = pos1_z + t0 * vect_z
                
                # Newton-Raphson iteration for first intersection
                for iter_count in range(maxIteration):
                    # Calculate ellipsoid function terms
                    Ex = c_pow(posi_x * posi_x * inv_a_squared, Sa)
                    Ey = c_pow(posi_y * posi_y * inv_b_squared, Sb)
                    Ez = c_pow(posi_z * posi_z * inv_c_squared, Sc)
                    f0 = Ex + Ey + Ez - 1.0
                    
                    # Check if we've converged
                    if abs(f0) < epsilon:
                        break
                    
                    # Calculate gradient components with robust handling of zero values
                    df_x = 0.0 if posi_x == 0 else Ex * Sa * vect_x / posi_x
                    df_y = 0.0 if posi_y == 0 else Ey * Sb * vect_y / posi_y
                    df_z = 0.0 if posi_z == 0 else Ez * Sc * vect_z / posi_z
                    df = 4.0 * (df_x + df_y + df_z)

                    # Calculate step and apply adaptive limit
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
                10
                # Update f0 after loop
                Ex = c_pow(posi_x * posi_x * inv_a_squared, Sa)
                Ey = c_pow(posi_y * posi_y * inv_b_squared, Sb)
                Ez = c_pow(posi_z * posi_z * inv_c_squared, Sc)
                f0 = Ex + Ey + Ez - 1.0

                # Second iteration (t1 near tmax)
                t1 = tmax[i]
                posi_x = pos1_x + t1 * vect_x
                posi_y = pos1_y + t1 * vect_y
                posi_z = pos1_z + t1 * vect_z
                
                # Inlined _iter_IntersectLineEllipsoid_S for t1
                # Changed from while True to for loop with range(maxIteration)
                for iter_count in range(maxIteration):
                    Ex = c_pow(posi_x * posi_x * inv_a_squared, Sa)
                    Ey = c_pow(posi_y * posi_y * inv_b_squared, Sb)
                    Ez = c_pow(posi_z * posi_z * inv_c_squared, Sc)
                    f1 = Ex + Ey + Ez - 1.0

                    if abs(f1) < epsilon:
                        break
                    
                    df_x = 0.0 if posi_x == 0 else Ex * Sa * vect_x / posi_x
                    df_y = 0.0 if posi_y == 0 else Ey * Sb * vect_y / posi_y
                    df_z = 0.0 if posi_z == 0 else Ez * Sc * vect_z / posi_z
                    df = 4.0 * (df_x + df_y + df_z)

                    delta = -f1 / df
                    if f1 < 2.0:    # When near the surface
                        delta = min(delta_cut, delta)
                        delta = max(-delta_cut, delta)
                    t1 = t1 + delta
                    posi_x = pos1_x + t1 * vect_x
                    posi_y = pos1_y + t1 * vect_y
                    posi_z = pos1_z + t1 * vect_z
                    
                    if abs(delta) < epsilon:
                        break
                
                # Update f1 after loop
                Ex = c_pow(posi_x * posi_x * inv_a_squared, Sa)
                Ey = c_pow(posi_y * posi_y * inv_b_squared, Sb)
                Ez = c_pow(posi_z * posi_z * inv_c_squared, Sc)
                f1 = Ex + Ey + Ez - 1.0

                # Determine intersection results
                if abs(t0 - t1) <= ten_epsilon:
                    # Single intersection point (tangent)
                    if (abs(f0) < ten_epsilon) or (abs(f1) < ten_epsilon):
                        ts[i, 0] = t0
                        ts[i, 1] = t1
                elif (abs(f0) <= ten_epsilon) and (abs(f1) <= ten_epsilon):
                    # Two distinct intersection points
                    ts[i, 0] = t0
                    ts[i, 1] = t1
                elif abs(f0) <= ten_epsilon:
                    # First point converged, second didn't
                    ts[i, 0] = t0
                    ts[i, 1] = t0 + 0.6 * (t1 - t0)
                elif abs(f1) <= ten_epsilon:
                    # Second point converged, first didn't
                    ts[i, 1] = t1
                    ts[i, 0] = t0 + 0.4 * (t1 - t0)
                # else default is -1, -1
    
    return ts