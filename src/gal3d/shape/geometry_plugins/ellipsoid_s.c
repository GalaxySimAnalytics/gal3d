#include "ellipsoid_s.h"
#include <math.h>
#include <omp.h>


void f_shaped_ellipsoid_cpp(
    double a, double b, double c, double Sa, double Sb, double Sc,
    const double* pos, int n, double* result, int num_threads)
{
    double inv_a2 = 1.0 / (a * a);
    double inv_b2 = 1.0 / (b * b);
    double inv_c2 = 1.0 / (c * c);

    omp_set_num_threads(num_threads);

    int i; // fix, windows compile error C3015: initialization in OpenMP 'for' statement has improper form
    #pragma omp parallel for schedule(static)
    for (i = 0; i < n; ++i) {
        double x = pos[i*3+0];
        double y = pos[i*3+1];
        double z = pos[i*3+2];
        double h1 = x * x * inv_a2;
        double h2 = y * y * inv_b2;
        double h3 = z * z * inv_c2;
        result[i] = pow(h1, Sa) + pow(h2, Sb) + pow(h3, Sc);
    }
}


void f_shaped_ellipsoid_jacobian_cpp(
    double a, double b, double c, double Sa, double Sb, double Sc,
    const double* pos, int n,
    double* da, double* db, double* dc,
    double* dSa, double* dSb, double* dSc,
    double* dx, double* dy, double* dz,
    int num_threads)
{
    double inv_a2 = 1.0 / (a * a);
    double inv_b2 = 1.0 / (b * b);
    double inv_c2 = 1.0 / (c * c);

    omp_set_num_threads(num_threads);

    int i;
    #pragma omp parallel for schedule(static)
    for (i = 0; i < n; ++i) {
        double x = pos[i*3+0];
        double y = pos[i*3+1];
        double z = pos[i*3+2];
        double h1 = x * x * inv_a2;
        double h2 = y * y * inv_b2;
        double h3 = z * z * inv_c2;

        double log_h1 = log(h1);
        double log_h2 = log(h2);
        double log_h3 = log(h3);

        double h1_Sa = exp(Sa * log_h1);
        double h2_Sb = exp(Sb * log_h2);
        double h3_Sc = exp(Sc * log_h3);

        da[i]  = -2.0 * h1_Sa * Sa / a;
        db[i]  = -2.0 * h2_Sb * Sb / b;
        dc[i]  = -2.0 * h3_Sc * Sc / c;

        
        dSa[i] = h1_Sa * log_h1;
        dSb[i] = h2_Sb * log_h2;
        dSc[i] = h3_Sc * log_h3;

        
        dx[i] = 2.0 * x * inv_a2 * Sa * h1_Sa / h1;
        dy[i] = 2.0 * y * inv_b2 * Sb * h2_Sb / h2;
        dz[i] = 2.0 * z * inv_c2 * Sc * h3_Sc / h3;
    }
}

// Newton's method， 1 ord
static double _ellipsoid_ray_newton(
    double d0,
    double Sa2, double ExddSa,
    double Sb2, double EyddSb,
    double Sc2, double EzddSc,
    double f)
{
    double df = Sa2 * ExddSa + Sb2 * EyddSb + Sc2 * EzddSc;
    return d0 - d0 * f / (2.0 * df);
}


// Halley's method, 2 ord
static double _ellipsoid_ray_halley(
    double d0,
    double Sa2, double ExddSa,
    double Sb2, double EyddSb,
    double Sc2, double EzddSc,
    double f)
{
    double df = Sa2 * ExddSa + Sb2 * EyddSb + Sc2 * EzddSc;
    double ddf = Sa2 * (Sa2 - 1) * ExddSa + Sb2 * (Sb2 - 1) * EyddSb + Sc2 * (Sc2 - 1) * EzddSc;
    return d0 - d0 * 2.0 * f * df / (3.0 * df * df - ddf * f);
}

// Householder's method, 3 ord
static double _ellipsoid_ray_householder(
    double d0,
    double Sa2, double ExddSa,
    double Sb2, double EyddSb,
    double Sc2, double EzddSc,
    double f)
{
    double df = Sa2 * ExddSa + Sb2 * EyddSb + Sc2 * EzddSc;
    double ddf = Sa2 * (Sa2 - 1) * ExddSa + Sb2 * (Sb2 - 1) * EyddSb + Sc2 * (Sc2 - 1) * EzddSc;
    double dddf = Sa2 * (Sa2 - 1) * (Sa2 - 2) * ExddSa
                + Sb2 * (Sb2 - 1) * (Sb2 - 2) * EyddSb
                + Sc2 * (Sc2 - 1) * (Sc2 - 2) * EzddSc;
    return d0 - d0 * (3.0 * f * (3.0 * df * df - f * ddf)) / (12.0 * df * df * df - 9.0 * f * df * ddf + f * f * dddf);
}

typedef double (*EllipsoidIterFunc)(
    double d0,
    double Sa2, double ExddSa,
    double Sb2, double EyddSb,
    double Sc2, double EzddSc,
    double f);

// Helper function for ray-ellipsoid intersection iteration
double solve_ray_shaped_ellipsoid(
    double x, double y, double z,
    double a, double b, double c,
    double Sa, double Sb, double Sc,
    int maxIterations, double epsilon, int method)
{
    double L = sqrt(x*x + y*y + z*z);
    double xi = x / L, yi = y / L, zi = z / L;
    double xi2 = xi*xi, yi2 = yi*yi, zi2 = zi*zi;

    double inv_a2 = 1.0 / (a * a);
    double inv_b2 = 1.0 / (b * b);
    double inv_c2 = 1.0 / (c * c);

    double initial_guess = (a+c+b) / 3.0;

    double ex_base = xi2 * inv_a2;
    double ey_base = yi2 * inv_b2;
    double ez_base = zi2 * inv_c2;
    // double log_ex_base = log(ex_base);
    // double log_ey_base = log(ey_base);
    // double log_ez_base = log(ez_base);

    double ex_base_Sa = pow(ex_base, Sa);
    double ey_base_Sb = pow(ey_base, Sb);
    double ez_base_Sc = pow(ez_base, Sc);

    double Sa2 = 2.0 * Sa;
    double Sb2 = 2.0 * Sb;
    double Sc2 = 2.0 * Sc;

    EllipsoidIterFunc iter_func;
    if (method == 1) {
        iter_func = _ellipsoid_ray_newton;
    } else if (method == 2) {
        iter_func = _ellipsoid_ray_halley;
    } else {
        iter_func = _ellipsoid_ray_householder;
    }


    double d0 = initial_guess, d1;
    double f, f1;
    double d2, ExddSa, EyddSb, EzddSc;

    //  double log_dd = 2.0 * log(d0); // log(d0^2)
    //  double ExddSa = exp(Sa * (log_ex_base + log_dd));
    //  double EyddSb = exp(Sb * (log_ey_base + log_dd));
    //  double EzddSc = exp(Sc * (log_ez_base + log_dd));
    d2 = d0 * d0;
    ExddSa = ex_base_Sa * pow(d2,Sa);
    EyddSb = ey_base_Sb * pow(d2,Sb);
    EzddSc = ez_base_Sc * pow(d2,Sc);

    f = ExddSa + EyddSb + EzddSc - 1.0;

    const int MAX_BACKTRACK = 10;


    for (int it = 0; it < maxIterations; ++it) {

        // Use the selected iterative method (Newton/Halley/Householder) to compute the next estimate d1
        d1 = iter_func(d0, Sa2, ExddSa, Sb2, EyddSb, Sc2, EzddSc, f);

        // Update function values at new estimate
        d2 = d1 * d1;
        ExddSa = ex_base_Sa * pow(d2, Sa);
        EyddSb = ey_base_Sb * pow(d2, Sb);
        EzddSc = ez_base_Sc * pow(d2, Sc);
        f1 = ExddSa + EyddSb + EzddSc - 1.0;

        // Adaptive step size: if the new function value is worse, reduce the step size
        double step_size = 1.0;
        int backtrack_count = 0;
        double delta_d = d1 - d0;

        while (fabs(f1) > fabs(f) && backtrack_count < MAX_BACKTRACK) {
            step_size *= 0.5;
            d1 = d0 + delta_d * step_size; // Reduce step size
            
            // Recalculate function value at the new estimate
            d2 = d1 * d1;
            ExddSa = ex_base_Sa * pow(d2, Sa);
            EyddSb = ey_base_Sb * pow(d2, Sb);
            EzddSc = ez_base_Sc * pow(d2, Sc);
            f1 = ExddSa + EyddSb + EzddSc - 1.0;
            
            backtrack_count++;
        }

        // Update current estimate and function value
        d0 = d1;
        f = f1;

        // Check for convergence
        if (fabs(f) < epsilon) break;

    }
    return d0;
}


void IntersectRaysEllipsoid_S_cpp(
    double a, double b, double c, double Sa, double Sb, double Sc,
    const double* pos, int n, int maxIterations,
    double* tarpos, double* result, int method, int num_threads)
{
    double epsilon = 1e-7;

    omp_set_num_threads(num_threads);

    int i;
    #pragma omp parallel for
    for (i = 0; i < n; ++i) {
        double x = pos[i*3+0], y = pos[i*3+1], z = pos[i*3+2];
        double L = sqrt(x*x + y*y + z*z);
        double xi = x / L, yi = y / L, zi = z / L;

        double d0 = solve_ray_shaped_ellipsoid(
            x, y, z,
            a, b, c,
            Sa, Sb, Sc,
            maxIterations, epsilon, method);

        tarpos[i*3+0] = d0 * xi;
        tarpos[i*3+1] = d0 * yi;
        tarpos[i*3+2] = d0 * zi;
        result[i] = L - d0;
    }
}

void f_ray_shaped_ellipsoid_cpp(
    double a, double b, double c, double Sa, double Sb, double Sc,
    const double* pos, int n, int maxIterations,
    double* result, int method, int num_threads)
{
    double epsilon = 1e-7;

    omp_set_num_threads(num_threads);

    int i;
    #pragma omp parallel for
    for (i = 0; i < n; ++i) {
        double x = pos[i*3+0], y = pos[i*3+1], z = pos[i*3+2];
        double L = sqrt(x*x + y*y + z*z);

        double d0 = solve_ray_shaped_ellipsoid(
            x, y, z,
            a, b, c,
            Sa, Sb, Sc,
            maxIterations, epsilon, method);

        result[i] = L / d0;
    }
}


struct EllipsoidIntersectResult {
    double t;
    double f;
};

// Helper for Newton-Raphson intersection along a line segment
struct EllipsoidIntersectResult newton_intersect_ellipsoid(
    double x1, double y1, double z1,
    double vx, double vy, double vz,
    double inv_a2, double inv_b2, double inv_c2,
    double Sa, double Sb, double Sc,
    double t_init, double delta_cut, double epsilon, int maxIterations)
{   

    double t = t_init;
    double posix, posiy, posiz, Ex, Ey, Ez, f, df_x, df_y, df_z, df, delta;
    int iter_count = 0;
    for (; iter_count < maxIterations; ++iter_count) {

        // Calculate position at current t
        posix = x1 + t * vx;
        posiy = y1 + t * vy;
        posiz = z1 + t * vz;

        // Calculate ellipsoid function terms
        Ex = pow((posix * posix) * inv_a2, Sa);
        Ey = pow((posiy * posiy) * inv_b2, Sb);
        Ez = pow((posiz * posiz) * inv_c2, Sc);

        // Check if converged
        f = Ex + Ey + Ez - 1.0;
        if (fabs(f) < epsilon) break;

        // Calculate gradient components
        df_x = (posix == 0) ? 0.0 : Ex * Sa * vx / posix;
        df_y = (posiy == 0) ? 0.0 : Ey * Sb * vy / posiy;
        df_z = (posiz == 0) ? 0.0 : Ez * Sc * vz / posiz;
        df = 4.0 * (df_x + df_y + df_z);

        // Calculate step
        delta = -f / df;

        // apply adaptive limit when near target pos
        if (f < 2.0) {
            if (delta > delta_cut) delta = delta_cut;
            if (delta < -delta_cut) delta = -delta_cut;
        }
        t += delta;
       // if (fabs(delta) < epsilon) break; // Do not use delta to check for convergence; sometimes delta meets the convergence condition but f does not.
    }
    /* Since the convergence check no longer uses delta, there is no need to update f again here 
    posix = x1 + t * vx;
    posiy = y1 + t * vy;
    posiz = z1 + t * vz;
    Ex = pow((posix * posix) * inv_a2, Sa);
    Ey = pow((posiy * posiy) * inv_b2, Sb);
    Ez = pow((posiz * posiz) * inv_c2, Sc);
    f = Ex + Ey + Ez - 1.0;
    */

    struct EllipsoidIntersectResult res;
    res.t = t;
    res.f = f;
    return res;
}

void IntersectLinesEllipsoid_S_cpp(
    double a, double b, double c, double Sa, double Sb, double Sc,
    const double* pos1, const double* pos2, int n, int maxIterations,
    double* ts, int num_threads)
{
    double inv_a2 = 1.0 / (a * a);
    double inv_b2 = 1.0 / (b * b);
    double inv_c2 = 1.0 / (c * c);
    double delta_cut = c / 2.0;
    double epsilon = 1e-9;
    double res_check = 1e-7;

    omp_set_num_threads(num_threads);

    int i;
    #pragma omp parallel for schedule(static)
    for (i = 0; i < n; ++i) {
        // pos1, pos2: [n, 3] row-major
        double x1 = pos1[i*3+0], y1 = pos1[i*3+1], z1 = pos1[i*3+2];
        double x2 = pos2[i*3+0], y2 = pos2[i*3+1], z2 = pos2[i*3+2];
        double vx = x2 - x1, vy = y2 - y1, vz = z2 - z1;
        double vlen = sqrt(vx*vx + vy*vy + vz*vz);
        if (vlen == 0) {
            ts[i*2+0] = -1;
            ts[i*2+1] = -1;
            continue;
        }
        vx /= vlen; vy /= vlen; vz /= vlen;
        double tmax = vlen;

        // First intersection (t0)
        struct EllipsoidIntersectResult res0 = newton_intersect_ellipsoid(
                        x1, y1, z1, vx, vy, vz,
                        inv_a2, inv_b2, inv_c2,
                        Sa, Sb, Sc,
                        epsilon, delta_cut, epsilon, maxIterations);

        // Second intersection (t1)
        struct EllipsoidIntersectResult res1 = newton_intersect_ellipsoid(
            x1, y1, z1, vx, vy, vz,
            inv_a2, inv_b2, inv_c2,
            Sa, Sb, Sc,
            tmax, delta_cut, epsilon, maxIterations);

        // Intersection result logic
        if (fabs(res0.t - res1.t) <= res_check) {
            if ((fabs(res0.f) < res_check) || (fabs(res1.f) < res_check)) {
                // Single intersection point (tangent)
                ts[i*2+0] = res0.t;
                ts[i*2+1] = res1.t;
            }
        } else if ((fabs(res0.f) <= res_check) && (fabs(res1.f) <= res_check)) {
            // Two distinct intersection points
            ts[i*2+0] = res0.t;
            ts[i*2+1] = res1.t;
        } else if (fabs(res0.f) <= res_check) {
            // First point converged, second didn't
            ts[i*2+0] = res0.t;
            ts[i*2+1] = res0.t + 0.6 * (res1.t - res0.t);
        } else if (fabs(res1.f) <= res_check) {
            // Second point converged, first didn't
            ts[i*2+1] = res1.t;
            ts[i*2+0] = res0.t + 0.4 * (res1.t - res0.t);
        } else {
            // else default is -1, -1
            ts[i*2+0] = -1;
            ts[i*2+1] = -1;
        }
    }
}
