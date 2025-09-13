#include "ellipsoid_s.h"
#include <math.h>
#include <omp.h>


void f_shaped_ellipsoid_cpp(
    double a, double b, double c, double Sa, double Sb, double Sc,
    const double* pos, int n, double* result, double* r, int num_threads)
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
        double x2 = x * x;
        double y2 = y * y;
        double z2 = z * z;
        double h1 = (x == 0) ? 0.0 : exp(Sa * log(x2 * inv_a2));
        double h2 = (y == 0) ? 0.0 : exp(Sb * log(y2 * inv_b2));
        double h3 = (z == 0) ? 0.0 : exp(Sc * log(z2 * inv_c2));
        result[i] = h1 + h2 + h3;
        r[i] = sqrt(x2 + y2 + z2);
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
    double xi, double yi, double zi,
    double inv_a2, double inv_b2, double inv_c2,
    double Sa, double Sb, double Sc, double initial_guess,
    int maxIterations, double epsilon, EllipsoidIterFunc iter_func)
{
    double xi2 = xi*xi, yi2 = yi*yi, zi2 = zi*zi;


    double log_ex_base = log(xi2 * inv_a2);
    double log_ey_base = log(yi2 * inv_b2);
    double log_ez_base = log(zi2 * inv_c2);

    double Sa_log_ex_base = Sa * log_ex_base;
    double Sb_log_ey_base = Sb * log_ey_base;
    double Sc_log_ez_base = Sc * log_ez_base;

    double Sa2 = 2.0 * Sa;
    double Sb2 = 2.0 * Sb;
    double Sc2 = 2.0 * Sc;

    double d0 = initial_guess, d1;
    double f, f1;
    double d2, ExddSa, EyddSb, EzddSc;


    // avoid pow, for acceleration
    double log_d = log(d0);
    ExddSa = exp(Sa_log_ex_base + Sa2*log_d);
    EyddSb = exp(Sb_log_ey_base + Sb2*log_d);
    EzddSc = exp(Sc_log_ez_base + Sc2*log_d);

    f = ExddSa + EyddSb + EzddSc - 1.0;

    const int MAX_BACKTRACK = 6;


    for (int it = 0; it < maxIterations; ++it) {

        // Use the selected iterative method (Newton/Halley/Householder) to compute the next estimate d1
        d1 = iter_func(d0, Sa2, ExddSa, Sb2, EyddSb, Sc2, EzddSc, f);

        // Update function values at new estimate
        log_d = log(d1);
        ExddSa = exp(Sa_log_ex_base + Sa2*log_d);
        EyddSb = exp(Sb_log_ey_base + Sb2*log_d);
        EzddSc = exp(Sc_log_ez_base + Sc2*log_d);
        f1 = ExddSa + EyddSb + EzddSc - 1.0;

        // Adaptive step size: if the new function value is worse, change the step size
        double step_size = 0.5;
        int backtrack_count = 0;
        double delta_d = d1 - d0;

        while (fabs(f1) > fabs(f) && backtrack_count < MAX_BACKTRACK) {
            d1 = d0 + delta_d * step_size; // Reduce step size
            
            // Recalculate function value at the new estimate
            log_d = log(d1);
            ExddSa = exp(Sa_log_ex_base + Sa2*log_d);
            EyddSb = exp(Sb_log_ey_base + Sb2*log_d);
            EzddSc = exp(Sc_log_ez_base + Sc2*log_d);
            f1 = ExddSa + EyddSb + EzddSc - 1.0;
            step_size *= 0.5;
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
    double* tarpos, double* result, double* r, int method, int num_threads)
{
    double epsilon = 1e-7;

    omp_set_num_threads(num_threads);
    EllipsoidIterFunc iter_func;
    if (method == 0){
        if (fabs(Sa - 1.0) < 0.1 && fabs(Sb - 1.0) < 0.1 && fabs(Sc - 1.0) < 0.1) {
            iter_func = _ellipsoid_ray_newton; // Newton for near-spherical ellipsoids
        } else {
            iter_func = _ellipsoid_ray_halley; // otherwise use Halley
        }
    } else if (method == 1) {
        iter_func = _ellipsoid_ray_newton;
    } else if (method == 2) {
        iter_func = _ellipsoid_ray_halley;
    } else {
        iter_func = _ellipsoid_ray_householder;
    }

    double inv_a2 = 1.0 / (a * a);
    double inv_b2 = 1.0 / (b * b);
    double inv_c2 = 1.0 / (c * c);

    double initial_guess = (a+c) / 2.0;

    int i;
    #pragma omp parallel for schedule(dynamic)
    for (i = 0; i < n; ++i) {
        double x = pos[i*3+0], y = pos[i*3+1], z = pos[i*3+2];
        double L = sqrt(x*x + y*y + z*z);
        double xi = x / L, yi = y / L, zi = z / L;

        double d0 = solve_ray_shaped_ellipsoid(
            xi, yi, zi,
            inv_a2, inv_b2, inv_c2,
            Sa, Sb, Sc, initial_guess,
            maxIterations, epsilon, iter_func);

        tarpos[i*3+0] = d0 * xi;
        tarpos[i*3+1] = d0 * yi;
        tarpos[i*3+2] = d0 * zi;
        result[i] = L - d0;
        r[i] = L;
    }
}

void f_ray_shaped_ellipsoid_cpp(
    double a, double b, double c, double Sa, double Sb, double Sc,
    const double* pos, int n, int maxIterations,
    double* result, double* r, int method, int num_threads)
{
    double epsilon = 1e-7;

    omp_set_num_threads(num_threads);

    EllipsoidIterFunc iter_func;
    if (method == 1) {
        iter_func = _ellipsoid_ray_newton;
    } else if (method == 2) {
        iter_func = _ellipsoid_ray_halley;
    } else {
        iter_func = _ellipsoid_ray_householder;
    }

    double inv_a2 = 1.0 / (a * a);
    double inv_b2 = 1.0 / (b * b);
    double inv_c2 = 1.0 / (c * c);

    double initial_guess = (a+c) / 2.0;

    int i;
    #pragma omp parallel for schedule(dynamic)
    for (i = 0; i < n; ++i) {
        double x = pos[i*3+0], y = pos[i*3+1], z = pos[i*3+2];
        double L = sqrt(x*x + y*y + z*z);
        double xi = x / L, yi = y / L, zi = z / L;


        double d0 = solve_ray_shaped_ellipsoid(
            xi, yi, zi,
            inv_a2, inv_b2, inv_c2,
            Sa, Sb, Sc, initial_guess,
            maxIterations, epsilon, iter_func);

        result[i] = L / d0;
        r[i] = L;
    }
}

void area_factor_cpp(
    double a, double b, double c, double Sa, double Sb, double Sc,
    const double* pos, int n, double* result, int num_threads)
{
    double Sa2 = 2*Sa;
    double Sb2 = 2*Sb;
    double Sc2 = 2*Sc;
    double Ex = 1./ pow(a,Sa2);
    double Ey = 1./ pow(b,Sb2);
    double Ez = 1./ pow(c,Sc2);

    double epsilon = 1e-10;

    omp_set_num_threads(num_threads);


    int i;
    #pragma omp parallel for schedule(static)
    for (i = 0; i < n; ++i) {
        double x = pos[i*3+0], y = pos[i*3+1], z = pos[i*3+2];
        double x2 = x*x, y2 = y*y, z2 = z*z;
        double L2 = x2+y2+z2;
        double L = sqrt(L2);
        double logL = log(L); // for pow
        double xi = x / L, yi = y / L, zi = z / L;
        double xi2 = xi * xi, yi2 = yi*yi, zi2 = zi*zi;


        double cos_theta = zi;
        double sin_theta = sqrt(1 - zi2);
        double cos_phi = xi / sqrt(xi2 + yi2);
        double sin_phi = yi / sqrt(xi2 + yi2);


        double alpha_x = Ex * exp(Sa * log(xi2));  // pow(a,b) = exp(log(a)*b), if a > 0
        double alpha_y = Ey * exp(Sb * log(yi2));
        double alpha_z = Ez * exp(Sc * log(zi2));

        double F_r = Sa2*exp(logL* (Sa2 -1))*alpha_x +Sb2*exp(logL * (Sb2 -1))*alpha_y + Sc2*exp(logL* (Sc2 -1))*alpha_z;

        double coef_x = exp(logL * Sa2)*Sa2*Ex*exp(log(xi2)* (Sa - 1))*xi;
        double coef_y = exp(logL * Sb2)*Sb2*Ey*exp(log(yi2)* (Sb - 1))*yi;
        double coef_z = exp(logL * Sc2)*Sc2*Ez*exp(log(zi2)* (Sc - 1))*zi;

        double F_theta = coef_x*cos_theta*cos_phi+
                        coef_y*cos_theta*sin_phi+
                        coef_z*(-sin_theta);
        double F_phi = coef_x*(-sin_theta*sin_phi)+
                       coef_y*(sin_theta*cos_phi)+
                       coef_z*(0.);

        double r_theta = - F_theta/(F_r+epsilon);   // avoid 
        double r_phi = - F_phi/(F_r+epsilon);

        double area_factor = sqrt(1 + (r_theta*r_theta + r_phi*r_phi/(sin_theta*sin_theta + epsilon))/(L2));

        result[i] = area_factor;

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
        Ex = (posix == 0) ? 0.0 : exp(Sa * log((posix * posix) * inv_a2));
        Ey = (posiy == 0) ? 0.0 : exp(Sb * log((posiy * posiy) * inv_b2));
        Ez = (posiz == 0) ? 0.0 : exp(Sc * log((posiz * posiz) * inv_c2));

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
    #pragma omp parallel for schedule(dynamic)
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
