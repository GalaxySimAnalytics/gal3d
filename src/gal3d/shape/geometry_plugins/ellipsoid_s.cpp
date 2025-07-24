#include "ellipsoid_s.hpp"
#include <cmath>
#include <algorithm>
#include <omp.h>


extern "C" void f_shaped_ellipsoid_cpp(
    double a, double b, double c, double Sa, double Sb, double Sc,
    const double* pos, int n, double* result, int num_threads)
{
    double inv_a2 = 1.0 / (a * a);
    double inv_b2 = 1.0 / (b * b);
    double inv_c2 = 1.0 / (c * c);

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        double x = pos[i*3+0];
        double y = pos[i*3+1];
        double z = pos[i*3+2];
        double h1 = x * x * inv_a2;
        double h2 = y * y * inv_b2;
        double h3 = z * z * inv_c2;
        result[i] = std::pow(h1, Sa) + std::pow(h2, Sb) + std::pow(h3, Sc);
    }
}


extern "C" void f_shaped_ellipsoid_jacobian_cpp(
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

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        double x = pos[i*3+0];
        double y = pos[i*3+1];
        double z = pos[i*3+2];
        double h1 = x * x * inv_a2;
        double h2 = y * y * inv_b2;
        double h3 = z * z * inv_c2;

        double log_h1 = std::log(h1);
        double log_h2 = std::log(h2);
        double log_h3 = std::log(h3);

        double h1_Sa = std::exp(Sa * log_h1);
        double h2_Sb = std::exp(Sb * log_h2);
        double h3_Sc = std::exp(Sc * log_h3);

        
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


// Helper function for ray-ellipsoid intersection Newton iteration
inline double solve_ray_shaped_ellipsoid(
    double x, double y, double z,
    double a, double b, double c,
    double Sa, double Sb, double Sc,
    int maxIterations, double epsilon)
{
    double L = std::sqrt(x*x + y*y + z*z);
    double xi = x / L, yi = y / L, zi = z / L;
    double xi2 = xi*xi, yi2 = yi*yi, zi2 = zi*zi;

    double inv_a2 = 1.0 / (a * a);
    double inv_b2 = 1.0 / (b * b);
    double inv_c2 = 1.0 / (c * c);

    double initial_guess = (a + c) / 2.0;

    double ex_base = xi2 * inv_a2;
    double ey_base = yi2 * inv_b2;
    double ez_base = zi2 * inv_c2;
    double log_ex_base = std::log(ex_base);
    double log_ey_base = std::log(ey_base);
    double log_ez_base = std::log(ez_base);

    double d0 = initial_guess, d1;
    for (int it = 0; it < maxIterations; ++it) {
        double log_dd = 2.0 * std::log(d0); // log(d0^2)
        double ExddSa = std::exp(Sa * (log_ex_base + log_dd));
        double EyddSb = std::exp(Sb * (log_ey_base + log_dd));
        double EzddSc = std::exp(Sc * (log_ez_base + log_dd));
        double f = ExddSa + EyddSb + EzddSc - 1.0;
        double df = 2.0 * (Sa * ExddSa + Sb * EyddSb + Sc * EzddSc) / d0;
        d1 = d0 - f / df;
        if (std::abs(d1 - d0) < epsilon) break;
        d0 = d1;
    }
    return d0;
}


extern "C" void IntersectRaysEllipsoid_S_cpp(
    double a, double b, double c, double Sa, double Sb, double Sc,
    const double* pos, int n, int maxIterations,
    double* tarpos, double* result, int num_threads)
{
    double epsilon = 1e-7;

    omp_set_num_threads(num_threads);

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        double x = pos[i*3+0], y = pos[i*3+1], z = pos[i*3+2];
        double L = std::sqrt(x*x + y*y + z*z);
        double xi = x / L, yi = y / L, zi = z / L;

        double d0 = solve_ray_shaped_ellipsoid(
            x, y, z,
            a, b, c,
            Sa, Sb, Sc,
            maxIterations, epsilon);

        tarpos[i*3+0] = d0 * xi;
        tarpos[i*3+1] = d0 * yi;
        tarpos[i*3+2] = d0 * zi;
        result[i] = L - d0;
    }
}

extern "C" void f_ray_shaped_ellipsoid_cpp(
    double a, double b, double c, double Sa, double Sb, double Sc,
    const double* pos, int n, int maxIterations,
    double* result, int num_threads)
{
    double epsilon = 1e-7;

    omp_set_num_threads(num_threads);

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        double x = pos[i*3+0], y = pos[i*3+1], z = pos[i*3+2];
        double L = std::sqrt(x*x + y*y + z*z);
        double xi = x / L, yi = y / L, zi = z / L;

        double d0 = solve_ray_shaped_ellipsoid(
            x, y, z,
            a, b, c,
            Sa, Sb, Sc,
            maxIterations, epsilon);

        result[i] = L / d0;
    }
}


struct EllipsoidIntersectResult {
    double t;
    double f;
};

// Helper for Newton-Raphson intersection along a line segment
inline EllipsoidIntersectResult newton_intersect_ellipsoid(
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
        Ex = std::pow((posix * posix) * inv_a2, Sa);
        Ey = std::pow((posiy * posiy) * inv_b2, Sb);
        Ez = std::pow((posiz * posiz) * inv_c2, Sc);

        // Check if converged
        f = Ex + Ey + Ez - 1.0;
        if (std::abs(f) < epsilon) break;

        // Calculate gradient components
        df_x = (posix == 0) ? 0.0 : Ex * Sa * vx / posix;
        df_y = (posiy == 0) ? 0.0 : Ey * Sb * vy / posiy;
        df_z = (posiz == 0) ? 0.0 : Ez * Sc * vz / posiz;
        df = 4.0 * (df_x + df_y + df_z);

        // Calculate step
        delta = -f / df;

        // apply adaptive limit when near target pos
        if (f < 2.0) {
            delta = std::min(delta_cut, std::max(-delta_cut, delta));
        }
        t += delta;
        if (std::abs(delta) < epsilon) break;
    }
    posix = x1 + t * vx;
    posiy = y1 + t * vy;
    posiz = z1 + t * vz;
    Ex = std::pow((posix * posix) * inv_a2, Sa);
    Ey = std::pow((posiy * posiy) * inv_b2, Sb);
    Ez = std::pow((posiz * posiz) * inv_c2, Sc);
    f = Ex + Ey + Ez - 1.0;

    return {t, f};
}

extern "C" void IntersectLinesEllipsoid_S_cpp(
    double a, double b, double c, double Sa, double Sb, double Sc,
    const double* pos1, const double* pos2, int n, int maxIterations,
    double* ts, int num_threads)
{
    double inv_a2 = 1.0 / (a * a);
    double inv_b2 = 1.0 / (b * b);
    double inv_c2 = 1.0 / (c * c);
    double delta_cut = c / 2.0;
    double epsilon = 1e-9;
    double ten_epsilon = 1e-8;

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        // pos1, pos2: [n, 3] row-major
        double x1 = pos1[i*3+0], y1 = pos1[i*3+1], z1 = pos1[i*3+2];
        double x2 = pos2[i*3+0], y2 = pos2[i*3+1], z2 = pos2[i*3+2];
        double vx = x2 - x1, vy = y2 - y1, vz = z2 - z1;
        double vlen = std::sqrt(vx*vx + vy*vy + vz*vz);
        if (vlen == 0) {
            ts[i*2+0] = -1;
            ts[i*2+1] = -1;
            continue;
        }
        vx /= vlen; vy /= vlen; vz /= vlen;
        double tmax = vlen;

        // First intersection (t0)
        EllipsoidIntersectResult res0 = newton_intersect_ellipsoid(
                        x1, y1, z1, vx, vy, vz,
                        inv_a2, inv_b2, inv_c2,
                        Sa, Sb, Sc,
                        epsilon, delta_cut, epsilon, maxIterations);

        // Second intersection (t1)
        EllipsoidIntersectResult res1 = newton_intersect_ellipsoid(
            x1, y1, z1, vx, vy, vz,
            inv_a2, inv_b2, inv_c2,
            Sa, Sb, Sc,
            tmax, delta_cut, epsilon, maxIterations);

        // Intersection result logic
        if (std::abs(res0.t - res1.t) <= ten_epsilon) {
            if ((std::abs(res0.f) < ten_epsilon) || (std::abs(res1.f) < ten_epsilon)) {
                // Single intersection point (tangent)
                ts[i*2+0] = res0.t;
                ts[i*2+1] = res1.t;
            }
        } else if ((std::abs(res0.f) <= ten_epsilon) && (std::abs(res1.f) <= ten_epsilon)) {
            // Two distinct intersection points
            ts[i*2+0] = res0.t;
            ts[i*2+1] = res1.t;
        } else if (std::abs(res0.f) <= ten_epsilon) {
            // First point converged, second didn't
            ts[i*2+0] = res0.t;
            ts[i*2+1] = res0.t + 0.6 * (res1.t - res0.t);
        } else if (std::abs(res1.f) <= ten_epsilon) {
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
