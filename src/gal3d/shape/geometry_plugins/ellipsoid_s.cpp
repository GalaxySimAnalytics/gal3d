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

        double h1_Sa = std::pow(h1, Sa);
        double h2_Sb = std::pow(h2, Sb);
        double h3_Sc = std::pow(h3, Sc);

        
        da[i]  = -2.0 * h1_Sa * Sa / a;
        db[i]  = -2.0 * h2_Sb * Sb / b;
        dc[i]  = -2.0 * h3_Sc * Sc / c;

        
        dSa[i] = h1_Sa * std::log(h1);
        dSb[i] = h2_Sb * std::log(h2);
        dSc[i] = h3_Sc * std::log(h3);

        
        dx[i] = 2.0 * x * inv_a2 * Sa * std::pow(h1, Sa - 1.0);
        dy[i] = 2.0 * y * inv_b2 * Sb * std::pow(h2, Sb - 1.0);
        dz[i] = 2.0 * z * inv_c2 * Sc * std::pow(h3, Sc - 1.0);
    }
}

extern "C" void IntersectRaysEllipsoid_S_cpp(
    double a, double b, double c, double Sa, double Sb, double Sc,
    const double* pos, int n, int maxIterations,
    double* tarpos, double* result, int num_threads)
{
    double inv_a2 = 1.0 / (a * a);
    double inv_b2 = 1.0 / (b * b);
    double inv_c2 = 1.0 / (c * c);
    double epsilon = 1e-7;
    double initial_guess = (a + c) / 2.0;

    omp_set_num_threads(num_threads);

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        double x = pos[i*3+0], y = pos[i*3+1], z = pos[i*3+2];
        // Calculate ray direction
        double L = std::sqrt(x*x + y*y + z*z);
        double xi = x / L, yi = y / L, zi = z / L;
        double xi2 = xi*xi, yi2 = yi*yi, zi2 = zi*zi;

        double ex_base = xi2 * inv_a2;
        double ey_base = yi2 * inv_b2;
        double ez_base = zi2 * inv_c2;
        double log_ex_base = std::log(ex_base);
        double log_ey_base = std::log(ey_base);
        double log_ez_base = std::log(ez_base);

        // Iteratively solve for intersection
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
    double inv_a2 = 1.0 / (a * a);
    double inv_b2 = 1.0 / (b * b);
    double inv_c2 = 1.0 / (c * c);
    double epsilon = 1e-7;
    double initial_guess = (a + c) / 2.0;

    omp_set_num_threads(num_threads);

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        double x = pos[i*3+0], y = pos[i*3+1], z = pos[i*3+2];
        double L = std::sqrt(x*x + y*y + z*z);

        double xi = x / L, yi = y / L, zi = z / L;
        double xi2 = xi*xi, yi2 = yi*yi, zi2 = zi*zi;

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
        result[i] = L / d0;
    }
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
        double t0 = epsilon;

        // Calculate position at current t
        double posix = x1 + t0 * vx, posiy = y1 + t0 * vy, posiz = z1 + t0 * vz;
        double Ex, Ey, Ez, f0, df_x, df_y, df_z, df, delta;
        int iter_count = 0;

        // Newton-Raphson iteration for first intersection
        while (true) {
            // Calculate ellipsoid function terms
            Ex = std::pow((posix * posix) * inv_a2, Sa);
            Ey = std::pow((posiy * posiy) * inv_b2, Sb);
            Ez = std::pow((posiz * posiz) * inv_c2, Sc);
            f0 = Ex + Ey + Ez - 1.0;

            // Check if we've converged
            if (std::abs(f0) < epsilon) break;

            // Calculate gradient components with robust handling of zero values
            df_x = (posix == 0) ? 0.0 : Ex * Sa * vx / posix;
            df_y = (posiy == 0) ? 0.0 : Ey * Sb * vy / posiy;
            df_z = (posiz == 0) ? 0.0 : Ez * Sc * vz / posiz;
            df = 4.0 * (df_x + df_y + df_z);

            // Calculate step and apply adaptive limit
            delta = -f0 / df;
            if (f0 < 2.0) {// when near target pos
                delta = std::min(delta_cut, std::max(-delta_cut, delta));
            }
            t0 += delta;
            posix = x1 + t0 * vx;
            posiy = y1 + t0 * vy;
            posiz = z1 + t0 * vz;
            iter_count++;
            if (std::abs(delta) < epsilon || iter_count > maxIterations) break;
        }
        // Update f0 after loop
        Ex = std::pow((posix * posix) * inv_a2, Sa);
        Ey = std::pow((posiy * posiy) * inv_b2, Sb);
        Ez = std::pow((posiz * posiz) * inv_c2, Sc);
        f0 = Ex + Ey + Ez - 1.0;

        // Second intersection (t1)
        double t1 = tmax;
        posix = x1 + t1 * vx; posiy = y1 + t1 * vy; posiz = z1 + t1 * vz;
        iter_count = 0;
        double f1;
        while (true) {
            Ex = std::pow((posix * posix) * inv_a2, Sa);
            Ey = std::pow((posiy * posiy) * inv_b2, Sb);
            Ez = std::pow((posiz * posiz) * inv_c2, Sc);
            f1 = Ex + Ey + Ez - 1.0;
            if (std::abs(f1) < epsilon) break;
            df_x = (posix == 0) ? 0.0 : Ex * Sa * vx / posix;
            df_y = (posiy == 0) ? 0.0 : Ey * Sb * vy / posiy;
            df_z = (posiz == 0) ? 0.0 : Ez * Sc * vz / posiz;
            df = 4.0 * (df_x + df_y + df_z);
            delta = -f1 / df;
            if (f1 < 2.0) {// When near the surface
                delta = std::min(delta_cut, std::max(-delta_cut, delta));
            }
            t1 += delta;
            posix = x1 + t1 * vx;
            posiy = y1 + t1 * vy;
            posiz = z1 + t1 * vz;
            iter_count++;
            if (std::abs(delta) < epsilon || iter_count > maxIterations) break;
        }
        // Update f1 after loop
        Ex = std::pow((posix * posix) * inv_a2, Sa);
        Ey = std::pow((posiy * posiy) * inv_b2, Sb);
        Ez = std::pow((posiz * posiz) * inv_c2, Sc);
        f1 = Ex + Ey + Ez - 1.0;

        // Intersection result logic
        if (std::abs(t0 - t1) <= ten_epsilon) {
            if ((std::abs(f0) < ten_epsilon) || (std::abs(f1) < ten_epsilon)) {
                // Single intersection point (tangent)
                ts[i*2+0] = t0;
                ts[i*2+1] = t1;
            }
        } else if ((std::abs(f0) <= ten_epsilon) && (std::abs(f1) <= ten_epsilon)) {
            // Two distinct intersection points
            ts[i*2+0] = t0;
            ts[i*2+1] = t1;
        } else if (std::abs(f0) <= ten_epsilon) {
            // First point converged, second didn't
            ts[i*2+0] = t0;
            ts[i*2+1] = t0 + 0.6 * (t1 - t0);
        } else if (std::abs(f1) <= ten_epsilon) {
            // Second point converged, first didn't
            ts[i*2+1] = t1;
            ts[i*2+0] = t0 + 0.4 * (t1 - t0);
        } else {
            // else default is -1, -1
            ts[i*2+0] = -1;
            ts[i*2+1] = -1;
        }
    }
}
