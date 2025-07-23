#include <cmath>

const double INV_PI = 0.31830988618379067154;

inline double cubic_spline_kernel(double r_over_h2) {
    double r_over_h = std::sqrt(r_over_h2);
    double rs = 2.0 - r_over_h;
    if (rs < 0) {
        rs = 0.0;
    } else if (r_over_h2 < 1.0) {
        rs = (1.0 - 0.75 * rs * r_over_h2);
    } else {
        rs = 0.25 * rs * rs * rs;
    }
    return rs;
}

double calc_sph_density(
    int i, int num_near,
    const double* n_d,
    const int* n_index,
    const double* mass,
    const double* hsm
) {
    double dens = 0.0;
    for (int j = 0; j < num_near; ++j) {
        int idx = i * num_near + j;
        int ni = n_index[idx];
        double h = hsm[ni];
        double r = n_d[idx];
        if (r > 2.0 * h) continue;
        double inv_h = 1.0 / h;
        double inv_h2 = inv_h * inv_h;
        double fNorm = INV_PI * inv_h * inv_h2;
        double r_over_h = r * inv_h;
        double r_over_h2 = r_over_h * r_over_h;
        dens += mass[ni] * cubic_spline_kernel(r_over_h2) * fNorm;
    }
    return dens;
}