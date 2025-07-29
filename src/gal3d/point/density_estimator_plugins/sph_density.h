#include <cmath>

const double COEF1 = 5.092958178940650744;  // 16/pi  0.318309886183790671 5.092958178940650744
const double COEF2 = 0.318309886183790671; // 1/pi

inline double cubic_spline_kernel_2(double r) {
    double rs = 2 - r;

    if ((r < 0) || (r > 2.0))
        return 0;

    if (r < 1.0) {
        return  (1.0 - 0.75 * rs * r * r);
    } else {
        return  0.25 * rs * rs * rs;
    }
}


inline double cubic_spline_kernel_1(double r) {
    double rs = 1.0 - r;

    if ((r < 0) || (r > 1.0))
        return 0;
    if (r<0.5){
        return  (0.5 - 3.0 * r * r * rs);
    } else {
        return  rs * rs * rs;
    }
}

double calc_sph_density(
    int i, int num_near,
    const double* n_d,
    const int* n_index,
    const double* mass,
    const double* hsm
) {
    double dens = 0.0;

    int id = i * num_near;

    double h = n_d[id + num_near - 1];

    double inv_h = 1.0 / h;
    double inv_h2 = inv_h * inv_h;
    double fNorm = inv_h * inv_h2 * COEF1;

    for (int j = 0; j < num_near; ++j) {
        int idx = id + j;

        int ni = n_index[idx];
        double ri = n_d[idx];
        double r_over_h = ri * inv_h;
        dens += mass[ni] * cubic_spline_kernel_1(r_over_h);
    }
    dens *= fNorm;

    return dens;
}