#include <cmath>

const double COEF = 5.092958178940650744;  // 16/pi  0.318309886183790671 5.092958178940650744
const double INV_PI = 0.318309886183790671;
const double NORM_V = 0.23873241463784303;  // 1/(4/3*np.pi)

inline double cubic_spline_kernel_2(double r) {
    double rs = 2 - r;
    if (rs < 0) 
        rs = 0;
    else if (rs < 1)
        rs = (1.0 - 0.75 * rs * r * r);
    else
        rs = 0.25 * rs * rs * rs;
    return rs;
}


inline double cubic_spline_kernel_1(double r) {
    double rs = 1 - r;
    if (rs < 0) 
        rs = 0;
    else if (rs < 0.5)
        rs = (0.5 - 3.0 * r * r * rs);
    else
        rs = rs * rs * rs;
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

    int id = i * num_near;

    double h = n_d[id + num_near - 1];

    double inv_h = 1.0 / h;
    double inv_h2 = inv_h * inv_h;
    double fNorm = COEF * inv_h * inv_h2;

    for (int j = 0; j < num_near; ++j) {
        int idx = id + j;

        int ni = n_index[idx];
        double ri = n_d[idx];
        double hi = hsm[ni];
        if (ri > hi) continue;

        double inv_hi = 1.0 / hi;
        double inv_h2i = inv_hi * inv_hi;
        double fNormi = COEF * inv_hi * inv_h2i;

        double r_over_h = ri * inv_hi;
        dens += mass[ni] * cubic_spline_kernel_1(r_over_h) * fNormi;

    }

    return dens;
}