#include <cmath>

const double COEF1 = 5.092958178940650744;  // 16/pi  0.318309886183790671 5.092958178940650744
const double COEF2 = 0.318309886183790671; // 1/pi

inline double cubic_spline_kernel_region2(double r_over_h) {
    double rs = 2 - r_over_h;

    if ((r_over_h < 0) || (r_over_h > 2.0))
        return 0;

    if (r_over_h < 1.0) {
        return  (1.0 - 0.75 * rs * r_over_h * r_over_h);
    } else {
        return  0.25 * rs * rs * rs;
    }
}


inline double cubic_spline_kernel_region1(double r_over_h) {
    double rs = 1.0 - r_over_h;

    if ((r_over_h < 0) || (r_over_h > 1.0))
        return 0;
    if (r_over_h < 0.5){
        return  (0.5 - 3.0 * r_over_h * r_over_h * rs);
    } else {
        return  rs * rs * rs;
    }
}

inline double cubic_spline_kernel_deriv1(double r_over_h){
    double rs = 1.0 - r_over_h;

    if ((r_over_h < 0) || (r_over_h > 1.0))
        return 0;
    if (r_over_h < 0.5){
        return  r_over_h * (9.0*r_over_h - 6.);
    } else {
        return  -3.0 * rs * rs;
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
        dens += mass[ni] * cubic_spline_kernel_region1(r_over_h);
    }
    dens *= fNorm;

    return dens;
}

void calc_sph_gradient(
    int i, int num_near,
    const double* n_d,
    const int* n_index,
    const double* mass,
    const double* pos,
    const double* hsm,
    const double* target_pos,
    double* grad
) {

    int id = i * num_near;

    double h = n_d[id + num_near - 1];

    double inv_h = 1.0 / h;
    double inv_h2 = inv_h * inv_h;
    double fNorm = inv_h2 * inv_h2 * COEF1;  // cubic_spline_kernel_deriv1 ,not x 1/h, so here * 1/h


    for (int j = 0; j < num_near; ++j) {
        int idx = id + j;

        int ni = n_index[idx];
        double ri = n_d[idx];
        if (ri == 0.0) continue; // Skip division by zero
        double r_over_h = ri * inv_h;

        double w = mass[ni] * fNorm * cubic_spline_kernel_deriv1(r_over_h) / ri;  // 1/ri from, pos vector normalization
        grad[i*3] += w * (pos[ni * 3] - target_pos[i * 3]);
        grad[i*3 + 1] += w * (pos[ni * 3 + 1] - target_pos[i * 3 + 1]);
        grad[i*3 + 2] += w * (pos[ni * 3 + 2] - target_pos[i * 3 + 2]);
    }

}
