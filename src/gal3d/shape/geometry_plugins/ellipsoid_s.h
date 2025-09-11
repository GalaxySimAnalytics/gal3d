#ifndef ELLIPSOID_S_H
#define ELLIPSOID_S_H

#ifdef __cplusplus
extern "C" {
#endif
void f_shaped_ellipsoid_cpp(
    double a, double b, double c, double Sa, double Sb, double Sc,
    const double* pos, int n, double* result, double* r, int num_threads);

void f_shaped_ellipsoid_jacobian_cpp(
    double a, double b, double c, double Sa, double Sb, double Sc,
    const double* pos, int n,
    double* da, double* db, double* dc,
    double* dSa, double* dSb, double* dSc,
    double* dx, double* dy, double* dz,
    int num_threads);

void IntersectRaysEllipsoid_S_cpp(
    double a, double b, double c, double Sa, double Sb, double Sc,
    const double* pos, int n, int maxIterations,
    double* tarpos, double* result, double* r, int method, int num_threads);

void f_ray_shaped_ellipsoid_cpp(
    double a, double b, double c, double Sa, double Sb, double Sc,
    const double* pos, int n, int maxIterations,
    double* result, double* r, int method, int num_threads);

void area_factor_cpp(
    double a, double b, double c, double Sa, double Sb, double Sc,
    const double* pos, int n, double* result, int num_threads);

void IntersectLinesEllipsoid_S_cpp(
    double a, double b, double c, double Sa, double Sb, double Sc,
    const double* pos1, const double* pos2, int n, int maxIterations,
    double* ts, int num_threads);

#ifdef __cplusplus
}
#endif

#endif