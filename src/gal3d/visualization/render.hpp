#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>

class Grid {
public:
    double xmin, ymin, xmax, ymax;
    int nx, ny;
    double dx, dy, dxdy;
    double xcenter0, ycenter0, inv_dxdy;
    std::vector<std::vector<double>> qty;

    Grid(double xmin_, double ymin_, double xmax_, double ymax_, int nx_, int ny_);
    void add_qty(double px, double py, double pqty);
};

class CubicSplineSmoothingKernel {
public:
    static constexpr double fnorm = 3.14159265358979323846 / 8.0;

    static std::vector<double> density_table;
    static std::vector<double> column_table;
    static constexpr double dr = 0.01;

    CubicSplineSmoothingKernel();

    double operator()(double r) const;
    static double density(double r);
    static double columnDensity(double R);

    static void init_table();
    static double lookup_density(double r);
    static double lookup_columnDensity(double r);
};

class KernelSampler {
public:
    Grid grid;
    const CubicSplineSmoothingKernel& kernel;

    KernelSampler(int nx, int ny, const CubicSplineSmoothingKernel& kernel_);
    void make_sample();

    std::vector<std::vector<double>> get_weights() const;
};

class RenderImage {
public:
    Grid image_grid;
    KernelSampler subsampler;
    std::vector<std::vector<double>> subsample_weights;
    bool do_subsample;

    RenderImage(double x_min, double x_max, double y_min, double y_max, int nx, int ny,
                const CubicSplineSmoothingKernel& kernel, int subsample_nx, int subsample_ny);

    void add_particle(double x, double y, double mass, double hsml);
    void add_particle(const std::vector<double>& x,
                      const std::vector<double>& y,
                      const std::vector<double>& mass,
                      const std::vector<double>& hsml);

    int circle_vs_canvas(double x, double y, double hsml) const;

    const std::vector<std::vector<double>>& get_values() const;
};