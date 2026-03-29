#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>


template<typename T>
class Grid {
public:
    T xmin, ymin, xmax, ymax;
    int nx, ny;
    T dx, dy, dxdy;
    T xcenter0, ycenter0, inv_dxdy;
    std::vector<T> qty;

    Grid(T xmin_, T ymin_, T xmax_, T ymax_, int nx_, int ny_);
    void add_qty(T px, T py, T pqty);
};

template<typename T>
class CubicSplineSmoothingKernel {
public:
    //static constexpr T fnorm = 3.14159265358979323846 / 8.0;

    static std::vector<T> density_table;
    static std::vector<T> column_table;
    static constexpr T dr = static_cast<T>(0.002); // Density table resolution

    CubicSplineSmoothingKernel();

    T operator()(T r) const;
    static T density(T r);
    static T columnDensity(T R);

    static void init_table();
    static T lookup_density(T r);
    static inline T lookup_columnDensity(T r);
};
template<typename T>
class KernelSampler {
public:
    Grid<T> grid;
    const CubicSplineSmoothingKernel<T>& kernel;

    KernelSampler(int nx, int ny, const CubicSplineSmoothingKernel<T>& kernel_);
    void make_sample();

    std::vector<T> get_weights_flat() const;
};
template<typename T>
class RenderImage {
public:
    Grid<T> image_grid;
    KernelSampler<T> subsampler;
    std::vector<T> subsample_weights;
    bool do_subsample;
    int numthreads;

    RenderImage(T x_min, T x_max, T y_min, T y_max, int nx, int ny,
                const CubicSplineSmoothingKernel<T>& kernel, int subsample_nx, int subsample_ny, int numthreads);


    void add_particle_to_qty(T x, T y, T mass, T hsml, std::vector<T>& qty);
    void add_particle(T x, T y, T mass, T hsml);
    void add_particle(const std::vector<T>& x,
                      const std::vector<T>& y,
                      const std::vector<T>& mass,
                      const std::vector<T>& hsml);

    int circle_vs_canvas(T x, T y, T hsml) const;

    std::vector<std::vector<T>> get_values() const;
    const std::vector<T>& get_flat_values() const;
};