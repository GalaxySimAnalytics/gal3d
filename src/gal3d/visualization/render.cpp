#include "render.hpp"

// Define M_PI if not defined (for Windows compatibility)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

template<typename T>
std::vector<T> CubicSplineSmoothingKernel<T>::density_table;

template<typename T>
std::vector<T> CubicSplineSmoothingKernel<T>::column_table;

template<typename T>
Grid<T>::Grid(T xmin_, T ymin_, T xmax_, T ymax_, int nx_, int ny_)
    : xmin(xmin_), ymin(ymin_), xmax(xmax_), ymax(ymax_), nx(nx_), ny(ny_) {
    dx = (xmax - xmin) / nx;
    dy = (ymax - ymin) / ny;
    dxdy = dx * dy;
    xcenter0 = xmin + static_cast<T>(0.5) * dx; // Initialize
    ycenter0 = ymin + static_cast<T>(0.5) * dy;
    inv_dxdy = static_cast<T>(1.0) / dxdy;
    qty.resize(nx * ny, static_cast<T>(0.0));
}

template<typename T>
void Grid<T>::add_qty(T px, T py, T pqty) {
    int ix = int((px - xmin) / dx);
    int iy = int((py - ymin) / dy);
    if (ix >= 0 && ix < nx && iy >= 0 && iy < ny) {
        qty[iy * nx + ix] += pqty * inv_dxdy;
    }
}

template<typename T>
CubicSplineSmoothingKernel<T>::CubicSplineSmoothingKernel() {
    if (density_table.empty() || column_table.empty()) {
        init_table();
    }
}

template<typename T>
T CubicSplineSmoothingKernel<T>::operator()(T r) const {
    return density(r);
}

template<typename T>
T CubicSplineSmoothingKernel<T>::density(T r) {
    if (r < static_cast<T>(0.0) || r >= static_cast<T>(1.0)) return static_cast<T>(0.0);
    if (r < static_cast<T>(0.5))
        return static_cast<T>(8.0) / static_cast<T>(M_PI) * (static_cast<T>(1.0) - static_cast<T>(6.0) * r * r * (static_cast<T>(1.0) - r));
    else
        return static_cast<T>(8.0) / static_cast<T>(M_PI) * static_cast<T>(2.0) * pow(static_cast<T>(1.0) - r, static_cast<T>(3.0));
}

template<typename T>
T CubicSplineSmoothingKernel<T>::columnDensity(T R) {
    if (R < 0.0 || R >= 1.0) return 0.0;
    T R2 = R * R;
    if (R < 1e-6)
        return 6.0 / M_PI * (1.0 - 8.0 * std::log(2.0) * R2);

    T s = std::sqrt((1.0 - R) * (1.0 + R));
    if (R < 0.5) {
        T t = std::sqrt((1.0 - 2.0 * R) * (1.0 + 2.0 * R));
        T p1 = (4.0 + 26.0 * R2) * s;
        T p2 = (1.0 + 26.0 * R2) * t;
        T p3 = 18.0 * R2 * R2 * std::log(2.0 * R / (1.0 + t));
        T p4 = 6.0 * R2 * (4.0 + R2) * std::log(2.0 * (1.0 + s) / (1.0 + t));
        return 2.0 / M_PI * (p1 - p2 - p3 - p4);
    } else {
        T p1 = (2.0 + 13.0 * R2) * s;
        T p2 = 3.0 * R2 * (4.0 + R2) * std::log(R / (1.0 + s));
        return 4.0 / M_PI * (p1 + p2);
    }
}

template<typename T>
T integrate_columnDensity(T x0, T x1, T y0, T y1, int Nx, int Ny, const CubicSplineSmoothingKernel<T>& kernel) {
    T dx = (x1 - x0) / Nx;
    T dy = (y1 - y0) / Ny;
    T sum = 0.0;
    for (int i = 0; i < Nx; ++i) {
        T xi = x0 + (i + 0.5) * dx;
        for (int j = 0; j < Ny; ++j) {
            T yj = y0 + (j + 0.5) * dy;
            T R2 = xi * xi + yj * yj;
            sum += kernel.lookup_columnDensity(R2);
        }
    }
    return sum * dx * dy;
}

template<typename T>
void CubicSplineSmoothingKernel<T>::init_table() {
    int n = int(1.0 / dr) + 1;
    density_table.resize(n);
    column_table.resize(n);
    for (int i = 0; i < n; ++i) {
        T r2 = T(i) * dr; // Uniform sampling of r2
        T r = std::sqrt(r2);
        density_table[i] = density(r);
        column_table[i] = columnDensity(r);
    }
}

template<typename T>
T CubicSplineSmoothingKernel<T>::lookup_density(T r2) {
    if (r2 < 0.0 || r2 > 1.0) return 0.0;
    size_t idx = static_cast<size_t>(r2 / dr);
    if (idx >= density_table.size()) return 0.0;
    return density_table[idx];
}

template<typename T>
inline T CubicSplineSmoothingKernel<T>::lookup_columnDensity(T R2) {
    if (R2 < 0.0 || R2 > 1.0) return 0.0;
    size_t idx = static_cast<size_t>(R2 / dr);
    if (idx >= column_table.size()) return 0.0;
    return column_table[idx];
}

template<typename T>
KernelSampler<T>::KernelSampler(int nx, int ny, const CubicSplineSmoothingKernel<T>& kernel_)
    : grid(-1, -1, 1, 1, nx, ny), kernel(kernel_) {
    make_sample();
}

template<typename T>
void KernelSampler<T>::make_sample() {
    for (int i = 0; i < grid.nx; ++i) {
        for (int j = 0; j < grid.ny; ++j) {
            T x = grid.xmin + 0.5 * grid.dx + i * grid.dx;
            T y = grid.ymin + 0.5 * grid.dy + j * grid.dy;
            T R2 = x * x + y * y;
            grid.add_qty(x, y, kernel.lookup_columnDensity(R2));
        }
    }
}

template<typename T>
std::vector<T> KernelSampler<T>::get_weights_flat() const {
    T total = 0.0;
    int nx = grid.nx, ny = grid.ny;
    for (T v : grid.qty) total += v;
    std::vector<T> weights(nx * ny, static_cast<T>(0.0));
    if (total > static_cast<T>(0.0)) {
        T inv_total = static_cast<T>(1.0) / total;
        for (int idx = 0; idx < nx * ny; ++idx)
            weights[idx] = grid.qty[idx] * inv_total;
    }
    return weights;
}

template<typename T>
RenderImage<T>::RenderImage(T x_min, T x_max, T y_min, T y_max, int nx, int ny,
                         const CubicSplineSmoothingKernel<T>& kernel, int subsample_nx, int subsample_ny, int numthreads)
    : image_grid(x_min, y_min, x_max, y_max, nx, ny),
      subsampler(subsample_nx, subsample_ny, kernel),
      subsample_weights(subsampler.get_weights_flat()),
      do_subsample(!(subsample_nx <= 1 && subsample_ny <= 1)),
      numthreads(numthreads)
{}

template<typename T>
void RenderImage<T>::add_particle_to_qty(T x, T y, T mass, T hsml, std::vector<T>& qty) {
    int canvas_status = circle_vs_canvas(x, y, hsml);

    // 0: completely outside the canvas
    if (canvas_status == 0) return;

    T dx = image_grid.dx;
    T dy = image_grid.dy;

    hsml = std::max(hsml, dx + dy);
    T inv_hsml2 = static_cast<T>(1.0) / (hsml * hsml);

    int x0 = int(std::floor((x - hsml - image_grid.xmin) / dx));
    int x1 = int(std::floor((x + hsml - image_grid.xmin) / dx));
    int y0 = int(std::floor((y - hsml - image_grid.ymin) / dy));
    int y1 = int(std::floor((y + hsml - image_grid.ymin) / dy));
    int nx = x1 - x0 + 1;
    int ny = y1 - y0 + 1;
    int area_npix_part = nx * ny;

    if (canvas_status == 2 && area_npix_part == 1) {
        //#pragma omp atomic
        qty[x0 + y0 * image_grid.nx] += mass * image_grid.inv_dxdy;
        return;
    }

    bool need_subsample = (do_subsample && (subsampler.grid.nx * subsampler.grid.ny > 4 * area_npix_part));

    // All inside
    if (canvas_status == 2) {


        // No need to resample, directly normalize and calculate the contribution to each pixel
        if (!need_subsample){

            thread_local std::vector<T> vals;
            thread_local std::vector<T> dx_arr;
            vals.resize(nx * ny);
            dx_arr.resize(nx);

            T total_val = static_cast<T>(0.0);
            for (int i = 0; i < nx; ++i)
                dx_arr[i] = image_grid.xcenter0 + (x0 + i) * image_grid.dx - x;

            // First calculate all val and accumulate
            for (int j = 0; j < ny; ++j) {
                T i_y = image_grid.ycenter0 + (y0 + j) * image_grid.dy - y;
                T iy2 = i_y * i_y;

                int row_offset = j * nx;

                for (int i = 0; i < nx; ++i) {
                    T r2 = iy2 + dx_arr[i] * dx_arr[i];
                    T val = CubicSplineSmoothingKernel<T>::lookup_columnDensity(
                        r2 * inv_hsml2
                    );
                    /* just lookup to table
                    T val = CubicSplineSmoothingKernel<T>::columnDensity(
                        std::sqrt(r2 * inv_hsml * inv_hsml)
                    );
                    */
                    vals[row_offset + i] = val;
                    total_val += val;
                }
            }

            // Normalize and add to the grid
            if (total_val > static_cast<T>(0.0)) {
                T scale = mass * image_grid.inv_dxdy / total_val;
                for (int j = 0; j < ny; ++j) {
                    int grid_row = (y0 + j) * image_grid.nx + x0;
                    int row_offset = j * nx;
                    for (int i = 0; i < nx; ++i)
                        qty[grid_row + i] += scale * vals[row_offset + i];
                }
            } 
            return;
        } else {

            int snx = subsampler.grid.nx, sny = subsampler.grid.ny;
            for (int j = 0; j < sny; ++j) {
                T yy = y + (subsampler.grid.ymin + 0.5 * subsampler.grid.dy + j * subsampler.grid.dy) * hsml;
                int iy = int((yy - image_grid.ymin) / image_grid.dy);
                if (iy < 0 || iy >= image_grid.ny) continue;
                int iy_row = iy * image_grid.nx;
                for (int i = 0; i < snx; ++i) {
                    T xx = x + (subsampler.grid.xmin + 0.5 * subsampler.grid.dx + i * subsampler.grid.dx) * hsml;
                    int ix = int((xx - image_grid.xmin) / image_grid.dx);
                    if (ix < 0 || ix >= image_grid.nx) continue;
                    // FIX P2: flat index
                    qty[iy_row + ix] += mass * subsample_weights[j * snx + i] * image_grid.inv_dxdy;
                }
            }
        }
        
    } else {
        // Partially inside the canvas

        // No need to resample, directly normalize and calculate the contribution to each pixel
        if (!need_subsample){
            thread_local std::vector<T> vals;
            thread_local std::vector<T> dx_arr;
            vals.resize(nx * ny);
            dx_arr.resize(nx);

            T total_val = static_cast<T>(0.0);
            for (int i = 0; i < nx; ++i)
                dx_arr[i] = image_grid.xcenter0 + (x0 + i) * image_grid.dx - x;

            // First calculate all val and accumulate
            for (int j = 0; j < ny; ++j) {
                T i_y = image_grid.ycenter0 + (y0 + j) * image_grid.dy - y;
                T iy2 = i_y * i_y;

                int row_offset = j * nx;

                for (int i = 0; i < nx; ++i) {
                    T r2 = iy2 + dx_arr[i] * dx_arr[i];
                    T val = CubicSplineSmoothingKernel<T>::lookup_columnDensity(
                        r2 * inv_hsml2
                    );
                    vals[row_offset + i] = val;
                    total_val += val;
                }
            }

            // Normalize and add to the grid
            if (total_val > static_cast<T>(0.0)) {
                T scale = mass * image_grid.inv_dxdy / total_val;
                int j_beg = std::max(y0, 0);
                int j_end = std::min(y0 + ny, image_grid.ny);
                int i_beg = std::max(x0, 0);
                int i_end = std::min(x0 + nx, image_grid.nx);
                for (int gj = j_beg; gj < j_end; ++gj) {
                    int row_offset = (gj - y0) * nx - x0;  // vals index = row_offset + gi
                    int grid_row   = gj * image_grid.nx;
                    for (int gi = i_beg; gi < i_end; ++gi)
                        qty[grid_row + gi] += scale * vals[row_offset + gi];
                }
            }

            
        } else {
            int snx = subsampler.grid.nx, sny = subsampler.grid.ny;
            for (int j = 0; j < sny; ++j) {
                T yy = y + (subsampler.grid.ymin + 0.5 * subsampler.grid.dy + j * subsampler.grid.dy) * hsml;
                int iy = int((yy - image_grid.ymin) / image_grid.dy);
                if (iy < 0 || iy >= image_grid.ny) continue;
                int iy_row = iy * image_grid.nx;
                for (int i = 0; i < snx; ++i) {
                    T xx = x + (subsampler.grid.xmin + 0.5 * subsampler.grid.dx + i * subsampler.grid.dx) * hsml;
                    int ix = int((xx - image_grid.xmin) / image_grid.dx);
                    if (ix >= 0 && ix < image_grid.nx)
                        qty[iy_row + ix] += mass * subsample_weights[j * snx + i] * image_grid.inv_dxdy;  // FIX P2
                }
            }
        }


    }
}

template<typename T>
void RenderImage<T>::add_particle(T x, T y, T mass, T hsml) {
    add_particle_to_qty(x, y, mass, hsml, image_grid.qty);
}

template<typename T>
void RenderImage<T>::add_particle(const std::vector<T>& x,
                               const std::vector<T>& y,
                               const std::vector<T>& mass,
                               const std::vector<T>& hsml) {
    size_t n = x.size();

    int nthreads = numthreads;
    omp_set_num_threads(nthreads);

    std::vector<std::vector<T>> thread_qty(nthreads, std::vector<T>(image_grid.nx * image_grid.ny, 0.0));

    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t idx = 0; idx < n; ++idx) {
        int tid = omp_get_thread_num();
        add_particle_to_qty(x[idx], y[idx], mass[idx], hsml[idx], thread_qty[tid]);
    }
    // Merge
    for (int t = 0; t < nthreads; ++t)
        for (size_t i = 0; i < image_grid.qty.size(); ++i)
            image_grid.qty[i] += thread_qty[t][i];
}



template<typename T>
int RenderImage<T>::circle_vs_canvas(T x, T y, T hsml) const {
    T x_min = x - hsml;
    T x_max = x + hsml;
    T y_min = y - hsml;
    T y_max = y + hsml;
    /*
    Check the relationship between the x and y intervals and the canvas.
    Each direction has six cases. Use .. to represent the distribution interval of a certain dimension x, and || to represent the grid interval:
    xmin >= grid_min, xmin <= grid_max, xmax >= grid_min, xmax <= grid_max;
    F T F T,  . . | |  No overlapping interval, status 0,
    T F T F,  | | . .  No overlapping interval, status 0,
    F T T T,  . | . |  Overlapping interval exists, status 1,
    T T T F,  | . | .  Overlapping interval exists, status 1,
    F T T F,  . | | .  x distribution encloses the grid, but here simply marked as 1
    T T T T,  | . . |  Grid encloses x's distribution, marked as status 2
    */
    int x_count_in = 0;
    if (x_min <= image_grid.xmax && x_max >=image_grid.xmin){
        if (x_min>=image_grid.xmin && x_max<=image_grid.xmax) x_count_in++;
        x_count_in++;
    } else {
        return 0;
    }

    int y_count_in = 0;

    if (y_min <= image_grid.ymax && y_max >= image_grid.ymin) {
        if (y_min >= image_grid.ymin && y_max <= image_grid.ymax) y_count_in++;
        y_count_in++;
    } else {
        return 0;
    }

    int count_in = x_count_in * y_count_in;

    if (count_in == 0) {
        return 0;
    } else if (count_in == 4) {
        return 2; // Completely inside
    } else if (count_in == 2) {
        return 1;  // Partially inside
    } else if (count_in == 1) {
        // Only one vertex is inside the canvas, need to further check if the circle intersects the canvas
        T closest_x = std::max(image_grid.xmin, std::min(x, image_grid.xmax));
        T closest_y = std::max(image_grid.ymin, std::min(y, image_grid.ymax));
        T dist2 = (closest_x - x) * (closest_x - x) + (closest_y - y) * (closest_y - y);
        if (dist2 > hsml * hsml) return 0; // Completely outside
        else return 1; // Partially inside
    }
    // Default return value to prevent missing return causing compilation errors
    return 0;
}

template<typename T>
std::vector<std::vector<T>> RenderImage<T>::get_values() const {
    int nx = image_grid.nx, ny = image_grid.ny;
    std::vector<std::vector<T>> values(ny, std::vector<T>(nx, 0.0));
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
            values[j][i] = image_grid.qty[j * nx + i];
    return values;
}

template<typename T>
const std::vector<T>& RenderImage<T>::get_flat_values() const {
    return image_grid.qty;
}


template class Grid<float>;
template class Grid<double>;
template class CubicSplineSmoothingKernel<float>;
template class CubicSplineSmoothingKernel<double>;
template class KernelSampler<float>;
template class KernelSampler<double>;
template class RenderImage<float>;
template class RenderImage<double>;