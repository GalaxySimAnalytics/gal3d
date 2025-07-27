#include "render.hpp"

std::vector<double> CubicSplineSmoothingKernel::density_table;
std::vector<double> CubicSplineSmoothingKernel::column_table;

Grid::Grid(double xmin_, double ymin_, double xmax_, double ymax_, int nx_, int ny_)
    : xmin(xmin_), ymin(ymin_), xmax(xmax_), ymax(ymax_), nx(nx_), ny(ny_) {
    dx = (xmax - xmin) / nx;
    dy = (ymax - ymin) / ny;
    dxdy = dx * dy;
    xcenter0 = xmin + 0.5 * dx; // 初始化
    ycenter0 = ymin + 0.5 * dy;
    inv_dxdy = 1.0 / dxdy;
    qty.resize(ny, std::vector<double>(nx, 0.0));
}

void Grid::add_qty(double px, double py, double pqty) {
    int ix = int((px - xmin) / dx);
    int iy = int((py - ymin) / dy);
    if (ix >= 0 && ix < nx && iy >= 0 && iy < ny) {
        qty[iy][ix] += pqty * inv_dxdy;
    }
}

CubicSplineSmoothingKernel::CubicSplineSmoothingKernel() {
    if (density_table.empty() || column_table.empty()) {
        init_table();
    }
}

double CubicSplineSmoothingKernel::operator()(double r) const {
    return density(r);
}

double CubicSplineSmoothingKernel::density(double r) {
    if (r < 0.0 || r >= 1.0) return 0.0;
    if (r < 0.5)
        return 8.0 / M_PI * (1.0 - 6.0 * r * r * (1.0 - r));
    else
        return 8.0 / M_PI * 2.0 * pow(1.0 - r, 3);
}

double CubicSplineSmoothingKernel::columnDensity(double R) {
    if (R < 0.0 || R >= 1.0) return 0.0;
    double R2 = R * R;
    if (R < 1e-6)
        return 6.0 / M_PI * (1.0 - 8.0 * std::log(2.0) * R2);

    double s = std::sqrt((1.0 - R) * (1.0 + R));
    if (R < 0.5) {
        double t = std::sqrt((1.0 - 2.0 * R) * (1.0 + 2.0 * R));
        double p1 = (4.0 + 26.0 * R2) * s;
        double p2 = (1.0 + 26.0 * R2) * t;
        double p3 = 18.0 * R2 * R2 * std::log(2.0 * R / (1.0 + t));
        double p4 = 6.0 * R2 * (4.0 + R2) * std::log(2.0 * (1.0 + s) / (1.0 + t));
        return 2.0 / M_PI * (p1 - p2 - p3 - p4);
    } else {
        double p1 = (2.0 + 13.0 * R2) * s;
        double p2 = 3.0 * R2 * (4.0 + R2) * std::log(R / (1.0 + s));
        return 4.0 / M_PI * (p1 + p2);
    }
}

double integrate_columnDensity(double x0, double x1, double y0, double y1, int Nx, int Ny, const CubicSplineSmoothingKernel& kernel) {
    double dx = (x1 - x0) / Nx;
    double dy = (y1 - y0) / Ny;
    double sum = 0.0;
    for (int i = 0; i < Nx; ++i) {
        double xi = x0 + (i + 0.5) * dx;
        for (int j = 0; j < Ny; ++j) {
            double yj = y0 + (j + 0.5) * dy;
            double R = std::sqrt(xi * xi + yj * yj);
            sum += kernel.lookup_columnDensity(R);
        }
    }
    return sum * dx * dy;
}


void CubicSplineSmoothingKernel::init_table() {
    int n = int(1.0 / dr) + 1;
    density_table.resize(n);
    column_table.resize(n);
    for (int i = 0; i < n; ++i) {
        double r = i * dr;
        density_table[i] = density(r);
        column_table[i] = columnDensity(r);
    }
}
double CubicSplineSmoothingKernel::lookup_density(double r) {
    if (r < 0.0 || r > 1.0) return 0.0;
    size_t idx = static_cast<size_t>(r / dr);
    if (idx >= density_table.size()) return 0.0;
    return density_table[idx];
}

double CubicSplineSmoothingKernel::lookup_columnDensity(double r) {
    if (r < 0.0 || r > 1.0) return 0.0;
    size_t idx = static_cast<size_t>(r / dr);
    if (idx >= column_table.size()) return 0.0;
    return column_table[idx];
}


KernelSampler::KernelSampler(int nx, int ny, const CubicSplineSmoothingKernel& kernel_)
    : grid(-1, -1, 1, 1, nx, ny), kernel(kernel_) {
    make_sample();
}

void KernelSampler::make_sample() {
    for (int i = 0; i < grid.nx; ++i) {
        for (int j = 0; j < grid.ny; ++j) {
            double x = grid.xmin + 0.5 * grid.dx + i * grid.dx;
            double y = grid.ymin + 0.5 * grid.dy + j * grid.dy;
            double R = std::sqrt(x * x + y * y);
            grid.add_qty(x, y, kernel.lookup_columnDensity(R));
        }
    }
}

std::vector<std::vector<double>> KernelSampler::get_weights() const {
    double total = 0.0;
    for (const auto& row : grid.qty)
        for (double v : row)
            total += v;
    std::vector<std::vector<double>> weights = grid.qty;
    if (total > 0.0) {
        for (auto& row : weights)
            for (double& v : row)
                v /= total;
    }
    return weights;
}


RenderImage::RenderImage(double x_min, double x_max, double y_min, double y_max, int nx, int ny,
                         const CubicSplineSmoothingKernel& kernel, int subsample_nx, int subsample_ny)
    : image_grid(x_min, y_min, x_max, y_max, nx, ny),
      subsampler(subsample_nx, subsample_ny, kernel),
      subsample_weights(subsampler.get_weights()),
      do_subsample(!(subsample_nx <= 1 && subsample_ny <= 1))
{}

void RenderImage::add_particle(double x, double y, double mass, double hsml) {
    int canvas_status = circle_vs_canvas(x, y, hsml);

    // 0: 完全不在画布内
    if (canvas_status == 0) return;

    double dx = image_grid.dx;
    double dy = image_grid.dy;
    int x0 = static_cast<int>(std::floor((x - hsml - image_grid.xmin) / dx));
    int x1 = static_cast<int>(std::ceil((x + hsml - image_grid.xmin) / dx));
    int y0 = static_cast<int>(std::floor((y - hsml - image_grid.ymin) / dy));
    int y1 = static_cast<int>(std::ceil((y + hsml - image_grid.ymin) / dy));
    int nx = x1 - x0 + 1;
    int ny = y1 - y0 + 1;
    int area_npix_part = nx * ny;

    bool need_subsample = (do_subsample && (4 * subsampler.grid.nx * subsampler.grid.ny > area_npix_part));

    // 全在画布内,
    if (canvas_status == 2) {

        // 并只影响 1 个 pixel，直接加入
        if (area_npix_part == 1) {
            image_grid.add_qty(x, y, mass);
            return;
        }

        // 不需要重采样，直接归一化计算对每一个pixel的贡献
        if (!need_subsample){

            std::vector<std::vector<double>> vals(ny, std::vector<double>(nx, 0.0));
            double total_val = 0.0;

            // 先计算所有 val 并累加
            for (int i = 0; i < nx; ++i) {
                for (int j = 0; j < ny; ++j) {
                    double i_x = image_grid.xcenter0 + (x0 + i) * image_grid.dx;
                    double j_y = image_grid.ycenter0 + (y0 + j) * image_grid.dy;
                    double val = CubicSplineSmoothingKernel::lookup_columnDensity(
                        std::sqrt((i_x - x) * (i_x - x) + (j_y - y) * (j_y - y)) / hsml
                    );
                    vals[j][i] = val;
                    total_val += val;
                }
            }

            // 归一化并加到网格
            if (total_val > 0.0) {
                #pragma omp parallel for collapse(2)
                for (int i = 0; i < nx; ++i) {
                    for (int j = 0; j < ny; ++j) {
                        int grid_i = x0 + i;
                        int grid_j = y0 + j;
                        double norm_val = mass * vals[j][i] / total_val;
                        /*
                        理论上canvas_status == 2情况下 grid_i 和 grid_j 应该在 [0, image_grid.nx) 和 [0, image_grid.ny) 之间， 
                        但是由于浮点误差 (我猜测)，一些特殊的情况下会超出范围
                        所以在这里加了这个条件
                        */
                        if (grid_i >= 0 && grid_i < image_grid.nx && grid_j >= 0 && grid_j < image_grid.ny) {
                            #pragma omp atomic
                            image_grid.qty[grid_j][grid_i] += norm_val * image_grid.inv_dxdy;
                        }
                    }
                }
            }
            return;
        } else {

            #pragma omp parallel for collapse(2)
            for (int i = 0; i < subsampler.grid.nx; ++i) {
                for (int j = 0; j < subsampler.grid.ny; ++j) {
                    double xx = x + (subsampler.grid.xmin + 0.5 * subsampler.grid.dx + i * subsampler.grid.dx) * hsml;
                    int ix = int((xx - image_grid.xmin) / image_grid.dx);
                    double yy = y + (subsampler.grid.ymin + 0.5 * subsampler.grid.dy + j * subsampler.grid.dy) * hsml;
                    double value = mass * subsample_weights[j][i];
                    int iy = int((yy - image_grid.ymin) / image_grid.dy);
                    #pragma omp atomic
                    image_grid.qty[iy][ix] += value * image_grid.inv_dxdy;
                }
            }
        }
        
    } else {
        // 部分在画布内

        // 不需要重采样，直接归一化计算对每一个pixel的贡献
        if (!need_subsample){


            std::vector<std::vector<double>> vals(ny, std::vector<double>(nx, 0.0));
            double total_val = 0.0;

            // 先计算所有 val 并累加
            for (int i = 0; i < nx; ++i) {
                for (int j = 0; j < ny; ++j) {
                    double i_x = image_grid.xcenter0 + (x0 + i) * image_grid.dx;
                    double j_y = image_grid.ycenter0 + (y0 + j) * image_grid.dy;
                    double val = CubicSplineSmoothingKernel::lookup_columnDensity(
                        std::sqrt((i_x - x) * (i_x - x) + (j_y - y) * (j_y - y)) / hsml
                    );
                    vals[j][i] = val;
                    total_val += val;
                }
            }

            // 归一化并加到网格
            if (total_val > 0.0) {
                #pragma omp parallel for collapse(2)
                for (int i = 0; i < nx; ++i) {
                    for (int j = 0; j < ny; ++j) {
                        int grid_i = x0 + i;
                        int grid_j = y0 + j;
                        // 判断在网格内
                        if (grid_i >= 0 && grid_i < image_grid.nx && grid_j >= 0 && grid_j < image_grid.ny) {
                            double norm_val = mass * vals[j][i] / total_val;
                            #pragma omp atomic
                            image_grid.qty[grid_j][grid_i] += norm_val * image_grid.inv_dxdy;
                        }
                    }
                }
            }
            return;

            
        } else {

            #pragma omp parallel for collapse(2)
            for (int i = 0; i < subsampler.grid.nx; ++i) {
                for (int j = 0; j < subsampler.grid.ny; ++j) {
                    double xx = x + (subsampler.grid.xmin + 0.5 * subsampler.grid.dx + i * subsampler.grid.dx) * hsml;
                    int ix = int((xx - image_grid.xmin) / image_grid.dx);
                    double yy = y + (subsampler.grid.ymin + 0.5 * subsampler.grid.dy + j * subsampler.grid.dy) * hsml;
                    int iy = int((yy - image_grid.ymin) / image_grid.dy);
                    if (ix >= 0 && ix < image_grid.nx && iy >= 0 && iy < image_grid.ny) {
                        double value = mass * subsample_weights[j][i];
                        #pragma omp atomic
                        image_grid.qty[iy][ix] += value * image_grid.inv_dxdy;
                    }
                }
            }
        }


    }
}

void RenderImage::add_particle(const std::vector<double>& x,
                               const std::vector<double>& y,
                               const std::vector<double>& mass,
                               const std::vector<double>& hsml) {
    size_t n = x.size();
    #pragma omp parallel for
    for (size_t idx = 0; idx < n; ++idx) {
        add_particle(x[idx], y[idx], mass[idx], hsml[idx]);
    }
}

int RenderImage::circle_vs_canvas(double x, double y, double hsml) const {
    double x_min = x - hsml;
    double x_max = x + hsml;
    double y_min = y - hsml;
    double y_max = y + hsml;

    // 判断 x 方向有几个点在画布内
    int x_count_in = 0;
    if (x_min >= image_grid.xmin && x_min <= image_grid.xmax) x_count_in++;
    if (x_max >= image_grid.xmin && x_max <= image_grid.xmax) x_count_in++;

    // 判断 y 方向有几个点在画布内
    int y_count_in = 0;
    if (y_min >= image_grid.ymin && y_min <= image_grid.ymax) y_count_in++;
    if (y_max >= image_grid.ymin && y_max <= image_grid.ymax) y_count_in++;

    int count_in = x_count_in * y_count_in;

    if (count_in == 0) {
        //判断圆心是否在画布内
        if (x >= image_grid.xmin && x <= image_grid.xmax &&
            y >= image_grid.ymin && y <= image_grid.ymax)
            return 1;
        return 0;
    } else if (count_in == 4) {
        return 2; // 全在
    } else if (count_in == 2) {
        return 1;  // 部分在
    } else if (count_in == 1) {
        // 只有一个顶点在画布内，需进一步判断圆是否与画布有交点
        double closest_x = std::clamp(x, image_grid.xmin, image_grid.xmax);
        double closest_y = std::clamp(y, image_grid.ymin, image_grid.ymax);
        double dist2 = (closest_x - x) * (closest_x - x) + (closest_y - y) * (closest_y - y);
        if (dist2 > hsml * hsml) return 0; // 完全不在
        else return 1; // 有部分在
    }
}

const std::vector<std::vector<double>>& RenderImage::get_values() const {
    return image_grid.qty;
}