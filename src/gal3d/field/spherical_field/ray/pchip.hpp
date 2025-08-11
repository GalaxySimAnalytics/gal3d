#pragma once
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <array>

class PchipInterpolator {
public:
    PchipInterpolator(const std::vector<double>& x, const std::vector<double>& y);
    PchipInterpolator(const double* x, const double* y, size_t n);
    double interpolate(double xval, int nu = 0) const;
    std::vector<double> interpolate(const std::vector<double>& xvals, int nu = 0) const;
    std::vector<double> interpolate(const double* xvals, size_t n, int nu = 0) const;
    
    double get_x_min() const { return x_.front(); }
    double get_x_max() const { return x_.back(); }

    template<typename T>
    double interpolate(T xval, int nu = 0) const;

    template<typename T>
    std::vector<double> interpolate(const std::vector<T>& xvals, int nu = 0) const;

    std::vector<double> solve(double y = 0.0, bool discontinuity = true, bool extrapolate = true) const;
    std::vector<std::vector<double>> solve(const std::vector<double>& yvals, bool discontinuity = true, bool extrapolate = true) const;
    std::vector<std::vector<double>> solve(const double* yvals, size_t n, bool discontinuity = true, bool extrapolate = true) const;

    template<typename T>
    std::vector<double> solve(T y, bool discontinuity = true, bool extrapolate = true) const;

    template<typename T>
    std::vector<std::vector<double>> solve(const std::vector<T>& y, bool discontinuity = true, bool extrapolate = true) const;



private:
    std::vector<double> x_, y_, d_; // nodes, values, derivatives
    std::vector<std::array<double, 4>> c_; // coefficients of cubic polynomials for each segment

    void compute_derivatives();
    void compute_coefficients();
    size_t find_interval(double xval) const;
};

template<typename T>
double PchipInterpolator::interpolate(T xval, int nu) const {
    return interpolate(static_cast<double>(xval), nu);
}

template<typename T>
std::vector<double> PchipInterpolator::interpolate(const std::vector<T>& xvals, int nu) const {
    std::vector<double> results(xvals.size());
    for (size_t i = 0; i < xvals.size(); ++i) {
        results[i] = interpolate(static_cast<double>(xvals[i]), nu);
    }
    return results;
}


template<typename T>
std::vector<double> PchipInterpolator::solve(T y, bool discontinuity, bool extrapolate) const {
    return solve(static_cast<double>(y), discontinuity, extrapolate);
}

template<typename T>
std::vector<std::vector<double>> PchipInterpolator::solve(const std::vector<T>& yvals, bool discontinuity, bool extrapolate) const {
    std::vector<std::vector<double>> results(yvals.size());
    for (size_t i = 0; i < yvals.size(); ++i) {
        results[i] = solve(static_cast<double>(yvals[i]), discontinuity, extrapolate);
    }
    return results;
}

template<typename Tx, typename Ty>
PchipInterpolator make_pchip_interpolator(const std::vector<Tx>& x, const std::vector<Ty>& y) {
    return PchipInterpolator(
        std::vector<double>(x.begin(), x.end()),
        std::vector<double>(y.begin(), y.end())
    );
}

