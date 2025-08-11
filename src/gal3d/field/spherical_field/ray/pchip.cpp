
#include <cmath>
#include "pchip.hpp"

PchipInterpolator::PchipInterpolator(const std::vector<double>& x, const std::vector<double>& y)
    : x_(x), y_(y), d_(x.size())
{
    if (x.size() != y.size() || x.size() < 2)
        throw std::invalid_argument("x and y must have same length >= 2");
    for (size_t i = 1; i < x.size(); ++i)
        if (x[i] <= x[i-1])
            throw std::invalid_argument("x must be strictly increasing");
    compute_derivatives();
    compute_coefficients();
}

PchipInterpolator::PchipInterpolator(const double* x, const double* y, size_t n)
    : x_(x, x + n), y_(y, y + n), d_(n)
{
    if (n < 2)
        throw std::invalid_argument("length must be >= 2");
    for (size_t i = 1; i < n; ++i)
        if (x[i] <= x[i-1])
            throw std::invalid_argument("x must be strictly increasing");
    compute_derivatives();
    compute_coefficients();
}

void PchipInterpolator::compute_derivatives() {
    size_t n = x_.size();
    std::vector<double> h(n-1), m(n-1);
    for (size_t i = 0; i < n-1; ++i) {
        h[i] = x_[i+1] - x_[i];
        m[i] = (y_[i+1] - y_[i]) / h[i];
    }

    // Endpoints: one-sided estimate
    if (n == 2) {
        d_[0] = m[0];
        d_[1] = m[0];
        return;
    }

    // Internal points
    for (size_t k = 1; k < n-1; ++k) {
        if ((m[k-1] == 0.0) || (m[k] == 0.0) || (m[k-1] * m[k] < 0.0)) {
            d_[k] = 0.0;
        } else {
            double w1 = 2*h[k] + h[k-1];
            double w2 = h[k] + 2*h[k-1];
            d_[k] = (w1 + w2) / (w1/m[k-1] + w2/m[k]);
        }
    }

    // End slopes (Fritsch-Butland one-sided)
    // Left
    {
        double h0 = h[0], h1 = h[1], m0 = m[0], m1 = m[1];
        double d = ((2*h0 + h1)*m0 - h0*m1) / (h0 + h1);
        if ((d * m0) <= 0.0)
            d_[0] = 0.0;
        else if ((m0 * m1 < 0.0) && (std::abs(d) > 3.0 * std::abs(m0)))
            d_[0] = 3.0 * m0;
        else
            d_[0] = d;
    }
    // Right
    {
        double hnm2 = h[n-3], hnm1 = h[n-2], mnm2 = m[n-3], mnm1 = m[n-2];
        double d = ((2*hnm1 + hnm2)*mnm1 - hnm1*mnm2) / (hnm1 + hnm2);
        if ((d * mnm1) <= 0.0)
            d_[n-1] = 0.0;
        else if ((mnm1 * mnm2 < 0.0) && (std::abs(d) > 3.0 * std::abs(mnm1)))
            d_[n-1] = 3.0 * mnm1;
        else
            d_[n-1] = d;
    }
}

void PchipInterpolator::compute_coefficients() {
    size_t n = x_.size();
    c_.resize(n-1);
    for (size_t k = 0; k < n-1; ++k) {
        double h = x_[k+1] - x_[k];
        double y0 = y_[k], y1 = y_[k+1];
        double d0 = d_[k], d1 = d_[k+1];
        double slope = (y1 - y0) / h;
        double t = (d0 + d1 - 2 * slope) / h;

        // Cubic coefficients for interval [x_k, x_{k+1}]
        // p(x) = c0*(x-x_k)^3 + c1*(x-x_k)^2 + c2*(x-x_k) + c3
        c_[k][0] = t / h;
        c_[k][1] = (slope - d0) / h - t;
        c_[k][2] = d0;
        c_[k][3] = y0;
    }
}

size_t PchipInterpolator::find_interval(double xval) const {
    // Binary search for interval
    if (xval <= x_.front()) return 0;
    if (xval >= x_.back()) return x_.size() - 2;
    return std::upper_bound(x_.begin(), x_.end(), xval) - x_.begin() - 1;
}

double PchipInterpolator::interpolate(double xval, int nu) const {
    size_t k = find_interval(xval);
    double dx = xval - x_[k];
    const auto& c = c_[k];
    if (nu == 0) {
    return ((c[0]*dx + c[1])*dx + c[2])*dx + c[3];
    } else if (nu == 1) {
        return (3*c[0]*dx + 2*c[1])*dx + c[2];
    } else if (nu == 2) {
        return 6*c[0]*dx + 2*c[1];
    } else if (nu == 3) {
        return 6*c[0];
    } else {
        return 0.0;
    }
}
std::vector<double> PchipInterpolator::interpolate(const std::vector<double>& xvals, int nu) const {
    std::vector<double> results(xvals.size());
    for (size_t i = 0; i < xvals.size(); ++i) {
        results[i] = interpolate(xvals[i], nu);
    }
    return results;
}
std::vector<double> PchipInterpolator::interpolate(const double* xvals, size_t n, int nu) const {
    std::vector<double> results(n);
    for (size_t i = 0; i < n; ++i) {
        results[i] = interpolate(xvals[i], nu);
    }
    return results;
}



std::vector<double> PchipInterpolator::solve(double y, bool discontinuity, bool extrapolate) const {
    // Find all x such that interpolate(x) == y
    // Only search within [x_[0], x_[n-1]]
    std::vector<double> roots;
    size_t n = x_.size();
    for (size_t k = 0; k < n - 1; ++k) {
        // Cubic: c0*(x-xk)^3 + c1*(x-xk)^2 + c2*(x-xk) + c3 - y = 0
        double c0 = c_[k][0];
        double c1 = c_[k][1];
        double c2 = c_[k][2];
        double c3 = c_[k][3] - y;
        // Solve cubic in s = x - x_[k], s in [0, x_[k+1] - x_[k]]
        double h = x_[k+1] - x_[k];
        std::vector<double> local_roots;
        // Use Cardano's method for cubic equation: c0*s^3 + c1*s^2 + c2*s + c3 = 0
        // If c0 == 0, reduce to quadratic/linear
        if (std::abs(c0) < 1e-14) {
            if (std::abs(c1) < 1e-14) {
                // Linear: c2*s + c3 = 0
                if (std::abs(c2) > 1e-14) {
                    double s = -c3 / c2;
                    if (s >= 0 && s <= h)
                        local_roots.push_back(x_[k] + s);
                }
            } else {
                // Quadratic: c1*s^2 + c2*s + c3 = 0
                double D = c2 * c2 - 4 * c1 * c3;
                if (D >= 0) {
                    double sqrtD = std::sqrt(D);
                    double s1 = (-c2 + sqrtD) / (2 * c1);
                    double s2 = (-c2 - sqrtD) / (2 * c1);
                    if (s1 >= 0 && s1 <= h)
                        local_roots.push_back(x_[k] + s1);
                    if (s2 >= 0 && s2 <= h)
                        local_roots.push_back(x_[k] + s2);
                }
            }
        } else {
            // Cubic: use standard roots formula
            // Depressed cubic: s^3 + a*s^2 + b*s + c = 0
            double a = c1 / c0;
            double b = c2 / c0;
            double c = c3 / c0;
            // Convert to depressed cubic: t^3 + p*t + q = 0, t = s + a/3
            double p = b - a * a / 3.0;
            double q = 2.0 * a * a * a / 27.0 - a * b / 3.0 + c;
            double offset = -a / 3.0;
            double discriminant = (q * q) / 4.0 + (p * p * p) / 27.0;
            if (discriminant > 0) {
                // One real root
                double sqrt_disc = std::sqrt(discriminant);
                double A = std::cbrt(-q / 2.0 + sqrt_disc);
                double B = std::cbrt(-q / 2.0 - sqrt_disc);
                double t = A + B;
                double s = t + offset;
                if (s >= 0 && s <= h)
                    local_roots.push_back(x_[k] + s);
            } else {
                // Three real roots
                double r = std::sqrt(-p / 3.0);
                double phi = std::acos(-q / (2.0 * r * r * r));
                for (int j = 0; j < 3; ++j) {
                    double t = 2.0 * r * std::cos((phi + 2.0 * M_PI * j) / 3.0);
                    double s = t + offset;
                    if (s >= 0 && s <= h)
                        local_roots.push_back(x_[k] + s);
                }
            }
        }
        // If discontinuity==false, only take first root in interval
        if (!local_roots.empty()) {
            if (discontinuity)
                roots.insert(roots.end(), local_roots.begin(), local_roots.end());
            else
                roots.push_back(local_roots[0]);
        }
    }
    // If extrapolate==false, filter roots to [x_[0], x_[n-1]]
    if (!extrapolate) {
        roots.erase(std::remove_if(roots.begin(), roots.end(),
            [this](double r) { return r < x_.front() || r > x_.back(); }), roots.end());
    }
    return roots;
}

std::vector<std::vector<double>> PchipInterpolator::solve(const std::vector<double>& yvals, bool discontinuity, bool extrapolate) const {
    std::vector<std::vector<double>> results(yvals.size());
    for (size_t i = 0; i < yvals.size(); ++i) {
        results[i] = solve(yvals[i], discontinuity, extrapolate);
    }
    return results;
}

std::vector<std::vector<double>> PchipInterpolator::solve(const double* yvals, size_t n, bool discontinuity, bool extrapolate) const {
    std::vector<std::vector<double>> results(n);
    for (size_t i = 0; i < n; ++i) {
        results[i] = solve(yvals[i], discontinuity, extrapolate);
    }
    return results;
}