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