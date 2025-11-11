// 轨迹规划实现：五次多项式与单圆弧插值
#include "planner.hpp"
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>

namespace fourws {

static inline double wrap_pi(double a) {
    const double two_pi = 2.0 * M_PI;
    double x = std::fmod(a + M_PI, two_pi);
    if (x < 0) x += two_pi;
    return x - M_PI;
}

static inline double wrap_2pi(double a) {
    const double two_pi = 2.0 * M_PI;
    double x = std::fmod(a, two_pi);
    if (x < 0) x += two_pi;
    return x;
}

std::vector<double> solve_quintic(double p0, double v0, double a0,
                                  double pT, double vT, double aT,
                                  double T) {
    T = (T <= 0.0) ? 1e-6 : T;
    // 6x6 线性方程求系数 a0..a5
    const double T2 = T*T;
    const double T3 = T2*T;
    const double T4 = T3*T;
    const double T5 = T4*T;

    // 矩阵行列式求解（显式求逆不必要，这里用 Cramer/直接构造解）
    // 使用标准推导：
    // a0=p0, a1=v0, a2=a0/2
    // a3=(20*(pT-p0) - (8*vT+12*v0)*T - (3*a0-aT)*T^2) / (2*T^3)
    // a4=(30*(p0-pT) + (14*vT+16*v0)*T + (3*a0-2*aT)*T^2) / (2*T^4)
    // a5=(12*(pT-p0) - (6*vT+6*v0)*T - (a0-aT)*T^2) / (2*T^5)
    std::vector<double> a(6);
    a[0] = p0;
    a[1] = v0;
    a[2] = a0 / 2.0;
    a[3] = (20.0*(pT - p0) - (8.0*vT + 12.0*v0)*T - (3.0*a0 - aT)*T2) / (2.0*T3);
    a[4] = (30.0*(p0 - pT) + (14.0*vT + 16.0*v0)*T + (3.0*a0 - 2.0*aT)*T2) / (2.0*T4);
    a[5] = (12.0*(pT - p0) - (6.0*vT + 6.0*v0)*T - (a0 - aT)*T2) / (2.0*T5);
    return a;
}

void sample_poly(const std::vector<double>& coeffs,
                 const std::vector<double>& t,
                 std::vector<double>& p,
                 std::vector<double>& v,
                 std::vector<double>& a) {
    const double a0 = coeffs[0];
    const double a1 = coeffs[1];
    const double a2 = coeffs[2];
    const double a3 = coeffs[3];
    const double a4 = coeffs[4];
    const double a5 = coeffs[5];
    const size_t N = t.size();
    p.resize(N); v.resize(N); a.resize(N);
    for (size_t i = 0; i < N; ++i) {
        const double tt = t[i];
        const double tt2 = tt*tt;
        const double tt3 = tt2*tt;
        const double tt4 = tt3*tt;
        const double tt5 = tt4*tt;
        p[i] = a0 + a1*tt + a2*tt2 + a3*tt3 + a4*tt4 + a5*tt5;
        v[i] = a1 + 2.0*a2*tt + 3.0*a3*tt2 + 4.0*a4*tt3 + 5.0*a5*tt4;
        a[i] = 2.0*a2 + 6.0*a3*tt + 12.0*a4*tt2 + 20.0*a5*tt3;
    }
}

std::vector<PlanPoint> plan_quintic_xy(const Pose2D& start,
                                       const Pose2D& end,
                                       double T,
                                       int N,
                                       double U_start) {
    const double x0 = start.x, y0 = start.y, psi0 = start.psi;
    const double xT = end.x,   yT = end.y,   psiT = end.psi;
    const double vx0 = U_start * std::cos(psi0);
    const double vy0 = U_start * std::sin(psi0);
    const double vxT = U_start * std::cos(psiT);
    const double vyT = U_start * std::sin(psiT);

    const auto cx = solve_quintic(x0, vx0, 0.0, xT, vxT, 0.0, T);
    const auto cy = solve_quintic(y0, vy0, 0.0, yT, vyT, 0.0, T);

    std::vector<double> t(N);
    for (int i = 0; i < N; ++i) t[i] = (N==1) ? 0.0 : (T * double(i) / double(N-1));
    std::vector<double> x, vx, ax;
    std::vector<double> y, vy, ay;
    sample_poly(cx, t, x, vx, ax);
    sample_poly(cy, t, y, vy, ay);

    std::vector<PlanPoint> plan; plan.reserve(N);
    for (int i = 0; i < N; ++i) {
        const double psi = std::atan2(vy[i], vx[i]);
        plan.push_back(PlanPoint{t[i], x[i], y[i], psi});
    }
    return plan;
}

std::vector<PlanPoint> plan_circle_arc(const Pose2D& start,
                                       const Pose2D& end,
                                       double T,
                                       int N,
                                       double /*U_start*/) {
    const double x0 = start.x, y0 = start.y, psi0 = start.psi;
    const double x1 = end.x,   y1 = end.y,   psi1 = end.psi;
    const double dx = x1 - x0, dy = y1 - y0;
    const double dnorm = std::hypot(dx, dy);
    if (dnorm < 1e-6) {
        std::vector<PlanPoint> out; out.reserve(N);
        const double Tuse = std::max(1e-3, T);
        for (int i = 0; i < N; ++i) {
            const double tt = (N==1) ? 0.0 : (Tuse * double(i) / double(N-1));
            out.push_back(PlanPoint{tt, x0, y0, psi0});
        }
        return out;
    }

    auto try_dir = [&](int dir_s, bool& ok, double& r, std::array<double,2>& C,
                       double& theta0, double& theta1, double& dtheta, double& err_end) {
        const double nx = std::cos(psi0 + dir_s * M_PI / 2.0);
        const double ny = std::sin(psi0 + dir_s * M_PI / 2.0);
        const double denom = dx*nx + dy*ny;
        if (std::abs(denom) < 1e-8) { ok = false; return; }
        r = (dnorm*dnorm) / (2.0 * denom);
        if (!std::isfinite(r) || r <= 1e-6) { ok = false; return; }
        C = {x0 + r*nx, y0 + r*ny};
        theta0 = std::atan2(y0 - C[1], x0 - C[0]);
        theta1 = std::atan2(y1 - C[1], x1 - C[0]);
        if (dir_s > 0) {
            dtheta = wrap_2pi(theta1 - theta0);
        } else {
            dtheta = -wrap_2pi(theta0 - theta1);
        }
        const double psi1_pred = theta1 + dir_s * M_PI / 2.0;
        err_end = std::abs(wrap_pi(psi1_pred - psi1));
        ok = true;
    };

    bool okL=false, okR=false; double rL=0, rR=0; std::array<double,2> CL{}, CR{};
    double th0L=0, th1L=0, dthL=0, errL=0;
    double th0R=0, th1R=0, dthR=0, errR=0;
    try_dir(+1, okL, rL, CL, th0L, th1L, dthL, errL);
    try_dir(-1, okR, rR, CR, th0R, th1R, dthR, errR);
    bool useL = false; bool useR = false;
    if (okL && okR) {
        useL = (errL <= errR); useR = !useL;
    } else if (okL) {
        useL = true;
    } else if (okR) {
        useR = true;
    } else {
        // 退化为直线插值
        std::vector<PlanPoint> out; out.reserve(N);
        const double psi_line = std::atan2(dy, dx);
        const double Tuse = std::max(1e-3, T);
        for (int i = 0; i < N; ++i) {
            const double s = (N==1) ? 0.0 : double(i) / double(N-1);
            const double tt = (N==1) ? 0.0 : (Tuse * s);
            const double xx = x0 + s*dx;
            const double yy = y0 + s*dy;
            out.push_back(PlanPoint{tt, xx, yy, psi_line});
        }
        return out;
    }

    const bool Lsel = useL;
    const double r = Lsel ? rL : rR;
    const auto& C = Lsel ? CL : CR;
    const double theta0 = Lsel ? th0L : th0R;
    const double dtheta = Lsel ? dthL : dthR;
    const int dir_s = Lsel ? +1 : -1;

    // 采样角度与时间
    std::vector<PlanPoint> out; out.reserve(N);
    const double Tuse = std::max(1e-3, T);
    for (int i = 0; i < N; ++i) {
        const double s = (N==1) ? 0.0 : double(i) / double(N-1);
        const double tt = (N==1) ? 0.0 : (Tuse * s);
        const double theta = theta0 + s * dtheta;
        const double xx = C[0] + r * std::cos(theta);
        const double yy = C[1] + r * std::sin(theta);
        const double psi = theta + dir_s * M_PI / 2.0;
        out.push_back(PlanPoint{tt, xx, yy, psi});
    }
    return out;
}

} // namespace fourws