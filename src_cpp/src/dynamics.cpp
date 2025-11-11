// 动力学实现：
// - Pacejka 横向力与线性派发
// - 静态法向载荷
// - 2DOF 侧偏角与横向力
// - 2DOF 状态导数与观测量
#include "dynamics.hpp"
#include "vehicle.hpp"
#include <cmath>
#include <algorithm>

namespace fourws {

// 有效纵向速度：使用 VehicleParams 的 U_eff()

// Pacejka 横向力（纯侧偏），方向约定为与 Python 相同
double pacejka_lateral(double alpha, double Fz, const PacejkaParams& p) {
    Fz = std::max(0.0, Fz);
    const double D = p.mu_y * Fz;
    const double Ba = p.B * alpha;
    const double atan_Ba = std::atan(Ba);
    const double inner = Ba - p.E * (Ba - atan_Ba);
    const double Fy = -D * std::sin(p.C * std::atan(inner));
    return Fy;
}

// 横向力派发：当 model=="linear" 使用 -k*alpha，否则调用 Pacejka
double lateral_force_dispatch(double alpha, double Fz, const std::string& model, double linear_k, const PacejkaParams& p) {
    const std::string m = model.empty() ? std::string("linear") : model;
    if (m == "linear") {
        return -linear_k * alpha;
    }
    return pacejka_lateral(alpha, Fz, p);
}

// 前后轴静态法向载荷（忽略载荷转移）
std::pair<double,double> static_loads(double a, double b, double m, double g) {
    double L = a + b;
    if (std::abs(L) < 1e-9) L = 1e-9;
    const double Fzf = m * g * (b / L);
    const double Fzr = m * g * (a / L);
    return {Fzf, Fzr};
}

// 封装 2DOF 车辆参数的载荷计算
std::pair<double,double> static_loads_2dof(const VehicleParams& p) {
    return static_loads(p.a, p.b, p.m, p.g);
}

// 2DOF 侧偏角计算：alpha_f/r 由 beta、r、df/dr 与几何参数确定
std::pair<double,double> slip_angles_2dof(double beta, double r, double df, double dr, double a, double b, double U) {
    const double alpha_f = beta + a * r / U - df;
    const double alpha_r = beta - b * r / U - dr;
    return {alpha_f, alpha_r};
}

std::pair<double,double> slip_angles_2dof(double beta, double r, double df, double dr, const VehicleParams& p) {
    const double U = p.U_eff();
    return slip_angles_2dof(beta, r, df, dr, p.a, p.b, U);
}

// 2DOF 横向力（统一派发）
std::pair<double,double> lateral_forces_2dof(double alpha_f, double alpha_r, const VehicleParams& p) {
    const std::string model_sel = p.tire_model.empty() ? std::string("linear") : p.tire_model;
    const auto [Fzf, Fzr] = static_loads_2dof(p);
    const PacejkaParams tp_f{10.0, 1.9, 0.97, p.mu};
    const PacejkaParams tp_r{10.0, 1.9, 0.97, p.mu};
    const double Fy_f = lateral_force_dispatch(alpha_f, Fzf, model_sel, p.kf, tp_f);
    const double Fy_r = lateral_force_dispatch(alpha_r, Fzr, model_sel, p.kr, tp_r);
    return {Fy_f, Fy_r};
}

// 2DOF 状态导数与观测量：
// beta_dot = (Fy_f + Fy_r)/(m*U) - r
// r_dot    = (a*Fy_f - b*Fy_r)/Iz
// ay       = U * (r + beta_dot)
Derivatives2D derivatives_2dof(double beta, double r, double delta_f, double delta_r, const VehicleParams& p) {
    const auto [alpha_f, alpha_r] = slip_angles_2dof(beta, r, delta_f, delta_r, p);
    const auto [Fy_f, Fy_r] = lateral_forces_2dof(alpha_f, alpha_r, p);
    const double U = p.U_eff();

    const double beta_dot = (Fy_f + Fy_r) / (p.m * U) - r;
    const double r_dot = (p.a * Fy_f - p.b * Fy_r) / p.Iz;
    const double ay = U * (r + beta_dot);

    Derivatives2D out;
    out.xdot = {beta_dot, r_dot};
    out.ay = ay;
    out.Fy_f = Fy_f;
    out.Fy_r = Fy_r;
    out.alpha_f = alpha_f;
    out.alpha_r = alpha_r;
    return out;
}

// 纵向 Pacejka（纯滑移），方向取正：驱动为正 Fx
double pacejka_longitudinal(double slip_ratio, double Fz, const PacejkaLongParams& p) {
    Fz = std::max(0.0, Fz);
    const double D = p.mu_x * Fz;
    const double Bl = p.B * slip_ratio;
    const double atan_Bl = std::atan(Bl);
    const double inner = Bl - p.E * (Bl - atan_Bl);
    const double Fx = D * std::sin(p.C * std::atan(inner));
    return Fx;
}

// 摩擦椭圆组合：若超过椭圆，按半径缩放至边界
std::pair<double,double> combine_friction_ellipse(double Fx_pure, double Fy_pure, double Fz, double mu_x, double mu_y) {
    Fz = std::max(0.0, Fz);
    if (Fz <= 1e-6) return {0.0, 0.0};
    const double nx = Fx_pure / (mu_x * Fz + 1e-12);
    const double ny = Fy_pure / (mu_y * Fz + 1e-12);
    const double radius2 = nx * nx + ny * ny;
    if (radius2 <= 1.0) return {Fx_pure, Fy_pure};
    const double scale = 1.0 / std::sqrt(radius2);
    return {Fx_pure * scale, Fy_pure * scale};
}

// 适配器不再需要，直接使用 VehicleParams 的接口

} // namespace fourws