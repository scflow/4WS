// 动力学与轮胎模型接口
#pragma once
// 提供 2DOF 自行车模型的侧偏角、横向力与状态导数计算
#include <array>
#include <string>
#include <unordered_map>

namespace fourws {

// 轮胎参数（横向 Pacejka 魔术方程简化）
struct PacejkaParams {
    double B{10.0};
    double C{1.9};
    double E{0.97};
    double mu_y{0.9};
};

// 纵向 Pacejka 参数（纯纵向滑移）
struct PacejkaLongParams {
    double B{12.0};
    double C{1.9};
    double E{1.0};
    double mu_x{0.95};
};

// 车辆参数：直接使用 vehicle.hpp 中的 VehicleParams（避免重复配置）
struct VehicleParams; // 前向声明；实际定义见 vehicle.hpp

// 横向力计算：线性侧偏刚度或 Pacejka 派发
double pacejka_lateral(double alpha, double Fz, const PacejkaParams& p);
double lateral_force_dispatch(double alpha, double Fz, const std::string& model, double linear_k, const PacejkaParams& p);

// 纵向 Pacejka 与摩擦椭圆组合
double pacejka_longitudinal(double slip_ratio, double Fz, const PacejkaLongParams& p);
std::pair<double,double> combine_friction_ellipse(double Fx_pure, double Fy_pure, double Fz, double mu_x, double mu_y);

// 前后轴静态法向载荷（忽略载荷转移）
std::pair<double, double> static_loads(double a, double b, double m, double g);
std::pair<double, double> static_loads_2dof(const VehicleParams& p);

// 2DOF 侧偏角计算：alpha_f/r 由 beta、r、df/dr 与几何参数确定
std::pair<double, double> slip_angles_2dof(double beta, double r, double df, double dr, double a, double b, double U);
std::pair<double, double> slip_angles_2dof(double beta, double r, double df, double dr, const VehicleParams& p);

// 2DOF 横向力（按模型派发）
std::pair<double, double> lateral_forces_2dof(double alpha_f, double alpha_r, const VehicleParams& p);

// 2DOF 状态导数与观测量打包
struct Derivatives2D {
    std::array<double,2> xdot; // [beta_dot, r_dot]
    double ay;
    double Fy_f;
    double Fy_r;
    double alpha_f;
    double alpha_r;
};

// 2DOF 状态导数（直接使用 VehicleParams）
Derivatives2D derivatives_2dof(double beta, double r, double delta_f, double delta_r, const VehicleParams& p);

} // namespace fourws