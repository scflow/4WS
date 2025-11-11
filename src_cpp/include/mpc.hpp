// MPC 接口与数据结构（移植自 mpc.py，排除旧的 linearize_2dof）
#pragma once
#include <vector>
#include <array>
#include <optional>
#include <utility>
#include "vehicle.hpp"
#include "dynamics2d.hpp"

namespace fourws {

// 轨迹点（与 Python plan 条目对应）
struct PlanPoint {
    double t{0.0};
    double x{0.0};
    double y{0.0};
    double psi{0.0};
};

// 最近参考点查找
int nearest_plan_index(const std::vector<PlanPoint>& plan, double x, double y);

// 运动学 + 2DOF 动力学的增广导数（无 "4dof" 术语）
// x_aug = [e_y, e_psi, beta, r], u = [df, dr]
std::array<double,4> kin_dyn_derivatives(const std::array<double,4>& x_aug,
                                         const std::array<double,2>& u,
                                         const VehicleParams& params,
                                         double U,
                                         double r_ref);

// 数值线性化并离散化：返回 (A_d, B_d)
// 注意：按用户要求，不移植旧的 linearize_2dof；此为增广模型的线性化
using Mat4 = std::array<std::array<double,4>,4>;
using Mat42 = std::array<std::array<double,2>,4>;
std::pair<Mat4, Mat42> linearize_kin_dyn(const VehicleParams& params,
                                         const std::array<double,4>& x0_aug,
                                         const std::array<double,2>& u0,
                                         double dt,
                                         double U,
                                         double r_ref_0);

// 基于增广模型的 MPC 求解器（返回 df/dr），命名避免 "4dof" 歧义
std::pair<double,double> solve_mpc_kin_dyn(const std::array<double,4>& state_aug,
                                           const std::array<double,2>& ctrl,
                                           const VehicleParams& params,
                                           const std::vector<PlanPoint>& plan,
                                           double dt,
                                           int H = 10,
                                           double Q_ey = 10.0,
                                           double Q_epsi = 5.0,
                                           double Q_beta = 0.1,
                                           double Q_r = 0.1,
                                           double R_df = 0.5,
                                           double R_dr = 0.5,
                                           double R_delta_df = 1.0,
                                           double R_delta_dr = 1.0,
                                           std::optional<double> delta_max = std::nullopt);

// 重载：提供当前位置 (x,y,psi) 以使用最近路径段（更贴近 Python 实现）
std::pair<double,double> solve_mpc_kin_dyn(const std::array<double,4>& state_aug,
                                           const std::array<double,2>& ctrl,
                                           const VehicleParams& params,
                                           const std::vector<PlanPoint>& plan,
                                           double dt,
                                           double x_cur,
                                           double y_cur,
                                           double psi_cur,
                                           int H = 10,
                                           double Q_ey = 10.0,
                                           double Q_epsi = 5.0,
                                           double Q_beta = 0.1,
                                           double Q_r = 0.1,
                                           double R_df = 0.5,
                                           double R_dr = 0.5,
                                           double R_delta_df = 1.0,
                                           double R_delta_dr = 1.0,
                                           std::optional<double> delta_max = std::nullopt);

} // namespace fourws