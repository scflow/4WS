#pragma once
#include <utility>

namespace fourws {

// 摩擦极限下的偏航角速度上限：|r| ≤ mu_y * g / vx_eff
double yaw_rate_limit(double mu_y, double g, double vx_eff, double eps = 1e-6);

// 若 |r| 超限，按增益对 r_dot 进行惩罚
double apply_yaw_saturation(double r, double r_dot, double mu_y, double g, double vx_eff, double gain);

// 2DOF 速度从车体到世界坐标系
std::pair<double,double> body_to_world_2dof(double U, double beta, double psi);

// 不移植 3DOF 相关函数（例如 body_to_world_3dof、slip_angles_3dof），按需后续加入

// 4WS 几何曲率近似：κ ≈ (tan(df) - tan(dr)) / L
double curvature_4ws(double df, double dr, double L);

} // namespace fourws