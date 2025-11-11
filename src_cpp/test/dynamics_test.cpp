#include <iostream>
#include <cmath>
#include "dynamics.hpp"
#include "vehicle.hpp"

using namespace fourws;

static bool approx(double a, double b, double eps = 1e-6) {
    return std::abs(a - b) <= eps;
}

int main() {
    // 基本车辆参数
    VehicleParams p{};
    p.U = 20.0;
    p.mu = 0.85;
    p.tire_model = "linear"; // 先用线性模型，便于验证解析矩阵一致性

    // 侧偏角与横向力
    const double beta = 0.02; // rad
    const double r = 0.1;     // rad/s
    const double df = 0.03;   // rad
    const double dr = -0.01;  // rad

    const auto alphas = slip_angles_2dof(beta, r, df, dr, p);
    const auto forces = lateral_forces_2dof(alphas.first, alphas.second, p);

    // 派生量与观测量
    Derivatives2D d = derivatives_2dof(beta, r, df, dr, p);

    // 简单断言：与线性侧偏刚度关系一致
    bool ok = true;
    ok &= approx(d.alpha_f, alphas.first, 1e-12);
    ok &= approx(d.alpha_r, alphas.second, 1e-12);
    ok &= approx(d.Fy_f, forces.first, 1e-9);
    ok &= approx(d.Fy_r, forces.second, 1e-9);
    // 解析 2DOF 方程校验（用线性模型时与 twodof.py 公式一致）
    const double U = p.U_eff();
    const double beta_dot_ref = (forces.first + forces.second) / (p.m * U) - r;
    const double r_dot_ref = (p.a * forces.first - p.b * forces.second) / p.Iz;
    const double ay_ref = U * (r + beta_dot_ref);
    ok &= approx(d.xdot[0], beta_dot_ref, 1e-9);
    ok &= approx(d.xdot[1], r_dot_ref, 1e-9);
    ok &= approx(d.ay, ay_ref, 1e-9);

    // 纵向 Pacejka 与摩擦椭圆约束测试
    {
        const double Fz = 4000.0; // 单轮法向载荷
        PacejkaLongParams plp;    // 默认：mu_x=0.95
        const double slip = 0.1;  // 10% 纵向滑移
        const double Fx_pure = pacejka_longitudinal(slip, Fz, plp);

        const double mu_y = 0.85;
        const double Fy_pure = 0.9 * mu_y * Fz; // 构造略超出摩擦椭圆的横向力
        auto [Fx, Fy] = combine_friction_ellipse(Fx_pure, Fy_pure, Fz, plp.mu_x, mu_y);
        const double nx = Fx / (plp.mu_x * Fz);
        const double ny = Fy / (mu_y * Fz);
        const double radius2 = nx * nx + ny * ny;
        ok &= (radius2 <= 1.0000001); // 允许极小数值误差
    }

    // 直接使用 VehicleParams 的 2DOF 导数已在上文验证，不再需要适配器

    std::cout << (ok ? "[PASS] dynamics 2DOF basic checks" : "[FAIL] dynamics 2DOF basic checks") << "\n";
    if (!ok) {
        std::cout << "alpha_f=" << d.alpha_f << ", alpha_r=" << d.alpha_r << "\n";
        std::cout << "Fy_f=" << d.Fy_f << ", Fy_r=" << d.Fy_r << "\n";
        std::cout << "beta_dot=" << d.xdot[0] << " vs " << beta_dot_ref << "\n";
        std::cout << "r_dot=" << d.xdot[1] << " vs " << r_dot_ref << "\n";
        std::cout << "ay=" << d.ay << " vs " << ay_ref << "\n";
    }

    return ok ? 0 : 1;
}