// 基本 MPPI 测试：线性系统最小二乘控制
#include "mppi.hpp"
#include <iostream>

using namespace fourws;

int main(){
    int nx = 2, nu = 1;
    // 简单线性系统：x_{t+1} = A x_t + B u_t
    Eigen::Matrix2d A; A << 1.0, 0.1, 0.0, 1.0;
    Eigen::Vector2d B; B << 0.0, 0.1;
    Eigen::Vector2d x_goal(0.0, 0.0);

    DynamicsFn F = [A,B](const Eigen::VectorXd& x, const Eigen::VectorXd& u, int){
        return A * x + B * u(0);
    };
    RunningCostFn C = [x_goal](const Eigen::VectorXd& x, const Eigen::VectorXd& u, int){
        double state_cost = (x - x_goal).squaredNorm();
        double control_cost = 0.01 * u.squaredNorm();
        return state_cost + control_cost;
    };
    TerminalCostFn Ct = [x_goal](const Eigen::VectorXd& x){ return (x - x_goal).squaredNorm(); };

    MPPISettings cfg;
    cfg.K = 128; cfg.T = 20; cfg.lambda = 1.0; cfg.u_per_command = 1;
    cfg.noise_mu = Eigen::VectorXd::Zero(nu);
    cfg.noise_sigma = Eigen::MatrixXd::Identity(nu,nu) * 0.2;
    cfg.U_init = Eigen::MatrixXd::Zero(cfg.T, nu);
    cfg.u_min = Eigen::VectorXd::Constant(nu, -1.0);
    cfg.u_max = Eigen::VectorXd::Constant(nu,  1.0);

    MPPI mppi(nx, nu, F, C, Ct, cfg);
    Eigen::Vector2d x0(1.0, 0.0);
    auto u = mppi.command(x0, true);
    double u0 = u(0,0);
    std::cout << "u0=" << u0 << std::endl;
    if (!std::isfinite(u0)) return 1;
    return 0;
}