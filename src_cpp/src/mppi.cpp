// 简化版 MPPI 实现：Eigen + 可选 OpenMP 并行
#include "mppi.hpp"
#include <random>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace fourws {

MPPI::MPPI(int nx, int nu,
           DynamicsFn dynamics,
           RunningCostFn running_cost,
           std::optional<TerminalCostFn> terminal_cost,
           const MPPISettings& cfg)
    : nx_(nx), nu_(nu), F_(std::move(dynamics)), cost_(std::move(running_cost)), term_cost_(std::move(terminal_cost)),
      K_(cfg.K), T_(cfg.T), u_per_command_(cfg.u_per_command), lambda_(cfg.lambda) {
    noise_mu_ = (cfg.noise_mu.size() ? cfg.noise_mu : Eigen::VectorXd::Zero(nu_));
    noise_sigma_ = (cfg.noise_sigma.size() ? cfg.noise_sigma : Eigen::MatrixXd::Identity(nu_, nu_));
    noise_sigma_inv_ = noise_sigma_.inverse();
    if (cfg.U_init.size()) {
        U_ = cfg.U_init;
    } else {
        U_.setZero(T_, nu_);
    }
    if (cfg.u_min.size()) u_min_ = cfg.u_min; else u_min_.resize(0);
    if (cfg.u_max.size()) u_max_ = cfg.u_max; else u_max_.resize(0);
    noise_.setZero(K_, T_*nu_);
}

void MPPI::shift_nominal_trajectory() {
    // 前移一格，并将最后一步置零或 u_init=0
    if (U_.rows() <= 1) return;
    U_.topRows(T_-1) = U_.bottomRows(T_-1);
    U_.row(T_-1).setZero();
}

void MPPI::sample_noise() {
    // 从 N(mu, sigma) 采样 K*T 次，存入 noise_ 行向量
    std::mt19937 rng(static_cast<unsigned>(std::random_device{}()));
    // 使用特征分解进行采样：sigma = Q Λ Q^T，z ~ N(0,I)，n = mu + Q sqrt(Λ) z
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(noise_sigma_);
    Eigen::MatrixXd Q = es.eigenvectors();
    Eigen::VectorXd L = es.eigenvalues().cwiseMax(1e-12).cwiseSqrt();
    std::normal_distribution<double> nd(0.0, 1.0);
    for (int k = 0; k < K_; ++k) {
        for (int t = 0; t < T_; ++t) {
            Eigen::VectorXd z(nu_);
            for (int i = 0; i < nu_; ++i) z(i) = nd(rng);
            Eigen::VectorXd n = noise_mu_ + Q * (L.asDiagonal() * z);
            noise_.row(k).segment(t*nu_, nu_) = n.transpose();
        }
    }
}

Eigen::MatrixXd MPPI::rollout_costs(const Eigen::VectorXd& x0,
                                    const Eigen::MatrixXd& perturbed_actions,
                                    Eigen::VectorXd& costs) {
    // 输入：K x T x nu 展平为 K x (T*nu) 的矩阵，逐轨迹 rollout 计算累计成本
    costs.setZero(K_);
    Eigen::MatrixXd actions = perturbed_actions; // K x (T*nu)
    Eigen::MatrixXd states(K_, nx_);
    states.rowwise() = x0.transpose();

    // 并行每条轨迹
    #pragma omp parallel for if(K_>32)
    for (int k = 0; k < K_; ++k) {
        Eigen::VectorXd x = states.row(k).transpose();
        double csum = 0.0;
        for (int t = 0; t < T_; ++t) {
            Eigen::VectorXd u = actions.row(k).segment(t*nu_, nu_).transpose();
            if (u_min_.size() == nu_) u = u.cwiseMax(u_min_);
            if (u_max_.size() == nu_) u = u.cwiseMin(u_max_);
            csum += cost_(x, u, t);
            x = F_(x, u, t);
        }
        if (term_cost_) csum += (*term_cost_)(x);
        costs(k) = csum;
    }
    return actions;
}

Eigen::MatrixXd MPPI::command(const Eigen::VectorXd& state, bool shift_nominal_trajectory) {
    if (shift_nominal_trajectory) this->shift_nominal_trajectory();
    if (U_.size() == 0) U_.setZero(T_, nu_);

    sample_noise();
    // 将 U 展平为 1 x (T*nu)
    Eigen::RowVectorXd Urow(T_*nu_);
    for (int t = 0; t < T_; ++t) {
        Urow.segment(t*nu_, nu_) = U_.row(t);
    }
    // 复制为 K x (T*nu) 并叠加噪声
    Eigen::MatrixXd perturbed(K_, T_*nu_);
    for (int k = 0; k < K_; ++k) {
        perturbed.row(k) = Urow + noise_.row(k);
    }

    // rollout 计算每条轨迹成本
    Eigen::VectorXd costs(K_);
    rollout_costs(state, perturbed, costs);
    // 计算权重 omega
    double beta = costs.minCoeff();
    Eigen::VectorXd cost_non_zero = (-(costs.array() - beta) / lambda_).exp().matrix();
    double eta = cost_non_zero.sum();
    Eigen::VectorXd omega = (1.0 / eta) * cost_non_zero;

    // 计算扰动聚合并更新 U
    Eigen::MatrixXd weighted = omega.replicate(1, T_*nu_).cwiseProduct(noise_);
    Eigen::RowVectorXd perturb = weighted.colwise().sum(); // 1 x (T*nu)
    for (int t = 0; t < T_; ++t) {
        U_.row(t) += perturb.segment(t*nu_, nu_);
    }

    // 输出前 u_per_command 步控制
    if (u_per_command_ == 1) {
        return U_.row(0);
    } else {
        return U_.topRows(u_per_command_);
    }
}

void MPPI::change_horizon(int T_new) {
    if (T_new == T_) return;
    Eigen::MatrixXd U_new = Eigen::MatrixXd::Zero(T_new, nu_);
    int keep = std::min(T_new, T_);
    U_new.topRows(keep) = U_.topRows(keep);
    U_ = std::move(U_new);
    T_ = T_new;
}

void MPPI::reset() {
    U_.setZero(T_, nu_);
}

} // namespace fourws