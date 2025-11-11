// MPPI 接口（简化版），基于 Eigen 并支持并行采样
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <functional>
#include <optional>

namespace fourws {

struct MPPISettings {
    int K{256};          // 采样轨迹数
    int T{15};           // 时域长度
    double lambda{1.0};  // 温度参数
    Eigen::VectorXd noise_mu;      // nu
    Eigen::MatrixXd noise_sigma;   // nu x nu
    Eigen::VectorXd u_min;         // nu（可选）
    Eigen::VectorXd u_max;         // nu（可选）
    Eigen::MatrixXd U_init;        // T x nu（可选）
    int u_per_command{1};          // 每次输出的控制步数
};

// 动力学/成本函数签名
using DynamicsFn = std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&, int)>; // state,u,t -> next_state
using RunningCostFn = std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&, int)>;       // state,u,t -> cost
using TerminalCostFn = std::function<double(const Eigen::VectorXd&)>;                                  // state -> cost

class MPPI {
public:
    MPPI(int nx, int nu,
         DynamicsFn dynamics,
         RunningCostFn running_cost,
         std::optional<TerminalCostFn> terminal_cost,
         const MPPISettings& cfg);

    // 执行一次命令，返回当前应施加的控制（nu 或 u_per_command x nu）
    Eigen::MatrixXd command(const Eigen::VectorXd& state, bool shift_nominal_trajectory = true);

    // 更改时域长度（会同步扩展或截断 U）
    void change_horizon(int T_new);

    // 重置：U 重新采样
    void reset();

    // 读取当前控制序列
    const Eigen::MatrixXd& get_action_sequence() const { return U_; }

private:
    // 内部辅助
    void shift_nominal_trajectory();
    void sample_noise();
    Eigen::MatrixXd rollout_costs(const Eigen::VectorXd& x0,
                                  const Eigen::MatrixXd& perturbed_actions,
                                  Eigen::VectorXd& costs);

private:
    int nx_{0}, nu_{0};
    DynamicsFn F_;
    RunningCostFn cost_;
    std::optional<TerminalCostFn> term_cost_;
    int K_{0}, T_{0}, u_per_command_{1};
    double lambda_{1.0};
    Eigen::VectorXd noise_mu_;
    Eigen::MatrixXd noise_sigma_;
    Eigen::MatrixXd noise_sigma_inv_;
    Eigen::MatrixXd U_;           // T x nu
    Eigen::MatrixXd noise_;       // K x T*nu（展平存储）
    Eigen::VectorXd u_min_, u_max_; // 可选边界
};

} // namespace fourws