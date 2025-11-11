// MPC 实现（移植自 mpc.py，排除旧的 linearize_2dof，重命名以避免 4dof 歧义）
#include "mpc.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <Eigen/Dense>

namespace fourws {

int nearest_plan_index(const std::vector<PlanPoint>& plan, double x, double y) {
    if (plan.empty()) return 0;
    int best_i = 0;
    double best_d = std::numeric_limits<double>::infinity();
    for (int i = 0; i < static_cast<int>(plan.size()); ++i) {
        const auto& p = plan[i];
        double dx = p.x - x;
        double dy = p.y - y;
        double d = dx*dx + dy*dy;
        if (d < best_d) { best_d = d; best_i = i; }
    }
    return best_i;
}

static inline double wrap_pi(double a) {
    double w = std::fmod(a + M_PI, 2.0 * M_PI);
    if (w < 0) w += 2.0 * M_PI;
    return w - M_PI;
}

std::array<double,4> kin_dyn_derivatives(const std::array<double,4>& x_aug,
                                         const std::array<double,2>& u,
                                         const VehicleParams& params,
                                         double U,
                                         double r_ref) {
    double e_y = x_aug[0];
    double e_psi = x_aug[1];
    double beta = x_aug[2];
    double r = x_aug[3];
    double df = u[0];
    double dr = u[1];

    // 运动学误差模型
    double e_y_dot = U * (beta - e_psi);
    double e_psi_dot = -r + r_ref;

    // 动力学（调用 2DOF 封装）
    Dynamics2DOF dyn(params);
    auto d = dyn.derivatives(beta, r, df, dr);
    double beta_dot = d.xdot[0];
    double r_dot = d.xdot[1];

    return {e_y_dot, e_psi_dot, beta_dot, r_dot};
}

std::pair<Mat4, Mat42> linearize_kin_dyn(const VehicleParams& params,
                                         const std::array<double,4>& x0_aug,
                                         const std::array<double,2>& u0,
                                         double dt,
                                         double U,
                                         double r_ref_0) {
    auto base = kin_dyn_derivatives(x0_aug, u0, params, U, r_ref_0);
    const int nx = 4, nu = 2;
    Mat4 A{}; Mat42 B{};
    double eps_x = 1e-4; double eps_u = 1e-3;

    // A: d f / d x
    for (int j = 0; j < nx; ++j) {
        auto x_eps = x0_aug; x_eps[j] += eps_x;
        auto xdot_eps = kin_dyn_derivatives(x_eps, u0, params, U, r_ref_0);
        for (int i = 0; i < nx; ++i) {
            A[i][j] = (xdot_eps[i] - base[i]) / eps_x;
        }
    }
    // B: d f / d u
    for (int j = 0; j < nu; ++j) {
        auto u_eps = u0; u_eps[j] += eps_u;
        auto xdot_eps = kin_dyn_derivatives(x0_aug, u_eps, params, U, r_ref_0);
        for (int i = 0; i < nx; ++i) {
            B[i][j] = (xdot_eps[i] - base[i]) / eps_u;
        }
    }

    // 离散化（欧拉）
    Mat4 A_d{}; Mat42 B_d{};
    // A_d = I + A*dt
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < nx; ++j) {
            A_d[i][j] = (i == j ? 1.0 : 0.0) + A[i][j] * dt;
        }
    }
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < nu; ++j) {
            B_d[i][j] = B[i][j] * dt;
        }
    }
    return {A_d, B_d};
}

std::pair<double,double> solve_mpc_kin_dyn(const std::array<double,4>& state_aug,
                                           const std::array<double,2>& ctrl,
                                           const VehicleParams& params,
                                           const std::vector<PlanPoint>& plan,
                                           double dt,
                                           int H,
                                           double Q_ey,
                                           double Q_epsi,
                                           double Q_beta,
                                           double Q_r,
                                           double R_df,
                                           double R_dr,
                                           double R_delta_df,
                                           double R_delta_dr,
                                           std::optional<double> delta_max) {
    if (plan.empty()) {
        return {ctrl[0], ctrl[1]};
    }

    // 1. 初始状态与输入
    auto x0_raw = state_aug; // [e_y, e_psi, beta, r]
    double df0 = ctrl[0];
    double dr0 = ctrl[1];

    // 2. 参考序列 r_ref[k] 基于曲率
    auto wrap = [](double a){ return wrap_pi(a); };
    // 这里 state_aug 只携带误差，不含绝对位置；为保持与 Python 接口相近，允许传入 plan 与当前位置
    // 若需要严格一致，可扩展 state_aug 结构或增加接口参数。当前用 0 点索引起始。
    int base_i = 0;
    int i_start = std::min(base_i + 1, static_cast<int>(plan.size()) - 1);
    double U_signed = params.U;
    std::vector<double> r_ref_seq(H, 0.0);

    auto seg_psi = [&](int i){
        const auto& a = plan[i];
        const auto& b = plan[i+1];
        double dx = b.x - a.x; double dy = b.y - a.y;
        double ds = std::hypot(dx, dy);
        if (ds > 1e-6) return std::atan2(dy, dx);
        return a.psi;
    };
    int n_plan = static_cast<int>(plan.size());
    for (int k = 0; k < H; ++k) {
        int idx_center = std::min(n_plan - 2, i_start + k);
        int j_prev = std::max(0, idx_center - 1);
        int j_next = std::min(n_plan - 2, idx_center + 1);
        double psi_a = seg_psi(j_prev);
        double psi_b = seg_psi(j_next);
        double dpsi = wrap(psi_b - psi_a);
        double ds_a = std::hypot(plan[j_prev + 1].x - plan[j_prev].x,
                                 plan[j_prev + 1].y - plan[j_prev].y);
        double ds_b = std::hypot(plan[j_next + 1].x - plan[j_next].x,
                                 plan[j_next + 1].y - plan[j_next].y);
        double ds_avg = std::max(1e-6, 0.5 * (ds_a + ds_b));
        double kappa_ref = dpsi / ds_avg;
        r_ref_seq[k] = U_signed * kappa_ref;
    }

    // 3. 线性化（k=0 点一次）
    double r_ref_0 = r_ref_seq[0];
    auto [A_d, B_d] = linearize_kin_dyn(params, x0_raw, {df0, dr0}, dt, U_signed, r_ref_0);
    // 转为 Eigen
    const int nx = 4, nu = 2;
    Eigen::Matrix4d A_de; A_de.setZero();
    Eigen::Matrix<double,4,2> B_de; B_de.setZero();
    for(int i=0;i<4;++i){ for(int j=0;j<4;++j){ A_de(i,j) = A_d[i][j]; }}
    for(int i=0;i<4;++i){ for(int j=0;j<2;++j){ B_de(i,j) = B_d[i][j]; }}

    // 4. 预测矩阵 Phi 与 Tm（完整 H）
    Eigen::MatrixXd Phi(H*nx, nx); Phi.setZero();
    Eigen::MatrixXd Tm(H*nx, H*nu); Tm.setZero();
    Eigen::Matrix4d Ak = Eigen::Matrix4d::Identity();
    for (int k = 0; k < H; ++k) {
        Ak = Ak * A_de;
        Phi.block(k*nx, 0, nx, nx) = Ak;
        for (int j = 0; j <= k; ++j) {
            Eigen::Matrix4d Ad_pow = Eigen::Matrix4d::Identity();
            for (int p = 0; p < (k - j); ++p) Ad_pow = Ad_pow * A_de;
            Tm.block(k*nx, j*nu, nx, nu) = Ad_pow * B_de;
        }
    }

    // 5. 成本矩阵 Qh, Rh, R_delta, 参考 Xref
    Eigen::Matrix4d Qk = Eigen::Matrix4d::Zero();
    Qk(0,0)=Q_ey; Qk(1,1)=Q_epsi; Qk(2,2)=Q_beta; Qk(3,3)=Q_r;
    Eigen::Matrix2d Rk = Eigen::Matrix2d::Zero(); Rk(0,0)=R_df; Rk(1,1)=R_dr;
    Eigen::Matrix2d Rdk = Eigen::Matrix2d::Zero(); Rdk(0,0)=R_delta_df; Rdk(1,1)=R_delta_dr;
    Eigen::MatrixXd Qh = Eigen::MatrixXd::Zero(H*nx, H*nx);
    Eigen::MatrixXd Rh = Eigen::MatrixXd::Zero(H*nu, H*nu);
    Eigen::MatrixXd R_delta = Eigen::MatrixXd::Zero(H*nu, H*nu);
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(H*nu, H*nu);
    Eigen::VectorXd g = Eigen::VectorXd::Zero(H*nu);
    Eigen::VectorXd Xref = Eigen::VectorXd::Zero(H*nx);
    for (int k = 0; k < H; ++k) {
        Qh.block(k*nx, k*nx, nx, nx) = Qk;
        Rh.block(k*nu, k*nu, nu, nu) = Rk;
        R_delta.block(k*nu, k*nu, nu, nu) = Rdk;
        D.block(k*nu, k*nu, nu, nu) = Eigen::Matrix2d::Identity();
        if (k == 0) {
            g.segment(0, nu) << df0, dr0;
        } else {
            D.block(k*nu, (k-1)*nu, nu, nu) = -Eigen::Matrix2d::Identity();
        }
        // Xref block: [0, 0, 0, r_ref[k]]
        Xref(k*nx + 0) = 0.0;
        Xref(k*nx + 1) = 0.0;
        Xref(k*nx + 2) = 0.0;
        Xref(k*nx + 3) = r_ref_seq[k];
    }

    // 6. QP 组装并求解
    Eigen::MatrixXd Hmat = Tm.transpose() * Qh * Tm + Rh + D.transpose() * R_delta * D;
    Eigen::VectorXd y = Phi * Eigen::Vector4d(x0_raw[0], x0_raw[1], x0_raw[2], x0_raw[3]) - Xref;
    Eigen::VectorXd fvec = Tm.transpose() * Qh * y - D.transpose() * R_delta * g;
    Eigen::VectorXd rhs = -fvec;
    // 正定性不保证，采用带正则化的 LDLT，失败则 QR 回退
    const double reg = 1e-8;
    Hmat.diagonal().array() += reg;
    Eigen::VectorXd Useq;
    Eigen::LDLT<Eigen::MatrixXd> ldlt(Hmat);
    if (ldlt.info() == Eigen::Success) {
        Useq = ldlt.solve(rhs);
    }
    if (Useq.size() == 0 || !Useq.allFinite()) {
        Useq = Hmat.colPivHouseholderQr().solve(rhs);
    }

    double df_cmd = Useq.size() >= 1 ? Useq(0) : df0;
    double dr_cmd = Useq.size() >= 2 ? Useq(1) : dr0;
    if (delta_max.has_value()) {
        double m = *delta_max;
        df_cmd = std::clamp(df_cmd, -m, m);
        dr_cmd = std::clamp(dr_cmd, -m, m);
    }
    return {df_cmd, dr_cmd};
}

std::pair<double,double> solve_mpc_kin_dyn(const std::array<double,4>& state_aug,
                                           const std::array<double,2>& ctrl,
                                           const VehicleParams& params,
                                           const std::vector<PlanPoint>& plan,
                                           double dt,
                                           double x_cur,
                                           double y_cur,
                                           double psi_cur,
                                           int H,
                                           double Q_ey,
                                           double Q_epsi,
                                           double Q_beta,
                                           double Q_r,
                                           double R_df,
                                           double R_dr,
                                           double R_delta_df,
                                           double R_delta_dr,
                                           std::optional<double> delta_max) {
    if (plan.empty()) {
        return {ctrl[0], ctrl[1]};
    }
    // Use actual nearest index
    int base_i = nearest_plan_index(plan, x_cur, y_cur);
    // 从最近点切片，确保参考曲率计算围绕当前位置
    std::vector<PlanPoint> plan2;
    for (int i = base_i; i < (int)plan.size(); ++i) plan2.push_back(plan[i]);
    if (plan2.size() < 2) plan2 = plan; // 兜底
    (void)psi_cur; // 当前实现不直接使用 psi_cur
    return solve_mpc_kin_dyn(state_aug, ctrl, params, plan2, dt, H,
                             Q_ey, Q_epsi, Q_beta, Q_r, R_df, R_dr,
                             R_delta_df, R_delta_dr, delta_max);
}

} // namespace fourws