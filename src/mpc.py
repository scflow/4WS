import numpy as np
from typing import List, Dict, Tuple, Optional
from .twodof import derivatives as deriv_2dof

# --- 旧的 2-DOF 函数 (保留) ---

def nearest_plan_index(plan: List[Dict[str, float]], x: float, y: float) -> int:
    """在参考轨迹中寻找与 (x,y) 最近的点索引。"""
    if not plan:
        return 0
    best_i = 0
    best_d = float("inf")
    for i, p in enumerate(plan):
        dx = float(p.get('x', 0.0)) - x
        dy = float(p.get('y', 0.0)) - y
        d = dx * dx + dy * dy
        if d < best_d:
            best_d = d
            best_i = i
    return best_i

def linearize_2dof(params, x_vec: np.ndarray, df0: float, dr0: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """[旧] 数值线性化 2DOF 并离散化：返回 (A_d, B_d)。"""
    base = deriv_2dof(x_vec, df0, dr0, params)
    xdot0 = np.array(base["xdot"], dtype=float)
    nx = 2; nu = 2
    A = np.zeros((nx, nx), dtype=float); B = np.zeros((nx, nu), dtype=float)
    eps_x = 1e-4; eps_u = 1e-3
    for j in range(nx):
        x_eps = np.array(x_vec, dtype=float); x_eps[j] += eps_x
        xdot_eps = np.array(deriv_2dof(x_eps, df0, dr0, params)["xdot"], dtype=float)
        A[:, j] = (xdot_eps - xdot0) / eps_x
    u0 = np.array([df0, dr0], dtype=float)
    for j in range(nu):
        u_eps = np.array(u0, dtype=float); u_eps[j] += eps_u
        xdot_eps = np.array(deriv_2dof(x_vec, float(u_eps[0]), float(u_eps[1]), params)["xdot"], dtype=float)
        B[:, j] = (xdot_eps - xdot0) / eps_u
    A_d = np.eye(nx) + A * float(dt); B_d = B * float(dt)
    return A_d, B_d

# --- 新的 4-DOF 轨迹跟踪模型 ---

def get_kin_dyn_4dof_derivatives(
    x_aug: np.ndarray, 
    u: np.ndarray, 
    params, 
    r_ref: float,
    U: float
) -> np.ndarray:
    """
    计算 4-DOF 增广状态的导数: x_aug = [e_y, e_psi, beta, r]
    返回: x_dot_aug = [e_y_dot, e_psi_dot, beta_dot, r_dot]
    """
    e_y, e_psi, beta, r = float(x_aug[0]), float(x_aug[1]), float(x_aug[2]), float(x_aug[3])
    df, dr = float(u[0]), float(u[1])
    
    # 1. 运动学误差模型 (Kinematic Error Model)
    # 假设 U 恒定 (来自 params)
    # de_y/dt = V * sin(e_psi) + V_y (其中 V_y = V * beta)
    # 我们线性化：sin(e_psi) ≈ e_psi, cos(e_psi) ≈ 1
    # V_y_global = (U * beta) * cos(e_psi) - (U) * sin(e_psi) <-- 这个太复杂
    # 简化版：de_y/dt ≈ U * e_psi + U * beta 
    # (U*e_psi 来自航向误差, U*beta 来自侧滑)
    e_y_dot = U * e_psi + U * beta

    # de_psi/dt = r - r_ref
    e_psi_dot = r - r_ref

    # 2. 动力学模型 (Dynamic Model)
    # [beta_dot, r_dot] 来自 twodof
    x_dyn = np.array([beta, r])
    d = deriv_2dof(x_dyn, df, dr, params)
    beta_dot, r_dot = float(d["xdot"][0]), float(d["xdot"][1])
    
    return np.array([e_y_dot, e_psi_dot, beta_dot, r_dot])

def linearize_kin_dyn_4dof(
    params, 
    x0_aug: np.ndarray, 
    u0: np.ndarray, 
    r_ref_0: float, 
    dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    数值线性化 4-DOF 轨迹跟踪模型 (LTI)
    状态 x_aug = [e_y, e_psi, beta, r]
    输入 u = [df, dr]
    """
    U = params.U_eff() # 获取当前速度
    base = get_kin_dyn_4dof_derivatives(x0_aug, u0, params, r_ref_0, U)
    xdot0 = np.array(base, dtype=float)
    
    nx = 4
    nu = 2
    A = np.zeros((nx, nx), dtype=float)
    B = np.zeros((nx, nu), dtype=float)
    eps_x = 1e-4
    eps_u = 1e-3

    # A: 对状态 x_aug 求导
    for j in range(nx):
        x_eps = np.array(x0_aug, dtype=float)
        x_eps[j] += eps_x
        xdot_eps = get_kin_dyn_4dof_derivatives(x_eps, u0, params, r_ref_0, U)
        A[:, j] = (xdot_eps - xdot0) / eps_x
        
    # B: 对控制 u 求导
    for j in range(nu):
        u_eps = np.array(u0, dtype=float)
        u_eps[j] += eps_u
        xdot_eps = get_kin_dyn_4dof_derivatives(x0_aug, u_eps, params, r_ref_0, U)
        B[:, j] = (xdot_eps - xdot0) / eps_u
        
    # 离散化（欧拉）
    A_d = np.eye(nx) + A * float(dt)
    B_d = B * float(dt)
    return A_d, B_d

def solve_mpc_kin_dyn_4dof(
    state_aug: Dict[str, float],
    ctrl: Dict[str, float],
    params,
    plan: List[Dict[str, float]],
    dt: float,
    H: int = 10,
    Q_ey: float = 10.0,
    Q_epsi: float = 5.0,
    Q_beta: float = 0.1,
    Q_r: float = 0.1,
    R_df: float = 0.5,
    R_dr: float = 0.5,
    R_delta_df: float = 1.0,
    R_delta_dr: float = 1.0,
    delta_max: Optional[float] = None,
) -> Tuple[float, float]:
    """
    基于 4-DOF [ey, epsi, beta, r] 模型的 MPC 求解器
    """
    if not plan:
        return float(ctrl.get('delta_f', 0.0)), float(ctrl.get('delta_r', 0.0))

    # --- 1. 初始状态 ---
    x0_raw = np.array([
        float(state_aug.get('e_y', 0.0)),
        float(state_aug.get('e_psi', 0.0)),
        float(state_aug.get('beta', 0.0)),
        float(state_aug.get('r', 0.0))
    ], dtype=float)
    df0 = float(ctrl.get('delta_f', 0.0))
    dr0 = float(ctrl.get('delta_r', 0.0))
    u0 = np.array([df0, dr0], dtype=float)

    # --- 2. 参考序列 ---
    def wrap(a: float) -> float:
        return float((a + np.pi) % (2*np.pi) - np.pi)
    
    x = float(state_aug.get('x', 0.0))
    y = float(state_aug.get('y', 0.0))
    psi0 = float(state_aug.get('psi', 0.0))
    U = params.U_eff()
    
    base_i = nearest_plan_index(plan, x, y)
    i_start = min(base_i + 1, len(plan) - 1)

    r_ref_seq = np.zeros(H, dtype=float)
    # 我们需要 H+1 个点来计算 H 个 r_ref
    psi_list: List[float] = []
    for k in range(H + 1):
        i_k = min(len(plan) - 1, i_start + k)
        psi_list.append(float(plan[i_k].get('psi', psi0)))

    for k in range(H):
        dpsi_k = wrap(psi_list[k + 1] - psi_list[k])
        # r_ref = dpsi/dt = dpsi/(ds/V) = V * (dpsi/ds) = V * kappa
        r_ref_seq[k] = dpsi_k / max(float(dt), 1e-6)

    # --- 3. 线性化 ---
    # 使用 LTI MPC：只在 k=0 处线性化一次
    r_ref_0 = r_ref_seq[0]
    A_d, B_d = linearize_kin_dyn_4dof(params, x0_raw, u0, r_ref_0, dt)

    # --- 4. 预测矩阵 Φ 和 T ---
    nx, nu = 4, 2
    Phi = np.zeros((H * nx, nx), dtype=float)
    Tm = np.zeros((H * nx, H * nu), dtype=float)
    Ak = np.eye(nx)
    for k in range(H):
        Ak_plus_1 = Ak @ A_d
        Phi[k*nx:(k+1)*nx, :] = Ak_plus_1
        for j in range(k + 1):
            if k >= j:
                power = k - j
                Ad_pow = np.linalg.matrix_power(A_d, power)
                block = Ad_pow @ B_d
                Tm[k*nx:(k+1)*nx, j*nu:(j+1)*nu] = block
        Ak = Ak_plus_1

    # --- 5. 成本矩阵 Q/R ---
    Qh = np.zeros((H * nx, H * nx), dtype=float)
    Rh = np.zeros((H * nu, H * nu), dtype=float)
    for k in range(H):
        Qk = np.diag([Q_ey, Q_epsi, Q_beta, Q_r])
        Rk = np.diag([R_df, R_dr])
        Qh[k*nx:(k+1)*nx, k*nx:(k+1)*nx] = Qk
        Rh[k*nu:(k+1)*nu, k*nu:(k+1)*nu] = Rk

    # --- 6. 参考状态 Xref ---
    # 我们的目标是让 e_y -> 0, e_psi -> 0, beta -> 0, r -> r_ref
    Xref = np.zeros(H * nx, dtype=float)
    for k in range(H):
        Xref[k*nx + 0] = 0.0  # e_y_ref
        Xref[k*nx + 1] = 0.0  # e_psi_ref
        Xref[k*nx + 2] = 0.0  # beta_ref
        Xref[k*nx + 3] = r_ref_seq[k] # r_ref

    # --- 7. 控制变率惩罚 D, g, R_delta ---
    D = np.zeros((H * nu, H * nu), dtype=float)
    g = np.zeros(H * nu, dtype=float)
    Iu = np.eye(nu)
    u_prev = np.array([df0, dr0], dtype=float)
    for k in range(H):
        D[k*nu:(k+1)*nu, k*nu:(k+1)*nu] = Iu
        if k == 0:
            g[0:nu] = u_prev
        else:
            D[k*nu:(k+1)*nu, (k-1)*nu:k*nu] = -Iu
    R_delta = np.zeros((H * nu, H * nu), dtype=float)
    for k in range(H):
        R_delta[k*nu:(k+1)*nu, k*nu:(k+1)*nu] = np.diag([R_delta_df, R_delta_dr])

    # --- 8. QP 组装 (H U = -f) ---
    # 注意：我们不再需要 S 矩阵，因为 e_psi 已经是状态
    Hmat = (
        Tm.T @ Qh @ Tm
        + Rh
        + D.T @ R_delta @ D
    )
    fvec = (
        (Phi @ x0_raw - Xref).T @ Qh @ Tm
        - g.T @ R_delta @ D
    ).T
    
    # 求解: Hmat * U = -fvec
    try:
        Useq = np.linalg.solve(Hmat, -fvec)
    except np.linalg.LinAlgError:
        Useq = np.linalg.pinv(Hmat) @ -fvec

    df_cmd = float(Useq[0])
    dr_cmd = float(Useq[1])
    
    if delta_max is not None:
        df_cmd = float(np.clip(df_cmd, -delta_max, delta_max))
        dr_cmd = float(np.clip(dr_cmd, -delta_max, delta_max))
    return df_cmd, dr_cmd