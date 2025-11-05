import numpy as np
import torch
from typing import Callable, List, Dict, Optional

from .params import VehicleParams
from .twodof_torch import slip_angles_2dof_torch, lateral_forces_2dof_torch
from .pytorch_mppi.mppi import MPPI
from .strategy import ideal_yaw_rate


class MPPIController4WS:
    """4WS 车辆的 MPPI 控制封装。

    - 动力学：复用 twodof 的物理公式（Torch 版），在 MPPI 内进行批量滚动。
    - 状态：s = [x, y, psi, beta, r, U, delta_f, delta_r]
    - 动作：u = [d_delta_f, d_delta_r, dU]（转角与速度均使用微分形式）
    """

    def __init__(
        self,
        params: VehicleParams,
        dt: float,
        plan_provider: Callable[[], List[Dict[str, float]]],
        delta_max: float,
        dU_max: float,
        U_max: float,
        device: Optional[str] = None,
        num_samples: int = 16,
        horizon: int = 20,
        lambda_: float = 120.0,
        delta_rate_frac: float = 0.5,
    ):
        self.params = params
        self.dt = float(dt)
        self.plan_provider = plan_provider
        self.delta_max = float(delta_max)
        self.dU_max = float(dU_max)
        self.U_max = float(U_max)
        # 每秒角度变化上限占比（相对于 delta_max），默认 0.5 更保守
        # 可调以加快转角速度，但不改变最终角度幅度上限
        self.delta_rate_frac = float(delta_rate_frac)
        # prefer MPS on macOS if available, else CUDA, else CPU
        try:
            mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        except Exception:
            mps_available = False
        self.device = device or ('mps' if mps_available else ('cuda' if torch.cuda.is_available() else 'cpu'))

        self.w_lat = 2000.0
        self.w_head = 4000.0
        self.w_speed = 20.0
        self.w_u = 1.0
        self.w_du = 2.0
        # 速度相关的相位约束（低速更强反相，高速弱同相）
        self.w_phase_base = 300.0
        self.k_low = -0.30
        self.k_high = 0.10
        self.U1 = 5.0
        self.U2 = 20.0
        # 额外约束：直接惩罚同相（df、dr 同号）与加入理想横摆率前馈
        self.w_phase_sign = 120.0
        self.w_yaw_ff = 80.0

        # 控制噪声协方差（三维动作，使用微分形式）
        noise_sigma = torch.diag(torch.tensor([
            0.03,  # d_delta_f 每步角度增量
            0.03,  # d_delta_r 每步角度增量
            0.15   # dU 速度增量
        ], dtype=torch.float32))

        # 动作边界（微分形式：转角增量限制为每秒上限的 dt 倍）
        # 角速度上限：delta_rate_frac * delta_max / s（可由外部配置传入）
        # 例如：delta_rate_frac=0.8 表示 1 秒可接近 0.8*delta_max 的角度变化
        delta_rate_max = float(self.delta_rate_frac) * self.delta_max  # rad/s
        d_delta_bound = delta_rate_max * self.dt
        u_min = torch.tensor([
            -d_delta_bound,
            -d_delta_bound,
            -self.dU_max * self.dt
        ], dtype=torch.float32)
        u_max = torch.tensor([
            d_delta_bound,
            d_delta_bound,
            self.dU_max * self.dt
        ], dtype=torch.float32)

        self.mppi = MPPI(
            dynamics=self._dynamics,
            running_cost=self._running_cost,
            nx=8,  # 状态维度: [x, y, psi, beta, r, U, delta_f, delta_r]
            noise_sigma=noise_sigma,
            num_samples=num_samples,
            horizon=horizon,
            lambda_=lambda_,
            u_min=u_min,
            u_max=u_max,
            device=self.device,
        )

        # 缓存上一时刻动作用于平滑代价
        self._last_u = None

    # --- 动力学：f(s, u) -> s_next ---
    def _dynamics(self, s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # s: [batch?, 6]; u: [batch?, 3]
        # 支持单样本和批量。内部全部使用张量广播。
        if s.ndim == 1:
            s = s.unsqueeze(0)
        if u.ndim == 1:
            u = u.unsqueeze(0)

        # 解包状态（支持 6/8 维；默认无转角时从 0 起始）
        if s.shape[-1] >= 8:
            x, y, psi, beta, r, U, df_cur, dr_cur = s.unbind(-1)
        else:
            x, y, psi, beta, r, U = s.unbind(-1)
            df_cur = torch.zeros_like(U)
            dr_cur = torch.zeros_like(U)

        # 解包动作：转角增量与速度增量
        d_df, d_dr, dU = u.unbind(-1)

        # 速度更新（带边界）
        U_next = torch.clamp(U + dU, 0.0, torch.tensor(self.U_max, dtype=s.dtype, device=s.device))
        # 角度更新（限幅）
        df_next = torch.clamp(df_cur + d_df, -self.delta_max, self.delta_max)
        dr_next = torch.clamp(dr_cur + d_dr, -self.delta_max, self.delta_max)

        # 侧偏角与横向力（复用 twodof 公式，但 U 使用当前状态值）
        alpha_f, alpha_r = slip_angles_2dof_torch(beta, r, df_next, dr_next, float(self.params.a), float(self.params.b), U_next)
        Fy_f, Fy_r = lateral_forces_2dof_torch(alpha_f, alpha_r, self.params)

        # 2DOF 动力学方程
        m = float(self.params.m)
        Iz = float(self.params.Iz)
        beta_dot = (Fy_f + Fy_r) / (m * torch.clamp(U_next, min=1e-6)) - r
        r_dot = (float(self.params.a) * Fy_f - float(self.params.b) * Fy_r) / Iz

        # 几何部分（与 body_to_world_2dof 一致）
        x_dot = U_next * torch.cos(psi + beta)
        y_dot = U_next * torch.sin(psi + beta)
        psi_dot = r

        s_next = torch.stack([
            x + self.dt * x_dot,
            y + self.dt * y_dot,
            psi + self.dt * psi_dot,
            beta + self.dt * beta_dot,
            r + self.dt * r_dot,
            U_next,
            df_next,
            dr_next,
        ], dim=-1)

        return s_next.squeeze(0)

    # --- 成本函数：l(s, u) ---
    def _running_cost(self, s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # 取参考路径
        plan = self.plan_provider() or []
        if len(plan) < 2:
            # 无参考时仅使用动作与速度代价
            return self.w_u * torch.sum(u * u, dim=-1)

        # 支持单样本和批量
        single = False
        if s.ndim == 1:
            s = s.unsqueeze(0)
            u = u.unsqueeze(0)
            single = True

        # 提取状态
        x = s[..., 0].detach().cpu().numpy()
        y = s[..., 1].detach().cpu().numpy()
        psi = s[..., 2].detach().cpu().numpy()
        U = s[..., 5].detach().cpu().numpy()

        # 参考数组
        px = np.array([p['x'] for p in plan], dtype=float)
        py = np.array([p['y'] for p in plan], dtype=float)
        # 段航向与曲率（近似）
        dpx = np.diff(px)
        dpy = np.diff(py)
        seg_psi = np.arctan2(dpy, dpx + 1e-12)

        # 逐样本误差
        e_y_list, e_psi_list, kappa_ref_list, U_des_list = [], [], [], []
        for i in range(x.shape[0]):
            # 最近点索引（按点距离）
            di = np.argmin((px - x[i])**2 + (py - y[i])**2)
            j = int(np.clip(di, 0, len(seg_psi) - 1))
            dx = px[j+1] - px[j]
            dy = py[j+1] - py[j]
            ds = float(np.hypot(dx, dy))
            psi_base = float(seg_psi[j])
            ex = float(x[i] - px[j])
            ey = float(y[i] - py[j])
            e_y = float(-ex * np.sin(psi_base) + ey * np.cos(psi_base))
            # e_psi = 参考 - 当前
            dpsi = psi_base - float(psi[i])
            # wrap 到 [-pi, pi]
            dpsi = (dpsi + np.pi) % (2*np.pi) - np.pi
            e_psi = float(dpsi)

            # 曲率参考（两段平均）
            j_prev = max(0, j-1)
            j_next = min(len(seg_psi)-1, j+1)
            dpsi_seg = (seg_psi[j_next] - seg_psi[j_prev] + np.pi) % (2*np.pi) - np.pi
            ds_a = float(np.hypot(px[j_prev+1]-px[j_prev], py[j_prev+1]-py[j_prev])) if j_prev+1 < len(px) else ds
            ds_b = float(np.hypot(px[j_next+1]-px[j_next], py[j_next+1]-py[j_next])) if j_next+1 < len(px) else ds
            ds_avg = max(1e-6, 0.5*(ds_a+ds_b))
            kappa_ref = float(dpsi_seg / ds_avg)

            # 速度期望（横向加速度约束近似）
            ay_limit = float(getattr(self.params, 'mu', 1.0) * getattr(self.params, 'g', 9.81)) * float(getattr(self, 'ay_limit_coeff', 0.85))
            U_des = float(np.sqrt(max(0.0, ay_limit / max(1e-6, abs(kappa_ref))))) if abs(kappa_ref) > 1e-6 else float(getattr(self.params, 'U', U[i]))

            e_y_list.append(e_y)
            e_psi_list.append(e_psi)
            kappa_ref_list.append(kappa_ref)
            U_des_list.append(U_des)

        e_y_t = torch.tensor(e_y_list, dtype=s.dtype, device=s.device)
        e_psi_t = torch.tensor(e_psi_list, dtype=s.dtype, device=s.device)
        U_des_t = torch.tensor(U_des_list, dtype=s.dtype, device=s.device)

        # 曲率误差近似：由当前 r/U 近似 kappa
        kappa_cur = torch.abs(s[..., 4] / torch.clamp(s[..., 5], min=1e-6))
        kappa_ref_t = torch.tensor(np.abs(kappa_ref_list), dtype=s.dtype, device=s.device)

        # 代价项（提高曲率跟踪权重、减小横摆率惩罚）
        cost = (
            self.w_lat * (e_y_t ** 2) +
            self.w_head * (e_psi_t ** 2) +
            24.0 * ((kappa_cur - kappa_ref_t) ** 2) +
            self.w_speed * ((s[..., 5] - U_des_t) ** 2) +
            0.03 * (s[..., 4] ** 2) +
            8.0 * (s[..., 3] ** 2) +
            self.w_u * torch.sum(u * u, dim=-1)
        )

        # 前后轮相位约束：低速鼓励反相，高速弱同相
        df_t = s[..., 6]
        dr_t = s[..., 7]
        U_t = s[..., 5]
        s_lin = torch.clamp((U_t - self.U1) / (self.U2 - self.U1), 0.0, 1.0)
        k_t = self.k_low * (1.0 - s_lin) + self.k_high * s_lin
        # 低速权重大，高速权重减小
        w_phase = self.w_phase_base * (1.0 - s_lin) + (0.25 * self.w_phase_base) * s_lin
        phase_err = dr_t - k_t * df_t
        cost = cost + w_phase * (phase_err ** 2)

        # 直接惩罚同相（df*dr>0），在低速更强
        same_sign_pos = torch.clamp(df_t * dr_t, min=0.0)
        cost = cost + (self.w_phase_sign * (1.0 - s_lin)) * (same_sign_pos ** 2)

        # 理想横摆率前馈参考，鼓励 dr 接近 dr_ff
        beta_np = s[..., 3].detach().cpu().numpy()
        r_np = s[..., 4].detach().cpu().numpy()
        df_np = s[..., 6].detach().cpu().numpy()
        dr_ff_list = []
        for i in range(df_np.shape[0]):
            try:
                dr_ff, _ = ideal_yaw_rate(float(df_np[i]), np.array([beta_np[i], r_np[i]], dtype=float), self.params)
            except Exception:
                dr_ff = 0.0
            dr_ff_list.append(float(dr_ff))
        dr_ff_t = torch.tensor(dr_ff_list, dtype=s.dtype, device=s.device)
        cost = cost + self.w_yaw_ff * ((dr_t - dr_ff_t) ** 2)

        # 动作变化惩罚（使用上一时刻）
        if self._last_u is not None:
            du = u - self._last_u
            cost = cost + self.w_du * torch.sum(du * du, dim=-1)

        return cost.squeeze(0) if single else cost

    def command(self, s_np: np.ndarray) -> np.ndarray:
        s = torch.tensor(s_np, dtype=torch.float32, device=self.device)
        u = self.mppi.command(s)
        # 缓存上一动作
        self._last_u = u.detach()
        return u.detach().cpu().numpy()