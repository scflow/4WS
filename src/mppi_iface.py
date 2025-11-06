import numpy as np
import torch
from typing import Callable, List, Dict, Optional

from .params import VehicleParams
from .twodof_torch import slip_angles_2dof_torch, lateral_forces_2dof_torch
from .threedof_torch import (
    slip_angles_3dof_torch,
    tire_forces_3dof_torch,
    derivatives_dfdr_torch,
    derivatives_speed_cmd_torch,
)
from .threedof import Vehicle3DOF
from .pytorch_mppi.mppi import MPPI, SMPPI, KMPPI, RBFKernel
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
        model_type: str = '2dof',
        device: Optional[str] = None,
        num_samples: int = 16,
        horizon: int = 30,
        lambda_: float = 30.0,
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
        self.model_type = str(model_type or '2dof').lower()
        # prefer MPS on macOS if available, else CUDA, else CPU
        try:
            mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        except Exception:
            mps_available = False
        self.device = device or ('mps' if mps_available else ('cuda' if torch.cuda.is_available() else 'cpu'))

        self.w_lat = 2000.0
        self.w_head = 40000
        # 直线速度激励（机会成本），弯道弱化
        self.w_speed = 10.0
        self.w_u = 1.0
        self.w_du = 3.0
        # 速度相关的相位约束（低速更强反相，高速弱同相）
        self.w_phase_base = 300.0
        self.k_low = -0.30
        self.k_high = 0.10
        self.U1 = 5.0
        self.U2 = 20.0
        # 额外约束：直接惩罚同相（df、dr 同号）与加入理想横摆率前馈
        self.w_phase_sign = 180.0
        self.w_yaw_ff = 60.0

        # 曲率门控与蟹行/稳定性权重
        self.kappa_turn_k0 = 0.02   # 转弯门控起始阈值 [1/m]
        self.kappa_turn_k1 = 0.06   # 转弯门控满载阈值 [1/m]
        self.w_yaw_track = 2400.0   # 强制 r ≈ U*kappa_ref
        self.w_crab = 220.0         # 弯道蟹行抑制 (df+dr)^2
        self.w_ay = 120.0           # 弯道横向加速度代价 ay^2
        self.w_beta = 10.0          # 弯道侧偏角代价 beta^2
        self.w_dU_turn = 1.2        # 弯道加速变化惩罚 dU^2
        self.k_max = 0.8            # 弯道相位比例的最大反相幅度

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

        # self.mppi = MPPI(
        #     dynamics=self._dynamics,
        #     running_cost=self._running_cost,
        #     nx=8,  # 状态维度: [x, y, psi, beta, r, U, delta_f, delta_r]
        #     noise_sigma=noise_sigma,
        #     num_samples=num_samples,
        #     horizon=horizon,
        #     lambda_=lambda_,
        #     u_min=u_min,
        #     u_max=u_max,
        #     device=self.device,
        # )

        # self.mppi = SMPPI(
        #     dynamics=self._dynamics,
        #     running_cost=self._running_cost,
        #     nx=8,  # 状态维度: [x, y, psi, beta, r, U, delta_f, delta_r]
        #     noise_sigma=noise_sigma,
        #     num_samples=num_samples,
        #     horizon=horizon,
        #     lambda_=lambda_,
        #     u_min=u_min,
        #     u_max=u_max,
        #     device=self.device,
        #     w_action_seq_cost=10,
        #     action_max=torch.tensor([1., 1.], dtype=torch.float32, device=self.device),
        # )
        # 状态维度：2DOF 为 8；3DOF 为 9
        nx = 8 if self.model_type == '2dof' else 9
        self.mppi = KMPPI(
            dynamics=self._dynamics,
            running_cost=self._running_cost,
            nx=nx,
            noise_sigma=noise_sigma,
            num_samples=num_samples,
            horizon=horizon,
            lambda_=lambda_,
            u_min=u_min,
            u_max=u_max,
            device=self.device,
            kernel=RBFKernel(sigma=2),
            num_support_pts=5,
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

        # 2DOF: s=[x, y, psi, beta, r, U, df, dr]
        # 3DOF: s=[vx, vy, r, x, y, psi, U_cmd, df, dr]
        if self.model_type == '2dof':
            if s.shape[-1] >= 8:
                x, y, psi, beta, r, U, df_cur, dr_cur = s.unbind(-1)
            else:
                x, y, psi, beta, r, U = s.unbind(-1)
                df_cur = torch.zeros_like(U)
                dr_cur = torch.zeros_like(U)
        else:
            # 3DOF 状态
            # 保证至少 9 维
            if s.shape[-1] < 9:
                # 若调用方传入不足 9 维，则补零（容错）
                pad = torch.zeros((*s.shape[:-1], 9 - s.shape[-1]), dtype=s.dtype, device=s.device)
                s = torch.cat([s, pad], dim=-1)
            vx, vy, r, x, y, psi, U_cmd, df_cur, dr_cur = s.unbind(-1)

        # 解包动作：转角增量与速度增量
        d_df, d_dr, dU = u.unbind(-1)

        # 角度更新（限幅）
        df_next = torch.clamp(df_cur + d_df, -self.delta_max, self.delta_max)
        dr_next = torch.clamp(dr_cur + d_dr, -self.delta_max, self.delta_max)
        if self.model_type == '2dof':
            # 速度更新（带边界）
            U_next = torch.clamp(U + dU, 0.0, torch.tensor(self.U_max, dtype=s.dtype, device=s.device))
        else:
            # 3DOF 使用速度指令（用于纵向驱动/制动），仍做边界
            U_next = torch.clamp(U_cmd + dU, 0.0, torch.tensor(self.U_max, dtype=s.dtype, device=s.device))

        if self.model_type == '2dof':
            # 侧偏角与横向力（复用 twodof 公式）
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
        else:
            # 3DOF 非线性动力学：使用速度指令（U_next）产生纵向驱动，耦合摩擦椭圆
            # 状态子向量为 [vx, vy, r, x, y, psi]
            s6 = torch.stack([vx, vy, r, x, y, psi], dim=-1)
            ds, _aux = derivatives_speed_cmd_torch(
                s6, df_next, dr_next, Vehicle3DOF(
                    m=float(self.params.m),
                    Iz=float(self.params.Iz),
                    a=float(self.params.a),
                    b=float(self.params.b),
                    g=float(self.params.g),
                    U_min=float(getattr(self.params, 'U_min', 0.5)),
                    kf=float(self.params.kf),
                    kr=float(self.params.kr),
                    tire_model=str(getattr(self.params, 'tire_model', 'linear')),
                ),
                U_next,
                k_v=float(getattr(self, 'k_v', 1.0)),
                front_bias=float(getattr(self, 'drive_bias_front', 0.5)),
                rear_bias=float(getattr(self, 'drive_bias_rear', 0.5)),
            )
            vx_dot, vy_dot, r_dot, x_dot, y_dot, psi_dot = ds.unbind(-1)
            s_next = torch.stack([
                # 状态排布保持 3DOF 版本的 9 维
                vx + self.dt * vx_dot,
                vy + self.dt * vy_dot,
                r + self.dt * r_dot,
                x + self.dt * x_dot,
                y + self.dt * y_dot,
                psi + self.dt * psi_dot,
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

        # 提取状态：按模型类型区分
        if self.model_type == '2dof':
            x = s[..., 0].detach().cpu().numpy()
            y = s[..., 1].detach().cpu().numpy()
            psi = s[..., 2].detach().cpu().numpy()
            U_mag_np = s[..., 5].detach().cpu().numpy()
        else:
            # 3DOF：速度幅值由 vx,vy 计算
            vx_np = s[..., 0].detach().cpu().numpy()
            vy_np = s[..., 1].detach().cpu().numpy()
            x = s[..., 3].detach().cpu().numpy()
            y = s[..., 4].detach().cpu().numpy()
            psi = s[..., 5].detach().cpu().numpy()
            U_mag_np = (vx_np**2 + vy_np**2)**0.5

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

        # 曲率与门控（使用有符号曲率以约束方向一致性）
        if self.model_type == '2dof':
            kappa_cur_signed = s[..., 4] / torch.clamp(s[..., 5], min=1e-6)
        else:
            U_mag_t = torch.sqrt(torch.clamp(s[..., 0] * s[..., 0] + s[..., 1] * s[..., 1], min=1e-12))
            kappa_cur_signed = s[..., 2] / torch.clamp(U_mag_t, min=1e-6)
        kappa_ref_t_signed = torch.tensor(kappa_ref_list, dtype=s.dtype, device=s.device)
        kappa_ref_mag = torch.abs(kappa_ref_t_signed)
        # 曲率门控：在 |kappa_ref| 较大时增强转弯相关惩罚
        G_turn = torch.clamp((kappa_ref_mag - self.kappa_turn_k0) / (self.kappa_turn_k1 - self.kappa_turn_k0 + 1e-9), 0.0, 1.0)
        G_straight = 1.0 - 0.7 * G_turn

        # 基础几何跟踪
        cost = (
            self.w_lat * (e_y_t ** 2) +
            self.w_head * (e_psi_t ** 2) +
            self.w_u * torch.sum(u * u, dim=-1)
        )

        # Yaw 跟踪：强制 r ≈ U * kappa_ref（方向与幅值一致）
        if self.model_type == '2dof':
            yaw_track_err = s[..., 4] - s[..., 5] * kappa_ref_t_signed
        else:
            U_mag_t = torch.sqrt(torch.clamp(s[..., 0] * s[..., 0] + s[..., 1] * s[..., 1], min=1e-12))
            yaw_track_err = s[..., 2] - U_mag_t * kappa_ref_t_signed
        cost = cost + self.w_yaw_track * (yaw_track_err ** 2)

        # 直线速度激励（机会成本），在转弯段弱化
        if self.model_type == '2dof':
            speed_shortfall = torch.clamp(torch.tensor(self.U_max, dtype=s.dtype, device=s.device) - s[..., 5], min=0.0)
        else:
            U_mag_t = torch.sqrt(torch.clamp(s[..., 0] * s[..., 0] + s[..., 1] * s[..., 1], min=1e-12))
            speed_shortfall = torch.clamp(torch.tensor(self.U_max, dtype=s.dtype, device=s.device) - U_mag_t, min=0.0)
        cost = cost + self.w_speed * speed_shortfall * G_straight

        # 稳定性与蟹行抑制（随转弯门控增强）
        if self.model_type == '2dof':
            df_t = s[..., 6]
            dr_t = s[..., 7]
            ay_approx = (s[..., 5] ** 2) * torch.abs(kappa_cur_signed)
            beta_t = s[..., 3]
        else:
            df_t = s[..., 7]
            dr_t = s[..., 8]
            # U_mag 近似横向加速度：U^2 * |kappa|
            U_mag_t = torch.sqrt(torch.clamp(s[..., 0] * s[..., 0] + s[..., 1] * s[..., 1], min=1e-12))
            ay_approx = (U_mag_t ** 2) * torch.abs(kappa_cur_signed)
            beta_t = torch.atan2(s[..., 1], torch.clamp(s[..., 0], min=1e-6))
        cost = cost + self.w_ay * (ay_approx ** 2) * G_turn
        cost = cost + self.w_beta * (beta_t ** 2) * (0.5 + 0.5 * G_turn)
        # 直接抑制同向同角（df+dr）
        cost = cost + self.w_crab * ((df_t + dr_t) ** 2) * G_turn

        # 前后轮相位约束：低速鼓励反相，高速弱同相
        U_t = s[..., 5] if self.model_type == '2dof' else torch.sqrt(torch.clamp(s[..., 0] * s[..., 0] + s[..., 1] * s[..., 1], min=1e-12))
        s_lin = torch.clamp((U_t - self.U1) / (self.U2 - self.U1), 0.0, 1.0)
        # 相位比例：在大曲率时强反相；在小曲率且高速时允许微弱同相
        k_t = (-self.k_max * G_turn) + (self.k_high * (1.0 - G_turn) * s_lin)
        # 弯道提升相位权重
        w_phase = self.w_phase_base * (1.0 + 2.0 * G_turn)
        phase_err = dr_t - k_t * df_t
        cost = cost + w_phase * (phase_err ** 2)

        # 直接惩罚同相（df*dr>0），随转弯增强
        same_sign_pos = torch.clamp(df_t * dr_t, min=0.0)
        cost = cost + (self.w_phase_sign * G_turn) * (same_sign_pos ** 2)

        # 理想横摆率前馈参考，鼓励 dr 接近 dr_ff
        if self.model_type == '2dof':
            beta_np = s[..., 3].detach().cpu().numpy()
            r_np = s[..., 4].detach().cpu().numpy()
            df_np = s[..., 6].detach().cpu().numpy()
        else:
            vx_np = s[..., 0].detach().cpu().numpy()
            vy_np = s[..., 1].detach().cpu().numpy()
            beta_np = np.arctan2(vy_np, np.clip(vx_np, 1e-6, None))
            r_np = s[..., 2].detach().cpu().numpy()
            df_np = s[..., 7].detach().cpu().numpy()
        dr_ff_list = []
        for i in range(df_np.shape[0]):
            try:
                dr_ff, _ = ideal_yaw_rate(float(df_np[i]), np.array([beta_np[i], r_np[i]], dtype=float), self.params)
            except Exception:
                dr_ff = 0.0
            dr_ff_list.append(float(dr_ff))
        dr_ff_t = torch.tensor(dr_ff_list, dtype=s.dtype, device=s.device)
        cost = cost + self.w_yaw_ff * ((dr_t - dr_ff_t) ** 2)

        # 弯道加速变化惩罚：抑制弯中激进加速或减速
        dU_t = u[..., 2]
        cost = cost + self.w_dU_turn * (dU_t ** 2) * G_turn

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