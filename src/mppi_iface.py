import numpy as np
import torch
from typing import Callable, List, Dict, Optional

from .params import VehicleParams
from .twodof_torch import slip_angles_2dof_torch, lateral_forces_2dof_torch
from .pytorch_mppi.mppi import MPPI


class MPPIController4WS:
    """4WS 车辆的 MPPI 控制封装。

    - 动力学：复用 twodof 的物理公式（Torch 版），在 MPPI 内进行批量滚动。
    - 状态：s = [x, y, psi, beta, r, U]
    - 动作：u = [delta_f, delta_r, dU]
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
        num_samples: int = 1000,
        horizon: int = 14,
        lambda_: float = 180.0,
    ):
        self.params = params
        self.dt = float(dt)
        self.plan_provider = plan_provider
        self.delta_max = float(delta_max)
        self.dU_max = float(dU_max)
        self.U_max = float(U_max)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.w_lat = 100.0
        self.w_head = 600.0
        self.w_speed = 4.0
        self.w_u = 1.0
        self.w_du = 0.4

        # 控制噪声协方差（三维动作）
        noise_sigma = torch.diag(torch.tensor([
            0.08,  # delta_f
            0.08,  # delta_r
            0.5    # dU
        ], dtype=torch.float32))

        # 动作边界
        u_min = torch.tensor([
            -self.delta_max,
            -self.delta_max,
            -self.dU_max * self.dt
        ], dtype=torch.float32)
        u_max = torch.tensor([
            self.delta_max,
            self.delta_max,
            self.dU_max * self.dt
        ], dtype=torch.float32)

        self.mppi = MPPI(
            dynamics=self._dynamics,
            running_cost=self._running_cost,
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

        x, y, psi, beta, r, U = s.unbind(-1)
        df, dr, dU = u.unbind(-1)

        # 速度更新（带边界）
        U_next = torch.clamp(U + dU, 0.0, torch.tensor(self.U_max, dtype=s.dtype, device=s.device))

        # 侧偏角与横向力（复用 twodof 公式，但 U 使用当前状态值）
        alpha_f, alpha_r = slip_angles_2dof_torch(beta, r, df, dr, float(self.params.a), float(self.params.b), U_next)
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

        # 代价项
        cost = (
            self.w_lat * (e_y_t ** 2) +
            self.w_head * (e_psi_t ** 2) +
            12.0 * ((kappa_cur - kappa_ref_t) ** 2) +
            self.w_speed * ((s[..., 5] - U_des_t) ** 2) +
            0.15 * (s[..., 4] ** 2) +
            8.0 * (s[..., 3] ** 2) +
            self.w_u * torch.sum(u * u, dim=-1)
        )

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