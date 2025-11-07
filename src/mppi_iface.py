import math
from typing import Callable, List, Dict, Optional
import mlx.core as mx
from .params import VehicleParams
from .twodof_mlx import slip_angles_2dof_mlx, lateral_forces_2dof_mlx
from .threedof_mlx import derivatives_speed_cmd_mlx, Vehicle3DOF
from .mlx_mppi.mppi import MPPI, SMPPI, KMPPI, RBFKernel
from .strategy import ideal_yaw_rate
import numpy as np

class MPPIController4WS:
    """4WS 车辆的 MPPI 控制封装。

    - 动力学：复用 twodof 的物理公式（mlx 版），在 MPPI 内进行批量滚动。
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
        self.w_lat = 20000
        self.w_head = 400000
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
        sig_vals = [
            0.03,  # d_delta_f 每步角度增量
            0.03,  # d_delta_r 每步角度增量
            0.15   # dU 速度增量
        ]
        noise_sigma = mx.eye(3, dtype=mx.float32) * mx.array(sig_vals, dtype=mx.float32)

        # 动作边界（微分形式：转角增量限制为每秒上限的 dt 倍）
        # 角速度上限：delta_rate_frac * delta_max / s（可由外部配置传入）
        # 例如：delta_rate_frac=0.8 表示 1 秒可接近 0.8*delta_max 的角度变化
        delta_rate_max = float(self.delta_rate_frac) * self.delta_max  # rad/s
        d_delta_bound = delta_rate_max * self.dt
        u_min = mx.array([-d_delta_bound, -d_delta_bound, -self.dU_max * self.dt], dtype=mx.float32)
        u_max = mx.array([d_delta_bound, d_delta_bound, self.dU_max * self.dt], dtype=mx.float32)

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
            kernel=RBFKernel(sigma=2),
            num_support_pts=5,
            )

        # 缓存上一时刻动作用于平滑代价
        self._last_u = None

    # --- 动力学：f(s, u) -> s_next ---
    def _dynamics(self, s: object, u: object) -> object:
        # s: [batch?, 6]; u: [batch?, 3]
        # 支持单样本和批量。内部全部使用张量广播。
        # 统一为至少二维
        if len(getattr(s, 'shape', ())) == 1:
            s = mx.reshape(s, (1, -1))
        if len(getattr(u, 'shape', ())) == 1:
            u = mx.reshape(u, (1, -1))

        # 2DOF: s=[x, y, psi, beta, r, U, df, dr]
        # 3DOF: s=[vx, vy, r, x, y, psi, U_cmd, df, dr]
        if self.model_type == '2dof':
            if (getattr(s, 'shape', (0,))[-1]) >= 8:
                x = s[..., 0]; y = s[..., 1]; psi = s[..., 2]; beta = s[..., 3]
                r = s[..., 4]; U = s[..., 5]; df_cur = s[..., 6]; dr_cur = s[..., 7]
            else:
                x = s[..., 0]; y = s[..., 1]; psi = s[..., 2]; beta = s[..., 3]
                r = s[..., 4]; U = s[..., 5]
                df_cur = mx.zeros_like(U)
                dr_cur = mx.zeros_like(U)
        else:
            # 3DOF 状态
            # 保证至少 9 维
            if (getattr(s, 'shape', (0,))[-1]) < 9:
                pad_shape = (*getattr(s, 'shape', (0,))[0:-1], 9 - getattr(s, 'shape', (0,))[-1])
                pad = mx.zeros(pad_shape, dtype=getattr(s, 'dtype', mx.float32))
                s = mx.concatenate([s, pad], axis=-1)
            vx = s[..., 0]; vy = s[..., 1]; r = s[..., 2]
            x = s[..., 3]; y = s[..., 4]; psi = s[..., 5]
            U_cmd = s[..., 6]; df_cur = s[..., 7]; dr_cur = s[..., 8]

        # 解包动作：转角增量与速度增量
        d_df = u[..., 0]; d_dr = u[..., 1]; dU = u[..., 2]

        # 角度更新（限幅）
        df_next = mx.maximum(mx.minimum(df_cur + d_df, self.delta_max), -self.delta_max)
        dr_next = mx.maximum(mx.minimum(dr_cur + d_dr, self.delta_max), -self.delta_max)
        if self.model_type == '2dof':
            # 速度更新（带边界）
            U_next = mx.maximum(mx.minimum(U + dU, self.U_max), 0.0)
        else:
            # 3DOF 使用速度指令（用于纵向驱动/制动），仍做边界
            U_next = mx.maximum(mx.minimum(U_cmd + dU, self.U_max), 0.0)

        if self.model_type == '2dof':
            # 侧偏角与横向力（复用 twodof 公式）
            alpha_f, alpha_r = slip_angles_2dof_mlx(beta, r, df_next, dr_next, float(self.params.a), float(self.params.b), U_next)
            Fy_f, Fy_r = lateral_forces_2dof_mlx(alpha_f, alpha_r, self.params)

            # 2DOF 动力学方程
            m = float(self.params.m)
            Iz = float(self.params.Iz)
            beta_dot = (Fy_f + Fy_r) / (m * mx.maximum(U_next, 1e-6)) - r
            r_dot = (float(self.params.a) * Fy_f - float(self.params.b) * Fy_r) / Iz

            # 几何部分（与 body_to_world_2dof 一致）
            x_dot = U_next * mx.cos(psi + beta)
            y_dot = U_next * mx.sin(psi + beta)
            psi_dot = r

            # 统一形状到与 x 相同的批量形状，避免 stack 形状不一致
            # 使用 reshape 将所有元素统一到与 x 相同的批量形状，避免 broadcast_to 的形状推断问题
            tshape = getattr(x, 'shape')
            bs = lambda a: mx.reshape(a, tshape)
            elems = [
                bs(x + self.dt * x_dot),
                bs(y + self.dt * y_dot),
                bs(psi + self.dt * psi_dot),
                bs(beta + self.dt * beta_dot),
                bs(r + self.dt * r_dot),
                bs(U_next),
                bs(df_next),
                bs(dr_next),
            ]
            s_next = mx.stack(elems, axis=-1)
        else:
            # 3DOF 非线性动力学：使用速度指令（U_next）产生纵向驱动，耦合摩擦椭圆
            # 状态子向量为 [vx, vy, r, x, y, psi]
            s6 = mx.stack([vx, vy, r, x, y, psi], axis=-1)
            ds, _aux = derivatives_speed_cmd_mlx(
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
            vx_dot = ds[..., 0]; vy_dot = ds[..., 1]; r_dot = ds[..., 2]
            x_dot = ds[..., 3]; y_dot = ds[..., 4]; psi_dot = ds[..., 5]
            # 统一形状到与 x 相同的批量形状，避免 stack 形状不一致
            tshape = getattr(x, 'shape')
            bs = lambda a: mx.reshape(a, tshape)
            s_next = mx.stack([
                # 状态排布保持 3DOF 版本的 9 维
                bs(vx + self.dt * vx_dot),
                bs(vy + self.dt * vy_dot),
                bs(r + self.dt * r_dot),
                bs(x + self.dt * x_dot),
                bs(y + self.dt * y_dot),
                bs(psi + self.dt * psi_dot),
                bs(U_next),
                bs(df_next),
                bs(dr_next),
            ], axis=-1)

        # 保持与输入维度一致
        return mx.squeeze(s_next, axis=0) if len(getattr(s_next, 'shape', ())) > 1 and s_next.shape[0] == 1 else s_next

    # --- 成本函数：l(s, u) ---
    def _running_cost(self, s: object, u: object) -> object:
        # 取参考路径
        plan = self.plan_provider() or []
        if len(plan) < 2:
            return self.w_u * mx.sum(u * u, axis=-1)

        # 输入形状：允许 (B, nx) 或 (M, K, nx)
        s_shape = getattr(s, 'shape', ())
        u_shape = getattr(u, 'shape', ())
        single = False
        if len(s_shape) == 1:
            s = mx.reshape(s, (1, -1))
            u = mx.reshape(u, (1, -1))
            single = True
            s_shape = getattr(s, 'shape', ())
            u_shape = getattr(u, 'shape', ())

        # 扁平化批量维度到一维，便于逐样本计算参考对齐
        batch_dims = s_shape[:-1] if len(s_shape) >= 1 else ()
        nx = s_shape[-1] if len(s_shape) >= 1 else 0
        B_total = 1
        for d in batch_dims:
            B_total *= int(d)
        s_flat = mx.reshape(s, (B_total, nx))
        u_flat = mx.reshape(u, (B_total, u_shape[-1] if len(u_shape) >= 1 else 0))

        # 提取状态：按模型类型区分（扁平批量）
        if self.model_type == '2dof':
            x = s_flat[..., 0]
            y = s_flat[..., 1]
            psi = s_flat[..., 2]
            U_mag_vec = s_flat[..., 5]
        else:
            vx_vec = s_flat[..., 0]
            vy_vec = s_flat[..., 1]
            x = s_flat[..., 3]
            y = s_flat[..., 4]
            psi = s_flat[..., 5]
            U_mag_vec = mx.sqrt(mx.maximum(vx_vec * vx_vec + vy_vec * vy_vec, 1e-12))

        # 计划几何缓存与向量化误差/曲率计算
        # 缓存版本：使用长度与首尾坐标的简化签名
        try:
            first = plan[0]
            last = plan[-1]
            plan_version = (len(plan), float(first.get('x', 0.0)), float(first.get('y', 0.0)), float(last.get('x', 0.0)), float(last.get('y', 0.0)))
        except Exception:
            plan_version = (len(plan),)
        if not hasattr(self, '_plan_cache'):
            self._plan_cache = None
            self._plan_cache_version = None
        if (self._plan_cache is None) or (self._plan_cache_version != plan_version):
            px_np = np.array([float(p['x']) for p in plan], dtype=np.float32)
            py_np = np.array([float(p['y']) for p in plan], dtype=np.float32)
            # 段几何
            dx_np = px_np[1:] - px_np[:-1]
            dy_np = py_np[1:] - py_np[:-1]
            seg_psi_np = np.arctan2(dy_np, dx_np + 1e-12).astype(np.float32)
            seg_ds_np = np.hypot(dx_np, dy_np).astype(np.float32)
            self._plan_cache = {
                'px': mx.array(px_np),
                'py': mx.array(py_np),
                'seg_psi': mx.array(seg_psi_np),
                'seg_ds': mx.array(seg_ds_np),
                'N': int(px_np.shape[0]),
                'S': int(seg_psi_np.shape[0]),
            }
            self._plan_cache_version = plan_version

        pc = self._plan_cache
        px_t, py_t = pc['px'], pc['py']
        seg_psi_t, seg_ds_t = pc['seg_psi'], pc['seg_ds']
        N, S = pc['N'], pc['S']

        # 最近点索引：对所有样本广播到所有节点
        dx_all = x[:, None] - px_t[None, :]
        dy_all = y[:, None] - py_t[None, :]
        dist2 = dx_all * dx_all + dy_all * dy_all
        j_node = mx.argmin(dist2, axis=1)
        # 段索引（最后一个节点对应上一段）
        j_seg = mx.minimum(j_node, max(0, S - 1))
        # 段基础航向
        psi_base = mx.take(seg_psi_t, j_seg)
        # 侧向与航向误差
        px_j = mx.take(px_t, j_node)
        py_j = mx.take(py_t, j_node)
        ex = x - px_j
        ey = y - py_j
        e_y_t = -(ex * mx.sin(psi_base)) + (ey * mx.cos(psi_base))
        # 角度归一化到 [-pi, pi)
        PI = 3.141592653589793
        def wrap_angle(a):
            return a - (2.0 * PI) * mx.floor((a + PI) / (2.0 * PI))
        e_psi_t = wrap_angle(psi_base - psi)

        # 曲率参考（两段平均）
        j_prev = mx.maximum(j_seg - 1, 0)
        j_next = mx.minimum(j_seg + 1, max(0, S - 1))
        psi_prev = mx.take(seg_psi_t, j_prev)
        psi_next = mx.take(seg_psi_t, j_next)
        dpsi_seg = wrap_angle(psi_next - psi_prev)
        ds_a = mx.take(seg_ds_t, j_prev)
        ds_b = mx.take(seg_ds_t, j_next)
        ds_avg = mx.maximum(1e-6, 0.5 * (ds_a + ds_b))
        kappa_ref_t_signed = dpsi_seg / ds_avg
        # U_des（未直接使用，保持与原逻辑一致）
        ay_limit = float(getattr(self.params, 'mu', 1.0) * getattr(self.params, 'g', 9.81)) * float(getattr(self, 'ay_limit_coeff', 0.85))
        kappa_mag = mx.abs(kappa_ref_t_signed)
        U_cur_vec = U_mag_vec
        U_des_t = mx.where(kappa_mag > 1e-6, mx.sqrt(mx.maximum(0.0, ay_limit / mx.maximum(1e-6, kappa_mag))), U_cur_vec)

        # 曲率与门控（使用有符号曲率以约束方向一致性）
        if self.model_type == '2dof':
            kappa_cur_signed = s_flat[..., 4] / mx.maximum(s_flat[..., 5], 1e-6)
        else:
            kappa_cur_signed = s_flat[..., 2] / mx.maximum(U_mag_vec, 1e-6)
        # kappa_ref 已向量化
        kappa_ref_mag = mx.abs(kappa_ref_t_signed)
        G_turn = mx.maximum(mx.minimum((kappa_ref_mag - self.kappa_turn_k0) / (self.kappa_turn_k1 - self.kappa_turn_k0 + 1e-9), 1.0), 0.0)
        G_straight = 1.0 - 0.7 * G_turn

        # 基础几何跟踪
        cost = (
            self.w_lat * (e_y_t ** 2) +
            self.w_head * (e_psi_t ** 2) +
            self.w_u * mx.sum(u_flat * u_flat, axis=-1)
        )

        # Yaw 跟踪
        if self.model_type == '2dof':
            yaw_track_err = s_flat[..., 4] - s_flat[..., 5] * kappa_ref_t_signed
        else:
            yaw_track_err = s_flat[..., 2] - U_mag_vec * kappa_ref_t_signed
        cost = cost + self.w_yaw_track * (yaw_track_err ** 2)

        # 直线速度激励
        if self.model_type == '2dof':
            speed_shortfall = mx.maximum(self.U_max - s_flat[..., 5], 0.0)
        else:
            speed_shortfall = mx.maximum(self.U_max - U_mag_vec, 0.0)
        cost = cost + self.w_speed * speed_shortfall * G_straight

        # 稳定性与蟹行抑制
        if self.model_type == '2dof':
            df_t = s_flat[..., 6]
            dr_t = s_flat[..., 7]
            ay_approx = (s_flat[..., 5] ** 2) * mx.abs(kappa_cur_signed)
            beta_t = s_flat[..., 3]
        else:
            df_t = s_flat[..., 7]
            dr_t = s_flat[..., 8]
            ay_approx = (U_mag_vec ** 2) * mx.abs(kappa_cur_signed)
            beta_t = mx.arctan2(s_flat[..., 1], mx.maximum(s_flat[..., 0], 1e-6))
        cost = cost + self.w_ay * (ay_approx ** 2) * G_turn
        cost = cost + self.w_beta * (beta_t ** 2) * (0.5 + 0.5 * G_turn)
        cost = cost + self.w_crab * ((df_t + dr_t) ** 2) * G_turn

        # 相位约束
        if self.model_type == '2dof':
            U_t = s_flat[..., 5]
        else:
            U_t = U_mag_vec
        s_lin = mx.maximum(mx.minimum((U_t - self.U1) / (self.U2 - self.U1), 1.0), 0.0)
        k_t = (-self.k_max * G_turn) + (self.k_high * (1.0 - G_turn) * s_lin)
        w_phase = self.w_phase_base * (1.0 + 2.0 * G_turn)
        phase_err = dr_t - k_t * df_t
        cost = cost + w_phase * (phase_err ** 2)

        # 同相惩罚
        same_sign_pos = mx.maximum(df_t * dr_t, 0.0)
        cost = cost + (self.w_phase_sign * G_turn) * (same_sign_pos ** 2)

        # 理想横摆率前馈
        if self.model_type == '2dof':
            beta_vec = s_flat[..., 3]
            r_vec = s_flat[..., 4]
            df_vec = s_flat[..., 6]
        else:
            beta_vec = mx.arctan2(s_flat[..., 1], mx.maximum(s_flat[..., 0], 1e-6))
            r_vec = s_flat[..., 2]
            df_vec = s_flat[..., 7]
        # 理想横摆率前馈（几何向量化：r_des = U * kappa_ref）
        dr_ff_t = U_t * kappa_ref_t_signed
        cost = cost + self.w_yaw_ff * ((dr_t - dr_ff_t) ** 2)

        # 弯道加速变化惩罚
        dU_t = u_flat[..., 2]
        cost = cost + self.w_dU_turn * (dU_t ** 2) * G_turn

        # 动作变化惩罚
        if self._last_u is not None:
            # 直接广播上一动作到批量维度
            du = u_flat - self._last_u
            cost = cost + self.w_du * mx.sum(du * du, axis=-1)

        # 还原到原始批量形状
        if single:
            return mx.squeeze(cost, axis=0)
        return mx.reshape(cost, batch_dims)

    def command(self, s_arr: object) -> object:
        s = mx.array(s_arr, dtype=mx.float32)
        u = self.mppi.command(s)
        # 缓存上一动作（保持数组即可）
        self._last_u = u
        return u