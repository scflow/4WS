import threading
import time
from typing import List, Dict, Literal
import numpy as np
from .params import VehicleParams
from .model import SimState, Control, TrackSettings
from .twodof import derivatives as deriv_2dof
from .dof_utils import body_to_world_2dof, body_to_world_3dof, curvature_4ws
# from .mpc import solve_mpc_2dof, linearize_2dof
from .mpc import solve_mpc_kin_dyn_4dof
from .threedof import (
    Vehicle3DOF,
    State3DOF,
    allocate_drive,
    derivatives_dfdr,
)
from .planner import plan_quintic_xy
from .strategy import ideal_yaw_rate

class SimEngine:
    """后端仿真引擎：维护状态、轨迹与控制，并在后台线程中积分。"""
    def __init__(self, params: VehicleParams, dt: float = 0.02):
        self.params = params
        self.dt = float(dt)
        # 2DOF 与 3DOF 双态，按模式选择返回
        self.state2 = SimState()
        self.state3 = State3DOF(vx=params.U, vy=0.0, r=0.0, x=0.0, y=0.0, psi=0.0)
        self.ctrl = Control(U=params.U)
        self.track: List[Dict[str, float]] = []  # 每项为 {x, y, t}
        self.track_cfg = TrackSettings()

        # 模式
        self.mode: Literal['2dof', '3dof'] = '2dof'
        self._sim_t = 0.0

        self.running = False
        self._alive = True
        self._lock = threading.RLock()
        self._thread = threading.Thread(target=self._loop, name="SimLoop", daemon=True)
        self._thread.start()

        self.delta_max = np.deg2rad(30.0)  # 轮角限幅（提升可达曲率）
        self.U_switch = 8.0                # 高速同相阈值（m/s）
        self.phase_auto = False            # 关闭高速同相覆盖，恢复手动后轮转向
        self.k_v = 0.8                     # 纵向速度跟踪增益（减弱牵引对Fy挤占）
        self.tau_ctrl = 0.15               # 控制输入滤波时间常数（s）
        # 可配置的横摆阻尼与饱和控制参数（由后端 API 可更新）
        self.yaw_damp = 220.0              # 横摆阻尼力矩系数
        self.yaw_sat_gain = 3.0            # 横摆率饱和额外阻尼增益
        self._df_filt = 0.0
        self._dr_filt = 0.0
        self._U_cmd_filt = float(self.ctrl.U)

        # 显示/诊断：轮角与角速度、车速与转弯半径
        self._df_cur = 0.0
        self._dr_cur = 0.0
        self._df_dot = 0.0
        self._dr_dot = 0.0
        self._speed = float(self.params.U)
        self._radius: float | None = None

        # 牵引分配：后轴为主；按转角对齐车体轴衰减前轴扭矩（减小纵向致转力矩并保留侧向容量）
        self.drive_bias_front = 0.1        # 前轴基础牵引比例
        self.drive_bias_rear = 0.9         # 后轴基础牵引比例
        # 低速融合时间常数（s）：yaw 与侧偏的几何/动力学混合
        self.tau_low = 0.25
        self.tau_beta = 0.35
        # 规划与自动跟踪（MPC 占位）：参考轨迹与自动跟踪开关
        self.plan: List[Dict[str, float]] = []  # 每项 {t, x, y, psi}
        self.autop_enabled: bool = False
        self.autop_mode: Literal['simple', 'mpc'] = 'mpc'
        self._plan_idx: int = 0
        # 目标位姿与重规划开关
        self.goal_pose_end: Dict[str, float] | None = None
        self.replan_every_step: bool = False
        # 纯追踪/几何参考的预瞄距离参数（可前端/配置调整）
        self.Ld_k = 0.8                 # 预瞄距离线性系数：Ld = k*U + b
        self.Ld_b = 4.0                 # 预瞄距离偏置
        self.Ld_min = 3.0               # 预瞄下限
        self.Ld_max = 18.0              # 预瞄上限
        # MPC 车速控制：巡航速度、加减速限、横向加速度上限系数
        self.U_cruise = float(params.U)
        self.U_max = float(params.U)           # 速度上限：默认等于巡航速度
        self.dU_max = 1.5                      # m/s 每秒加减速上限
        self.ay_limit_coeff = 0.85             # 横向加速度占比（mu*g 的比例）

    # 线程主循环
    def _loop(self):
        last = time.perf_counter()
        while self._alive:
            start = time.perf_counter()
            if self.running:
                # 固定步长积分（避免漂移，按设定 dt 计算步数）
                with self._lock:
                    self._step(self.dt)
            # 控制节拍：尽量与 dt 接近
            spent = time.perf_counter() - start
            sleep = max(0.0, self.dt - spent)
            time.sleep(sleep)

    # 单步积分
    def _step(self, dt: float):
        # 将控制量回写到参数（速度作为 VehicleParams.U）
        self.params.U = float(self.ctrl.U)

        # 自动跟踪：在积分前根据参考轨迹更新 df/dr（支持 2DOF）
        if self.autop_enabled and len(self.plan) > 0:
            # 若启用每步重规划，则先根据当前状态与目标位姿重算参考轨迹
            if self.replan_every_step:
                try:
                    self._replan_to_goal()
                except Exception:
                    pass
            if self.autop_mode == 'mpc' and self.mode == '2dof':
                self._autop_update_mpc()
            elif self.mode == '2dof':
                self._autop_update_simple()

        # 模式分支：2DOF 线性 或 3DOF 非线性
        if self.mode == '2dof':
            # 计算 beta/r 导数（2DOF）
            x_vec = np.array([self.state2.beta, self.state2.r], dtype=float)
            d = deriv_2dof(x_vec, self.ctrl.delta_f, self.ctrl.delta_r, self.params)
            beta_dot, r_dot = float(d["xdot"][0]), float(d["xdot"][1])

            # 姿态与位置积分（2DOF）：psi_dot = r；x_dot/y_dot 使用带符号 U（允许倒车）
            U_signed = float(self.params.U)
            psi_dot = self.state2.r
            x_dot, y_dot = body_to_world_2dof(U_signed, self.state2.beta, self.state2.psi)

            # 低速融合：根据 |U| 与 U_blend 计算权重（平滑步进 smoothstep）
            U_mag = self.params.U_eff()
            U_blend = max(1e-9, float(getattr(self.params, 'U_blend', 0.3)))
            t = max(0.0, min(1.0, U_mag / U_blend))
            w = t * t * (3.0 - 2.0 * t)
            # 几何目标横摆率与几何导数（r 指令跟踪；beta 轻微阻尼）
            kappa = curvature_4ws(float(self.ctrl.delta_f), float(self.ctrl.delta_r), self.params.L)
            r_des = U_signed * kappa
            r_dot_kin = (r_des - self.state2.r) / max(1e-6, self.tau_low)
            beta_dot_kin = - self.state2.beta / max(1e-6, self.tau_beta)
            # 混合导数
            beta_dot = w * beta_dot + (1.0 - w) * beta_dot_kin
            r_dot = w * r_dot + (1.0 - w) * r_dot_kin

            self.state2.beta += beta_dot * dt
            self.state2.r += r_dot * dt
            self.state2.psi += psi_dot * dt
            self.state2.x += x_dot * dt
            self.state2.y += y_dot * dt

            # 2DOF：近似使用输入角计算角速度；速度与半径
            df_now = float(self.ctrl.delta_f)
            dr_now = float(self.ctrl.delta_r)
            self._df_dot = (df_now - self._df_cur) / dt
            self._dr_dot = (dr_now - self._dr_cur) / dt
            self._df_cur = df_now
            self._dr_cur = dr_now

            Ueff = self.params.U_eff()
            self._speed = Ueff
            self._radius = (Ueff / abs(self.state2.r)) if abs(self.state2.r) > 1e-6 else None

            if self.track_cfg.enabled:
                self._push_track_point(self.state2.x, self.state2.y)
        else:
            # 3DOF 非线性：直接使用 df/dr 控制（与前端一致）
            vp3 = Vehicle3DOF(
                m=self.params.m,
                Iz=self.params.Iz,
                a=self.params.a,
                b=self.params.b,
                g=self.params.g,
                U_min=self.params.U_min,
                kf=self.params.kf,
                kr=self.params.kr,
                tire_model=self.params.tire_model,
            )
            # 为提升曲率、减弱过强抑制，适当降低横摆阻尼与饱和增益
            vp3.yaw_damp = float(self.yaw_damp)
            vp3.yaw_sat_gain = float(self.yaw_sat_gain)
            # 将全局附着系数映射到 3DOF 轮胎参数（保持与 2DOF 一致的体验）
            try:
                mu_val = float(self.params.mu)
                vp3.tire_params_f.mu_y = mu_val
                vp3.tire_params_r.mu_y = mu_val
                vp3.tire_long_params_f.mu_x = mu_val
                vp3.tire_long_params_r.mu_x = mu_val
            except Exception:
                pass

            # 直接使用控制输入的 df/dr（前端原始角度）
            df_raw = float(self.ctrl.delta_f)
            dr_raw = float(self.ctrl.delta_r)

            # 控制角不滤波、不限幅，直接使用原始输入；速度命令可保留轻微滤波
            alpha = 1.0 - np.exp(-dt / max(1e-6, self.tau_ctrl))
            self._U_cmd_filt += alpha * (float(self.ctrl.U) - self._U_cmd_filt)
            df = float(df_raw)
            dr = float(dr_raw)

            # 纵向驱动/制动：目标速度跟踪，体现 Fx-Fy 耦合
            ax_cmd = self.k_v * (self._U_cmd_filt - self.state3.vx)
            Fx_total = vp3.m * ax_cmd
            # 牵引分配（模块化函数）：角度相关衰减 + 轴向比例
            Fx_f_pure, Fx_r_pure = allocate_drive(Fx_total, df, dr, self.drive_bias_front, self.drive_bias_rear)

            # 3DOF 导数（模块化函数）：包含轮胎侧偏、摩擦椭圆与动力学方程
            ds, aux = derivatives_dfdr(self.state3, df, dr, vp3, Fx_f_pure, Fx_r_pure)

            # 低速融合（3DOF）：将动力学导数与几何导数按速度平滑混合
            U_signed = float(self.ctrl.U)
            speed_mag = float(np.hypot(self.state3.vx, self.state3.vy))
            U_blend = max(1e-9, float(getattr(self.params, 'U_blend', 0.3)))
            t = max(0.0, min(1.0, speed_mag / U_blend))
            w = t * t * (3.0 - 2.0 * t)
            # 几何：目标横摆率、纵向速度跟踪、侧向速度阻尼与位移沿航向
            kappa = curvature_4ws(df, dr, vp3.L)
            r_des = U_signed * kappa
            r_dot_kin = (r_des - self.state3.r) / max(1e-6, self.tau_low)
            vx_dot_kin = ax_cmd  # 与上方速度跟踪一致
            vy_dot_kin = - self.state3.vy / max(1e-6, self.tau_beta)
            xdot_kin, ydot_kin = body_to_world_2dof(U_signed, 0.0, self.state3.psi)
            # 动力学导数分量
            vx_dot_dyn, vy_dot_dyn, r_dot_dyn, x_dot_dyn, y_dot_dyn, psi_dot_dyn = map(float, ds)
            # 混合
            vx_dot = w * vx_dot_dyn + (1.0 - w) * vx_dot_kin
            vy_dot = w * vy_dot_dyn + (1.0 - w) * vy_dot_kin
            r_dot  = w * r_dot_dyn  + (1.0 - w) * r_dot_kin
            x_dot  = w * x_dot_dyn  + (1.0 - w) * xdot_kin
            y_dot  = w * y_dot_dyn  + (1.0 - w) * ydot_kin
            psi_dot= self.state3.r

            # 积分（显式欧拉，保持与 3DOF 脚本一致）
            self.state3.vx += vx_dot * dt
            self.state3.vy += vy_dot * dt
            self.state3.r  += r_dot  * dt
            self.state3.x  += x_dot  * dt
            self.state3.y  += y_dot  * dt
            self.state3.psi+= psi_dot * dt

            # 3DOF：使用滤波/限幅后的轮角计算角速度；速度与半径
            self._df_dot = (df - self._df_cur) / dt
            self._dr_dot = (dr - self._dr_cur) / dt
            self._df_cur = df
            self._dr_cur = dr
            self._speed = float(np.hypot(self.state3.vx, self.state3.vy))
            self._radius = (self._speed / abs(self.state3.r)) if abs(self.state3.r) > 1e-6 else None

            # 仿真时间推进（保留用于后续可能的功能）
            self._sim_t += dt

            if self.track_cfg.enabled:
                self._push_track_point(self.state3.x, self.state3.y)

    def _push_track_point(self, x: float, y: float):
        t = time.perf_counter()
        self.track.append({"x": float(x), "y": float(y), "t": float(t)})
        # 裁剪保留时长
        keep = self.track_cfg.retention_sec
        if keep is not None and keep > 0:
            tcut = t - keep
            # 快速按时间裁剪（前部）
            i = 0
            while i < len(self.track) and self.track[i]["t"] < tcut:
                i += 1
            if i > 0:
                del self.track[:i]
        # 最大点数限制
        if len(self.track) > self.track_cfg.max_points:
            del self.track[:len(self.track) - self.track_cfg.max_points]

    # 公共方法：状态/轨迹/控制访问
    def get_state(self) -> Dict[str, float]:
        with self._lock:
            if self.mode == '2dof':
                return {
                    "x": self.state2.x,
                    "y": self.state2.y,
                    "psi": self.state2.psi,  # rad
                    "beta": self.state2.beta,
                    "r": self.state2.r,
                    "speed": self._speed,
                    "radius": self._radius if self._radius is not None else None,
                    "df": self._df_cur,
                    "dr": self._dr_cur,
                    "df_dot": self._df_dot,
                    "dr_dot": self._dr_dot,
                }
            else:
                # 将 3DOF 的 (vx, vy) 映射为 beta 显示，保持前端一致
                beta = float(np.arctan2(self.state3.vy, max(1e-6, self.state3.vx)))
                return {
                    "x": self.state3.x,
                    "y": self.state3.y,
                    "psi": self.state3.psi,
                    "beta": beta,
                    "r": self.state3.r,
                    "speed": self._speed,
                    "radius": self._radius if self._radius is not None else None,
                    "df": self._df_cur,
                    "dr": self._dr_cur,
                    "df_dot": self._df_dot,
                    "dr_dot": self._dr_dot,
                }

    def get_track(self) -> List[Dict[str, float]]:
        with self._lock:
            # 返回浅拷贝避免并发修改
            return list(self.track)

    def get_ctrl(self) -> Dict[str, float]:
        with self._lock:
            return {
                "U": self.ctrl.U,
                "df": self.ctrl.delta_f,
                "dr": self.ctrl.delta_r,
                "running": self.running,
                "mode": self.mode,
            }

    # 规划/自动跟踪接口
    def load_plan(self, points: List[Dict[str, float]]):
        """加载参考轨迹：points 为 {t, x, y, psi} 列表。"""
        with self._lock:
            self.plan = [
                {
                    't': float(p.get('t', 0.0)),
                    'x': float(p.get('x', 0.0)),
                    'y': float(p.get('y', 0.0)),
                    'psi': float(p.get('psi', 0.0)),
                }
                for p in points
            ]
            self._plan_idx = 0
            # 存储目标位姿（计划终点），用于每步重规划
            if len(self.plan) > 0:
                pend = self.plan[-1]
                self.goal_pose_end = {
                    'x': float(pend['x']),
                    'y': float(pend['y']),
                    'psi': float(pend.get('psi', 0.0)),
                }

    def set_autop(self, enabled: bool):
        with self._lock:
            self.autop_enabled = bool(enabled)
            # 移除启用即每步重规划，避免参考轨迹每步从当前点起导致误差始终为零
            # 默认保持当前重规划设置（初始为 False），如需开启由设置接口控制
            # self.replan_every_step = bool(enabled)

    def set_autop_mode(self, mode: str):
        with self._lock:
            m = str(mode or '').lower()
            if m in ('simple', 'mpc'):
                self.autop_mode = m
            else:
                # 保底：不识别即保持当前
                pass

    def _replan_to_goal(self):
        """基于当前状态与目标位姿进行短周期重规划，减少累积误差。"""
        # 若没有目标位姿，尝试从现有计划的终点推断
        if self.goal_pose_end is None:
            if len(self.plan) == 0:
                return
            pend = self.plan[-1]
            self.goal_pose_end = {
                'x': float(pend['x']),
                'y': float(pend['y']),
                'psi': float(pend.get('psi', 0.0)),
            }

        # 当前状态作为起点（使用 2DOF 状态）
        start = {
            'x': float(self.state2.x),
            'y': float(self.state2.y),
            'psi': float(self.state2.psi),
        }
        end = dict(self.goal_pose_end)

        # 按当前距离与有效速度估算规划时长 T
        dist = float(np.hypot(end['x'] - start['x'], end['y'] - start['y']))
        U_eff = float(max(0.3, abs(self.params.U_eff())))
        T = float(np.clip(dist / U_eff if U_eff > 1e-6 else 1.0, 0.5, 30.0))
        # 采样数：与步长匹配，限制范围避免过大
        N = int(np.clip(T / max(1e-6, self.dt), 60, 400))

        # 生成新计划并替换（起点即当前状态，终点为目标）
        plan = plan_quintic_xy(start, end, T, N, U_start=float(self.params.U))
        self.plan = plan
        self._plan_idx = 0

    def _wrap_angle(self, a: float) -> float:
        return float((a + np.pi) % (2.0 * np.pi) - np.pi)

    def _plan_ref_geometry(self, x: float, y: float, psi_cur: float, U: float) -> Dict[str, float]:
        """统一计算参考几何与误差（Frenet风格）。

        返回字典包含：
        - base_i: 最近点索引（推进用）
        - ref_i: 预瞄参考段索引（用于计算法线/曲率）
        - psi_ref: 参考段航向
        - e_lat: 有符号横向误差（左为正）
        - psi_err: 航向误差（参考-当前，包角到 [-pi,pi]）
        - kappa_ref: 参考段曲率估计（dpsi/ds）
        - Ld: 预瞄距离（m）
        - ds_ref: 参考段长度（m）
        """
        n = len(self.plan)
        if n < 2:
            return {
                'base_i': 0, 'ref_i': 0, 'psi_ref': psi_cur,
                'e_lat': 0.0, 'psi_err': 0.0, 'kappa_ref': 0.0,
                'Ld': 5.0, 'ds_ref': 1.0,
            }
        # 最近点索引（允许回退）：在 self._plan_idx 附近窗口内搜索最近点
        start_hint = int(self._plan_idx)
        window = 200
        i0 = max(0, start_hint - window)
        i1 = min(n - 1, start_hint + window)
        best_i = i0
        best_d2 = float('inf')
        for i in range(i0, i1 + 1):
            px = float(self.plan[i]['x'])
            py = float(self.plan[i]['y'])
            d2 = (px - x) * (px - x) + (py - y) * (py - y)
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        base_i = int(best_i)
        self._plan_idx = base_i
        base_i = max(0, min(n - 2, base_i))
        # 速度调度的预瞄距离：低速更短，高速更长（夹到范围），由前端/配置可调
        U_mag = float(max(0.0, abs(U)))
        Ld = float(np.clip(self.Ld_k * U_mag + self.Ld_b, self.Ld_min, self.Ld_max))
        # 沿轨迹累计弧长到 Ld，得到参考段索引
        s_acc = 0.0
        ref_i = base_i
        while ref_i < n - 1 and s_acc < Ld:
            p = self.plan[ref_i]
            q = self.plan[ref_i + 1]
            ds_i = float(np.hypot(q['x'] - p['x'], q['y'] - p['y']))
            s_acc += ds_i
            if s_acc < Ld:
                ref_i += 1
        ref_i = min(ref_i, n - 2)

        # 参考段航向与长度
        p0 = self.plan[ref_i]
        p1 = self.plan[ref_i + 1]
        dx = float(p1['x'] - p0['x']); dy = float(p1['y'] - p0['y'])
        ds_ref = float(np.hypot(dx, dy))
        psi_ref = float(np.arctan2(dy, dx)) if ds_ref > 1e-6 else float(p0.get('psi', psi_cur))

        # 横向误差（左法线为正）：n = (-sin psi_ref, cos psi_ref)
        ex = float(x - p0['x']); ey = float(y - p0['y'])
        e_lat = float(-ex * np.sin(psi_ref) + ey * np.cos(psi_ref))

        # 航向误差（参考-当前）
        psi_err = self._wrap_angle(psi_ref - float(psi_cur))

        # 曲率估计：dpsi/ds（使用相邻两段的差分）
        j_prev = max(0, ref_i - 1)
        j_next = min(n - 2, ref_i + 1)
        # 段航向
        def seg_psi(i: int) -> float:
            a = self.plan[i]; b = self.plan[i + 1]
            dx_i = float(b['x'] - a['x']); dy_i = float(b['y'] - a['y'])
            ds_i = float(np.hypot(dx_i, dy_i))
            return float(np.arctan2(dy_i, dx_i)) if ds_i > 1e-6 else float(a.get('psi', psi_ref))
        psi_a = seg_psi(j_prev)
        psi_b = seg_psi(j_next)
        dpsi = self._wrap_angle(psi_b - psi_a)
        ds_a = float(np.hypot(self.plan[j_prev + 1]['x'] - self.plan[j_prev]['x'], self.plan[j_prev + 1]['y'] - self.plan[j_prev]['y']))
        ds_b = float(np.hypot(self.plan[j_next + 1]['x'] - self.plan[j_next]['x'], self.plan[j_next + 1]['y'] - self.plan[j_next]['y']))
        ds_avg = max(1e-6, 0.5 * (ds_a + ds_b))
        kappa_ref = float(dpsi / ds_avg)

        return {
            'base_i': base_i,
            'ref_i': ref_i,
            'psi_ref': psi_ref,
            'e_lat': e_lat,
            'psi_err': psi_err,
            'kappa_ref': kappa_ref,
            'Ld': Ld,
            'ds_ref': ds_ref,
        }

    def _autop_update_simple(self):
        """纯追踪（Pure Pursuit）：选择预瞄点，用几何公式输出前轮角，后轮用理想横摆补偿。"""
        if not (self.autop_enabled and self.mode == '2dof' and len(self.plan) > 0):
            return
        x = float(self.state2.x)
        y = float(self.state2.y)
        psi_cur = float(self.state2.psi)
        U_signed = float(self.ctrl.U)
        U_mag = float(self.params.U_eff())
        ref = self._plan_ref_geometry(x, y, psi_cur, U_mag)
        # 选择预瞄点为 ref_i+1 的点（更靠前一点更稳定）
        i_goal = min(len(self.plan) - 1, int(ref['ref_i'] + 1))
        p_goal = self.plan[i_goal]
        # 视线角
        alpha = self._wrap_angle(float(np.arctan2(p_goal['y'] - y, p_goal['x'] - x)) - psi_cur)
        L = float(self.params.L)
        Ld = float(max(1.0, ref['Ld']))
        # 纯追踪前轮角（自行车模型公式）
        df_cmd_raw = float(np.arctan2(2.0 * L * np.sin(alpha), Ld))
        df_cmd = float(np.clip(df_cmd_raw, -self.delta_max, self.delta_max))
        # 后轮理想横摆补偿（基于几何目标曲率）
        x_vec = np.array([self.state2.beta, self.state2.r], dtype=float)
        try:
            dr_cmd_raw, _diag = ideal_yaw_rate(df_cmd, x_vec, self.params)
        except Exception:
            dr_cmd_raw = 0.0
        dr_cmd = float(np.clip(dr_cmd_raw, -self.delta_max, self.delta_max))
        # 平滑与输出
        alpha_f = 1.0 - np.exp(-self.dt / max(1e-6, self.tau_ctrl))
        self._df_filt += alpha_f * (df_cmd - self._df_filt)
        self._dr_filt += alpha_f * (dr_cmd - self._dr_filt)
        self.ctrl.delta_f = float(self._df_filt)
        self.ctrl.delta_r = float(self._dr_filt)

    def _linearize_2dof(self, x_vec: np.ndarray, df0: float, dr0: float) -> tuple[np.ndarray, np.ndarray]:
        """数值线性化 2DOF：xdot ≈ A x + B u，返回 A(2x2), B(2x2)。"""
        base = deriv_2dof(x_vec, df0, dr0, self.params)
        xdot0 = np.array(base["xdot"], dtype=float)
        nx = 2
        nu = 2
        A = np.zeros((nx, nx), dtype=float)
        B = np.zeros((nx, nu), dtype=float)
        eps_x = 1e-4
        eps_u = 1e-3
        # A: 对状态求导
        for j in range(nx):
            x_eps = np.array(x_vec, dtype=float)
            x_eps[j] += eps_x
            xdot_eps = np.array(deriv_2dof(x_eps, df0, dr0, self.params)["xdot"], dtype=float)
            A[:, j] = (xdot_eps - xdot0) / eps_x
        # B: 对控制求导
        u0 = np.array([df0, dr0], dtype=float)
        for j in range(nu):
            u_eps = np.array(u0, dtype=float)
            u_eps[j] += eps_u
            xdot_eps = np.array(deriv_2dof(x_vec, float(u_eps[0]), float(u_eps[1]), self.params)["xdot"], dtype=float)
            B[:, j] = (xdot_eps - xdot0) / eps_u
        return A, B

    # def _autop_update_mpc(self):
    #     """调用外部模块的 MPC 求解，并加卡尔曼滤波与 PID 平滑以抑制抖动。"""
    #     if not (self.autop_enabled and self.mode == '2dof' and len(self.plan) > 0):
    #         return

    #     # 原始状态与控制
    #     state_raw = {
    #         'x': float(self.state2.x),
    #         'y': float(self.state2.y),
    #         'psi': float(self.state2.psi),
    #         'beta': float(self.state2.beta),
    #         'r': float(self.state2.r),
    #     }
    #     ctrl_raw = {
    #         'U': float(self.ctrl.U),
    #         'delta_f': float(self.ctrl.delta_f),
    #         'delta_r': float(self.ctrl.delta_r),
    #     }

    #     # 使用原始状态进行 MPC 求解
    #     state_for_mpc = state_raw

    #     # 求解 MPC 首步控制
    #     df_cmd, dr_cmd = solve_mpc_2dof(
    #         state_for_mpc,
    #         ctrl_raw,
    #         self.params,
    #         self.plan,
    #         self.dt,
    #         H=12,
    #         Q_beta=0.1,
    #         Q_r=1.0,
    #         Q_psi=2.0,
    #         R_df=30.0,
    #         R_dr=30.0,
    #         R_delta_df=20.0,
    #         R_delta_dr=20.0,
    #         delta_max=self.delta_max,
    #     )
    #     self.ctrl.delta_f = float(df_cmd)
    #     self.ctrl.delta_r = float(dr_cmd)

    def _autop_update_mpc(self):
        """调用外部模块的 MPC 求解（使用 4-DOF 运动学-动力学模型）。"""
        if not (self.autop_enabled and self.mode == '2dof' and len(self.plan) > 0):
            return

        # --- 1. 获取当前状态 ---
        state_raw = {
            'x': float(self.state2.x),
            'y': float(self.state2.y),
            'psi': float(self.state2.psi),
            'beta': float(self.state2.beta),
            'r': float(self.state2.r),
        }
        ctrl_raw = {
            'U': float(self.ctrl.U),
            'delta_f': float(self.ctrl.delta_f),
            'delta_r': float(self.ctrl.delta_r),
        }
        U_mag = float(self.params.U_eff())

        # --- 2. 计算误差 (e_y, e_psi) ---
        # 调用已有的几何函数
        ref_geom = self._plan_ref_geometry(
            state_raw['x'], 
            state_raw['y'], 
            state_raw['psi'], 
            U_mag
        )
        # 使用最近点作为误差锚点，避免预瞄段掩盖真实偏差
        try:
            n_plan = len(self.plan)
            base_i = int(ref_geom.get('base_i', 0))
            i_seg = max(0, min(n_plan - 2, base_i))
            p0 = self.plan[i_seg]
            p1 = self.plan[i_seg + 1]
            dx = float(p1['x'] - p0['x']); dy = float(p1['y'] - p0['y'])
            ds = float(np.hypot(dx, dy))
            psi_base = float(np.arctan2(dy, dx)) if ds > 1e-6 else float(p0.get('psi', state_raw['psi']))
            ex = float(state_raw['x'] - p0['x']); ey = float(state_raw['y'] - p0['y'])
            # e_y: 左法线为正；e_psi: 参考 - 当前（与 MPC 模型一致）
            e_y = float(-ex * np.sin(psi_base) + ey * np.cos(psi_base))
            e_psi = float(self._wrap_angle(psi_base - float(state_raw['psi'])))
        except Exception:
            # 回退到预瞄段误差
            e_y = float(ref_geom['e_lat'])
            e_psi = float(ref_geom['psi_err'])

        # --- 3. 构建 4-DOF 增广状态 ---
        state_for_mpc = {
            'x': state_raw['x'],
            'y': state_raw['y'],
            'psi': state_raw['psi'],
            'e_y': e_y,
            'e_psi': e_psi,
            'beta': state_raw['beta'],
            'r': state_raw['r'],
        }
        
        # --- 4. 求解 MPC ---
        # 注意：这里的权重和以前完全不同！
        # Q_ey 和 Q_epsi 应该是主要驱动力
        # R 和 R_delta 应该相对较小，以允许控制器动作
        df_cmd, dr_cmd = solve_mpc_kin_dyn_4dof(
            state_for_mpc,
            ctrl_raw,
            self.params,
            self.plan,
            self.dt,
            H=20,           # 预测时域
            Q_ey=100,      # !! 高横向误差惩罚
            Q_epsi=500,     # !! 高航向误差惩罚
            Q_beta=20,     # 低侧滑惩罚
            Q_r=0.1,        # 低横摆率惩罚 (主要靠 e_psi)
            R_df=2,       # 低控制代价
            R_dr=2,       # 低控制代价
            R_delta_df=0.2, # 低控制变化率代价
            R_delta_dr=0.2,
            delta_max=self.delta_max,
        )
        
        # --- 5. 应用控制 ---
        self.ctrl.delta_f = float(df_cmd)
        self.ctrl.delta_r = float(dr_cmd)

    

    def set_ctrl(self, **kw):
        with self._lock:
            if "U" in kw and kw["U"] is not None:
                try:
                    self.ctrl.U = float(kw["U"])
                except (TypeError, ValueError):
                    pass
            if "df" in kw and kw["df"] is not None:
                try:
                    self.ctrl.delta_f = float(kw["df"])
                except (TypeError, ValueError):
                    pass
            if "dr" in kw and kw["dr"] is not None:
                try:
                    self.ctrl.delta_r = float(kw["dr"])
                except (TypeError, ValueError):
                    pass

    def set_mode(self, mode: str):
        with self._lock:
            if mode in ('2dof', '3dof'):
                if mode != self.mode:
                    self.mode = mode
                    self.track.clear()
                    self._sim_t = 0.0
        

    def set_track_settings(self, enabled: bool | None = None, retention_sec: float | None = None, max_points: int | None = None):
        with self._lock:
            if enabled is not None:
                self.track_cfg.enabled = bool(enabled)
            if retention_sec is not None:
                try:
                    self.track_cfg.retention_sec = max(0.0, float(retention_sec))
                except (TypeError, ValueError):
                    pass
            if max_points is not None:
                try:
                    self.track_cfg.max_points = max(100, int(max_points))
                except (TypeError, ValueError):
                    pass

    def set_init_pose(self, x: float = 0.0, y: float = 0.0, psi_rad: float = 0.0):
        with self._lock:
            if self.mode == '2dof':
                self.state2.x = float(x)
                self.state2.y = float(y)
                self.state2.psi = float(psi_rad)
            else:
                self.state3.x = float(x)
                self.state3.y = float(y)
                self.state3.psi = float(psi_rad)
            self.track.clear()

    # 运行控制
    def start(self):
        with self._lock:
            self.running = True

    def pause(self):
        with self._lock:
            self.running = False

    def toggle(self):
        with self._lock:
            self.running = not self.running
        return self.running

    def reset(self):
        with self._lock:
            self.state2 = SimState()
            self.state3 = State3DOF(vx=self.params.U, vy=0.0, r=0.0, x=0.0, y=0.0, psi=0.0)
            self.track.clear()
            self.running = False
            self._sim_t = 0.0

    def shutdown(self):
        self._alive = False
        try:
            self._thread.join(timeout=1.0)
        except RuntimeError:
            pass