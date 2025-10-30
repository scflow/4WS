import threading
import time
from typing import List, Dict, Literal

import numpy as np

from .params import VehicleParams
from .model import SimState, Control, TrackSettings
from .twodof import derivatives as deriv_2dof
from .dof_utils import body_to_world_2dof, body_to_world_3dof, curvature_4ws
from .threedof import (
    Vehicle3DOF,
    State3DOF,
    allocate_drive,
    derivatives_dfdr,
)

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

        # 3DOF 控制与稳定性改进：限幅、相位切换、速度跟踪与滤波
        self.delta_max = np.deg2rad(40.0)  # 轮角限幅（约 20°，提升可达曲率）
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

    def set_ctrl(self, **kw):
        with self._lock:
            if "U" in kw and kw["U"] is not None:
                try:
                    # 3DOF：将 U 作为目标速度，vx 不再瞬时设置
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
                    # 切换模式时复位轨迹与仿真时间，保持干净状态
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
            # 清空轨迹以避免旧点影响
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