from __future__ import annotations
from typing import Tuple, Dict, Optional
import mlx.core as mx
from dataclasses import dataclass, field
from .tire_mlx import (
    lateral_force_dispatch_mlx,
    pacejka_longitudinal_mlx,
    combine_friction_ellipse_mlx,
    PacejkaParams, 
    PacejkaLongParams,
)


def static_loads_3dof_mlx(
    a: float,
    b: float,
    m: float,
    g: float,
    dtype: object | None = None,
) -> Tuple[object, object]:
    """前后轴静态法向载荷（忽略载荷转移，MLX-only）。"""
    L = a + b
    if abs(L) < 1e-9:
        L = 1e-9
    Fzf = m * g * (b / L)
    Fzr = m * g * (a / L)
    dt = dtype if dtype is not None else mx.float32
    return (mx.array(Fzf, dtype=dt), mx.array(Fzr, dtype=dt))


def slip_angles_3dof_mlx(
    vx: object,
    vy: object,
    r: object,
    df: object,
    dr: object,
    a: float,
    b: float,
    U_min: float,
) -> Tuple[object, object]:
    """3DOF 自行车模型侧偏角（MLX-only）。

    alpha_f = atan2(vy + a*r, vx_eff) - df
    alpha_r = atan2(vy - b*r, vx_eff) - dr
    近零保护：使用 vx_eff = sign(vx) * max(|vx|, U_min) 避免低速时 |alpha| 过大。
    """
    U_min_t = mx.array(float(U_min), dtype=getattr(vx, 'dtype', mx.float32))
    vx_mag = mx.abs(vx)
    vx_eff = mx.where(vx_mag < U_min_t, mx.sign(vx) * U_min_t, vx)
    alpha_f = mx.arctan2(vy + (a * r), vx_eff) - df
    alpha_r = mx.arctan2(vy - (b * r), vx_eff) - dr
    return alpha_f, alpha_r


def yaw_rate_limit_mlx(mu_y: float, g: float, vx_eff: object) -> object:
    """附着-受限的横摆率边界 |r| ≤ mu_y * g / vx_eff（MLX-only）。"""
    vx_mag = mx.abs(vx_eff)
    tiny = mx.array(1e-9, dtype=getattr(vx_mag, 'dtype', mx.float32))
    return (float(mu_y) * float(g)) / mx.maximum(vx_mag, tiny)


def apply_yaw_saturation_mlx(
    r: object,
    r_dot: object,
    mu_y: float,
    g: float,
    vx_eff: object,
    gain: float,
) -> object:
    """当 |r| 超出边界时施加附加阻尼（MLX-only）。返回调整后的 r_dot。"""
    r_max = yaw_rate_limit_mlx(mu_y, g, vx_eff)
    exceed = mx.abs(r) > r_max
    correction = float(gain) * (mx.abs(r) - r_max) * mx.sign(r)
    r_dot_adj = mx.where(exceed, r_dot - correction, r_dot)
    return r_dot_adj


def allocate_drive_mlx(
    Fx_total: object,
    df: object,
    dr: object,
    front_bias: float,
    rear_bias: float,
) -> Tuple[object, object]:
    """角度感知的纵向力分配（MLX-only）。

    使用 cos^2 衰减与轴向比例权重，将 Fx_total 分配到前/后轴。
    """
    cosdf2 = mx.cos(df) ** 2
    cosdr2 = mx.cos(dr) ** 2
    front_share = float(front_bias) * cosdf2
    rear_share = float(rear_bias) * cosdr2
    share_sum = front_share + rear_share
    tiny = mx.array(1e-9, dtype=getattr(Fx_total, 'dtype', mx.float32))
    Fx_f_pure = Fx_total * front_share / mx.maximum(share_sum, tiny)
    Fx_r_pure = Fx_total * rear_share / mx.maximum(share_sum, tiny)
    return Fx_f_pure, Fx_r_pure


def tire_forces_3dof_mlx(
    alpha_f: object,
    alpha_r: object,
    vp: Vehicle3DOF,
    dtype: object | None = None,
) -> Tuple[object, object]:
    """根据选择的轮胎模型计算前/后轮横向力。"""
    Fzf, Fzr = static_loads_3dof_mlx(vp.a, vp.b, vp.m, vp.g, dtype=dtype)
    model_sel = (vp.tire_model or 'linear').lower()
    Fy_f = lateral_force_dispatch_mlx(alpha_f, Fzf, model_sel, vp.kf, vp.tire_params_f)
    Fy_r = lateral_force_dispatch_mlx(alpha_r, Fzr, model_sel, vp.kr, vp.tire_params_r)
    return Fy_f, Fy_r


def derivatives_dfdr_mlx(
    s: object,
    df: object,
    dr: object,
    vp: Vehicle3DOF,
    Fx_f_pure: Optional[object] = None,
    Fx_r_pure: Optional[object] = None,
) -> Tuple[object, Dict[str, object]]:
    """3DOF 非线性动力学的 MLX-only 版导数计算。

    输入
    - s: [..., 6] 张量，内容为 [vx, vy, r, x, y, psi]
    - df, dr: 轮角（广播兼容）
    - vp: 车辆参数（dataclass，标量字段）
    - Fx_f_pure, Fx_r_pure: 纯纵向力（可选；若未给出则视为 0）

    输出
    - ds: [..., 6] 张量导数
    - aux: 字典，包含中间量（alpha_f/r、Fy/Fx 等）便于诊断
    """
    if getattr(s, 'ndim', None) == 1:
        s = s[0:1, ...]
    vx = s[..., 0]
    vy = s[..., 1]
    r = s[..., 2]
    x = s[..., 3]
    y = s[..., 4]
    psi = s[..., 5]

    # 侧偏角与横向力
    alpha_f, alpha_r = slip_angles_3dof_mlx(vx, vy, r, df, dr, float(vp.a), float(vp.b), float(vp.U_min))
    Fy_f_pure, Fy_r_pure = tire_forces_3dof_mlx(alpha_f, alpha_r, vp, dtype=getattr(s, 'dtype', None))

    # 静载荷
    Fzf, Fzr = static_loads_3dof_mlx(vp.a, vp.b, vp.m, vp.g, dtype=getattr(s, 'dtype', None))

    # 纵向纯力：若未提供则置零
    zero = mx.array(0.0, dtype=getattr(s, 'dtype', mx.float32))
    Fx_f_pure_t = Fx_f_pure if Fx_f_pure is not None else zero
    Fx_r_pure_t = Fx_r_pure if Fx_r_pure is not None else zero

    # 摩擦椭圆合成
    Fx_f, Fy_f = combine_friction_ellipse_mlx(
        Fx_f_pure_t, Fy_f_pure, Fzf, float(vp.tire_long_params_f.mu_x), float(vp.tire_params_f.mu_y)
    )
    Fx_r, Fy_r = combine_friction_ellipse_mlx(
        Fx_r_pure_t, Fy_r_pure, Fzr, float(vp.tire_long_params_r.mu_x), float(vp.tire_params_r.mu_y)
    )

    # 动力学：vx_dot, vy_dot
    inv_m = 1.0 / float(vp.m)
    vx_dot = inv_m * (
        Fx_f * mx.cos(df) - Fy_f * mx.sin(df) + Fx_r * mx.cos(dr) - Fy_r * mx.sin(dr)
        - float(vp.Fwx) - float(vp.Fsx)
    ) + r * vy
    vy_dot = inv_m * (
        Fx_f * mx.sin(df) + Fy_f * mx.cos(df) + Fx_r * mx.sin(dr) + Fy_r * mx.cos(dr)
        + float(vp.Fwy) + float(vp.Fsy)
    ) - r * vx

    # 横摆力矩与 r_dot（含阻尼与饱和）
    Mz = (
        (Fx_f * mx.sin(df) + Fy_f * mx.cos(df)) * float(vp.a)
        - (Fx_r * mx.sin(dr) + Fy_r * mx.cos(dr)) * float(vp.b)
    )
    r_dot_raw = (Mz - float(vp.yaw_damp) * r) / float(vp.Iz)
    vx_eff = mx.maximum(mx.abs(vx), mx.array(float(vp.U_min), dtype=getattr(s, 'dtype', mx.float32)))
    mu_y = min(float(vp.tire_params_f.mu_y), float(vp.tire_params_r.mu_y))
    r_dot = apply_yaw_saturation_mlx(r, r_dot_raw, mu_y, float(vp.g), vx_eff, float(vp.yaw_sat_gain))

    # 姿态与世界坐标速度
    x_dot = vx * mx.cos(psi) - vy * mx.sin(psi)
    y_dot = vx * mx.sin(psi) + vy * mx.cos(psi)
    psi_dot = r
    # 统一所有导数的批量形状，避免在堆叠时出现形状不一致
    tshape = getattr(vx, 'shape', ())
    bs = lambda a: mx.reshape(a, tshape)
    ds = mx.stack([
        bs(vx_dot),
        bs(vy_dot),
        bs(r_dot),
        bs(x_dot),
        bs(y_dot),
        bs(psi_dot),
    ], axis=-1)

    ay = vy_dot + r * vx
    aux = {
        'df': df,
        'dr': dr,
        'alpha_f': alpha_f,
        'alpha_r': alpha_r,
        'Fy_f': Fy_f,
        'Fy_r': Fy_r,
        'Fx_f': Fx_f,
        'Fx_r': Fx_r,
        'ay': ay,
    }
    # 等效 squeeze(0)
    if hasattr(ds, 'shape') and len(ds.shape) >= 2 and ds.shape[0] == 1:
        ds_out = ds[0]
    else:
        ds_out = ds
    return ds_out, aux


def derivatives_speed_cmd_mlx(
    s: object,
    df: object,
    dr: object,
    vp: Vehicle3DOF,
    U_cmd: object,
    k_v: float = 1.0,
    front_bias: float = 0.5,
    rear_bias: float = 0.5,
) -> Tuple[object, Dict[str, object]]:
    """基于速度指令的 3DOF 导数计算（为 MPPI 使用，MLX-only）。

    - 使用 ax_cmd = k_v * (U_cmd - vx) 生成总纵向力 Fx_total
    - 按角度感知与轴向比例分配到前后轴
    - 经摩擦椭圆与侧偏力组合后进入动力学
    """
    if getattr(s, 'ndim', None) == 1:
        s = s[0:1, ...]
    vx = s[..., 0]
    vy = s[..., 1]
    r = s[..., 2]
    x = s[..., 3]
    y = s[..., 4]
    psi = s[..., 5]

    # 纵向驱动/制动（纯力）
    ax_cmd = float(k_v) * (U_cmd - vx)
    Fx_total = ax_cmd * float(vp.m)
    Fx_f_pure, Fx_r_pure = allocate_drive_mlx(Fx_total, df, dr, front_bias, rear_bias)

    # 调用主导数
    ds, aux = derivatives_dfdr_mlx(s, df, dr, vp, Fx_f_pure, Fx_r_pure)
    aux.update({'U_cmd': U_cmd, 'ax_cmd': ax_cmd})
    # squeeze(0)
    if hasattr(ds, 'shape') and len(ds.shape) >= 2 and ds.shape[0] == 1:
        ds_out = ds[0]
    else:
        ds_out = ds
    return ds_out, aux


@dataclass
class Vehicle3DOF:
    """
    3-DOF 非线性横向动力学的参数。

    与 threedof.py 中的数据类字段保持一致，但本模块不依赖 NumPy。
    """
    m: float = 1500.0
    Iz: float = 2500.0
    a: float = 1.2
    b: float = 1.6
    g: float = 9.81
    U_min: float = 0.5
    kf: float = 1.6e5
    kr: float = 1.7e5
    tire_model: str = 'pacejka'

    tire_params_f: PacejkaParams = field(default_factory=PacejkaParams)
    tire_params_r: PacejkaParams = field(default_factory=PacejkaParams)
    n_sw: float = 16.0
    c1: float = 0.3
    c2: float = 0.05

    yaw_damp: float = 300.0
    yaw_sat_gain: float = 4.0

    Fwx: float = 0.0
    Fwy: float = 0.0
    Fsx: float = 0.0
    Fsy: float = 0.0

    tire_long_params_f: PacejkaLongParams = field(default_factory=PacejkaLongParams)
    tire_long_params_r: PacejkaLongParams = field(default_factory=PacejkaLongParams)

    @property
    def L(self) -> float:
        return self.a + self.b