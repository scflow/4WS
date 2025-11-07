from typing import Tuple, Dict
import mlx.core as mx
from .params import VehicleParams


def matrices_mlx(
    p: VehicleParams,
    dtype: object | None = None,
) -> Tuple[object, object]:
    """根据论文模型生成状态空间矩阵 A、B。
    状态 X=[beta, r]^T；输入 U_in=[delta_f, delta_r]^T
    与 src/twodof.py.matrices 保持公式一致。
    """
    U = float(p.U_eff())
    m, Iz, a, b, kf, kr = float(p.m), float(p.Iz), float(p.a), float(p.b), float(p.kf), float(p.kr)

    A11 = - (kf + kr) / (m * U)
    A12 = - (a * kf - b * kr) / (m * U * U) - 1.0
    A21 = - (a * kf - b * kr) / Iz
    A22 = - (a * a * kf + b * b * kr) / (Iz * U)

    B11 = kf / (m * U)
    B12 = kr / (m * U)
    B21 = a * kf / Iz
    B22 = - b * kr / Iz

    dt = dtype if dtype is not None else mx.float32
    A = mx.array([[A11, A12], [A21, A22]], dtype=dt)
    B = mx.array([[B11, B12], [B21, B22]], dtype=dt)
    return A, B


def slip_angles_2dof_mlx(
    beta: object,
    r: object,
    delta_f: object,
    delta_r: object,
    a: float,
    b: float,
    U: object,
) -> Tuple[object, object]:
    """2DOF 侧偏角。
    alpha_f = beta + a*r/U - delta_f
    alpha_r = beta - b*r/U - delta_r
    支持标量或批量张量（会按广播规则计算）。
    """
    U_eff = mx.maximum(U, mx.array(1e-6, dtype=getattr(U, 'dtype', mx.float32)))
    alpha_f = beta + (a * r) / U_eff - delta_f
    alpha_r = beta - (b * r) / U_eff - delta_r
    return alpha_f, alpha_r


def static_loads_2dof_mlx(
    a: float,
    b: float,
    m: float,
    g: float,
    dtype: object | None = None,
) -> Tuple[object, object]:
    """前后轴静态法向载荷（忽略载荷转移）。"""
    Fzf = (b * m * g) / (a + b)
    Fzr = (a * m * g) / (a + b)
    dt = dtype if dtype is not None else mx.float32
    return (
        mx.array(Fzf, dtype=dt),
        mx.array(Fzr, dtype=dt),
    )


def lateral_forces_2dof_mlx(
    alpha_f: object,
    alpha_r: object,
    p: VehicleParams,
) -> Tuple[object, object]:
    """根据选择的轮胎模型计算前/后轮横向力。

    - 线性模型（推荐用于 MPPI 批量并行）：Fy = -k * alpha（与惯例一致）
    - Pacejka 模型：使用 MLX 版 Pacejka（支持张量广播与梯度）。
    """
    model_sel = str(getattr(p, 'tire_model', 'linear') or 'linear').lower()
    if model_sel in ('linear', ''):
        kf = mx.array(float(p.kf), dtype=getattr(alpha_f, 'dtype', mx.float32))
        kr = mx.array(float(p.kr), dtype=getattr(alpha_r, 'dtype', mx.float32))
        Fy_f = (-kf) * alpha_f
        Fy_r = (-kr) * alpha_r
        return Fy_f, Fy_r

    # 使用 MLX 版 Pacejka，实现张量广播与梯度可用
    from .tire_mlx import lateral_force_dispatch_mlx, PacejkaParams
    Fzf_t, Fzr_t = static_loads_2dof_mlx(
        float(p.a), float(p.b), float(p.m), float(p.g),
        dtype=getattr(alpha_f, 'dtype', None),
    )
    tp_f = PacejkaParams(mu_y=float(p.mu))
    tp_r = PacejkaParams(mu_y=float(p.mu))

    Fy_f = lateral_force_dispatch_mlx(alpha_f, Fzf_t, model_sel, float(p.kf), tp_f)
    Fy_r = lateral_force_dispatch_mlx(alpha_r, Fzr_t, model_sel, float(p.kr), tp_r)
    return Fy_f, Fy_r


def derivatives_mlx(
    x: object,
    delta_f: object,
    delta_r: object,
    p: VehicleParams,
) -> Dict[str, object]:
    """计算 2DOF 状态导数与观测量（MLX-only）。

    参数：
    - x: [..., 2]，分别为 beta, r；支持批量维度
    - delta_f, delta_r: 与 x 广播的张量；支持批量
    - p: VehicleParams（数值参数将被常量注入）

    返回：字典，包含 xdot（[..., 2]）、ay、Fy_f、Fy_r、alpha_f、alpha_r
    """
    beta = x[..., 0]
    r = x[..., 1]

    U = mx.array(float(p.U_eff()), dtype=getattr(x, 'dtype', mx.float32))
    a = float(p.a)
    b = float(p.b)
    m = float(p.m)
    Iz = float(p.Iz)

    alpha_f, alpha_r = slip_angles_2dof_mlx(beta, r, delta_f, delta_r, a, b, U)
    Fy_f, Fy_r = lateral_forces_2dof_mlx(alpha_f, alpha_r, p)

    beta_dot = (Fy_f + Fy_r) / (m * U) - r
    r_dot = (a * Fy_f - b * Fy_r) / Iz

    ay = U * (r + beta_dot)

    xdot = mx.stack([beta_dot, r_dot], axis=-1)
    return {
        "xdot": xdot,
        "ay": ay,
        "Fy_f": Fy_f,
        "Fy_r": Fy_r,
        "alpha_f": alpha_f,
        "alpha_r": alpha_r,
    }