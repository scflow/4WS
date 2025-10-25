import numpy as np
from typing import Tuple, Dict

from .params import VehicleParams


def matrices(p: VehicleParams) -> Tuple[np.ndarray, np.ndarray]:
    """根据论文模型生成状态空间矩阵 A、B。
    状态 X=[beta, r]^T；输入 U_in=[delta_f, delta_r]^T
    """
    m, Iz, a, b, kf, kr, U = p.m, p.Iz, p.a, p.b, p.kf, p.kr, p.U_eff()

    A11 = - (kf + kr) / (m * U)
    A12 = - (a * kf - b * kr) / (m * U * U) - 1.0
    A21 = - (a * kf - b * kr) / Iz
    A22 = - (a * a * kf + b * b * kr) / (Iz * U)

    B11 = kf / (m * U)
    B12 = kr / (m * U)
    B21 = a * kf / Iz
    B22 = - b * kr / Iz

    A = np.array([[A11, A12], [A21, A22]], dtype=float)
    B = np.array([[B11, B12], [B21, B22]], dtype=float)
    return A, B


def derivatives(x: np.ndarray, delta_f: float, delta_r: float, p: VehicleParams) -> Dict[str, float | np.ndarray]:
    """计算状态导数与观测量。
    返回：xdot, ay, Fy_f, Fy_r, alpha_f, alpha_r
    """
    beta, r = float(x[0]), float(x[1])
    A, B = matrices(p)
    u = np.array([delta_f, delta_r], dtype=float)
    xdot = A @ np.array([beta, r]) + B @ u

    beta_dot, r_dot = float(xdot[0]), float(xdot[1])
    U = p.U_eff()
    ay = U * (r + beta_dot)  # 横向加速度

    # 轮胎侧向力与侧偏角
    alpha_f = beta + p.a * r / U - delta_f
    alpha_r = beta - p.b * r / U - delta_r
    Fy_f = -p.kf * alpha_f
    Fy_r = -p.kr * alpha_r

    return {
        "xdot": np.array([beta_dot, r_dot]),
        "ay": ay,
        "Fy_f": Fy_f,
        "Fy_r": Fy_r,
        "alpha_f": alpha_f,
        "alpha_r": alpha_r,
    }