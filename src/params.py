from dataclasses import dataclass
import numpy as np


@dataclass
class VehicleParams:
    """车辆参数与派生量，按论文口径。

    字段单位：SI（m, kg, s, rad, N）
    - m: 车辆质量
    - Iz: 横摆转动惯量
    - a: 质心到前轴距离
    - b: 质心到后轴距离
    - width: 车身宽度（用于前端绘制矩形）
    - kf: 前轴侧偏刚度（轴级聚合）
    - kr: 后轴侧偏刚度（轴级聚合）
    - U: 纵向速度（常值或缓变，仿真中可更新）
    - mu: 附着系数（用于摩擦极限约束）
    - g: 重力加速度
    - U_min: 速度下限，避免 r/U 奇异
    """

    m: float = 1500.0
    Iz: float = 2500.0
    a: float = 1.2
    b: float = 1.6
    width: float = 1.8
    track: float = 1.5
    kf: float = 1.6e5
    kr: float = 1.7e5
    U: float = 20.0
    mu: float = 0.85
    g: float = 9.81
    U_min: float = 0.5
    # 轮胎模型选择：'pacejka'（魔术方程）或 'linear'（线性侧偏刚度）
    tire_model: str = 'pacejka'

    @property
    def L(self) -> float:
        return self.a + self.b

    def U_eff(self) -> float:
        """数值保护下的有效速度。"""
        return max(self.U, self.U_min)

    def understeer_gradient(self) -> float:
        """稳定性因数 K（Understeer Gradient）默认口径。
        K = (m/L) * (b/kr - a/kf)
        可根据具体论文口径在配置阶段替换。
        """
        return (self.m / self.L) * (self.b / self.kr - self.a / self.kf)

    def r_ref(self, delta_f: float) -> float:
        """理想稳态横摆率（未施加摩擦极限约束）。
        r_ref = U / (L + K U^2) * delta_f
        """
        U = self.U_eff()
        K = self.understeer_gradient()
        return (U / (self.L + K * U * U)) * float(delta_f)

    def yaw_rate_cmd(self, delta_f: float) -> float:
        """考虑摩擦极限的横摆率指令 r_cmd。
        |r| ≤ mu * g / U
        """
        r = self.r_ref(delta_f)
        r_max = self.mu * self.g / self.U_eff()
        return float(np.clip(r, -r_max, r_max))

    def to_dict(self) -> dict:
        return {
            "m": self.m,
            "Iz": self.Iz,
            "a": self.a,
            "b": self.b,
            "width": self.width,
            "track": self.track,
            "kf": self.kf,
            "kr": self.kr,
            "U": self.U,
            "mu": self.mu,
            "g": self.g,
            "U_min": self.U_min,
            "tire_model": self.tire_model,
            "L": self.L,
            "K": self.understeer_gradient(),
        }