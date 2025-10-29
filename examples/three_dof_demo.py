"""
3DOF 非线性侧向动力学仿真实验（基于 documents/3dof.md）

实验描述：
- 车辆初始速度约 70 km/h（19.44 m/s）
- 在 t=1s 时给方向盘一个阶跃输入（30°）并保持
- 使用 Pacejka 轮胎模型计算侧向力，后轮角度由简单 4WS 控制律给出
- 输出时域数据到 examples/out/three_dof_demo.csv
"""

import os
import sys
import numpy as np

# 允许从项目根目录导入 src 包
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.threedof import Vehicle3DOF, State3DOF, simulate


def main():
    # 参数设置（可根据论文或车辆参数调整）
    vp = Vehicle3DOF(
        m=1500.0,
        Iz=2500.0,
        a=1.2,
        b=1.6,
        n_sw=16.0,
        c1=0.3,
        c2=0.05,
    )

    # 初始状态
    s0 = State3DOF(vx=19.44, vy=0.0, r=0.0, x=0.0, y=0.0, psi=0.0)

    # 输入：方向盘阶跃（默认单位为弧度，这里用角度转弧度）
    step_deg = 30.0
    step_rad = np.deg2rad(step_deg)

    def delta_sw_fn(t: float) -> float:
        return step_rad if t >= 1.0 else 0.0

    # 仿真参数
    T = 10.0
    dt = 0.01

    out = simulate(
        T=T,
        dt=dt,
        vp=vp,
        s0=s0,
        delta_sw_fn=delta_sw_fn,
        lambda_f_fn=None,
        lambda_r_fn=None,
    )

    # 导出 CSV
    out_dir = os.path.join("examples", "out")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "three_dof_demo.csv")

    cols = [
        "t", "vx", "vy", "r", "x", "y", "psi",
        "df", "dr", "Fx_f", "Fx_r", "Fy_f", "Fy_r", "alpha_f", "alpha_r", "ay",
    ]
    data = np.column_stack([out[k] for k in cols])

    header = ",".join(cols)
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")

    print(f"Saved 3DOF demo results to {csv_path}")


if __name__ == "__main__":
    main()