"""
3DOF 扩展场景仿真：制动/驱动联合工况、双移线、蛇行输入。

生成多份 CSV 便于与论文曲线对比：
- examples/out/three_dof_brake.csv
- examples/out/three_dof_drive.csv
- examples/out/three_dof_double.csv
- examples/out/three_dof_sine.csv
"""

import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.threedof import Vehicle3DOF, State3DOF, simulate


def save_csv(path: str, out: dict):
    cols = [
        "t", "vx", "vy", "r", "x", "y", "psi",
        "df", "dr", "Fx_f", "Fx_r", "Fy_f", "Fy_r", "alpha_f", "alpha_r", "ay",
    ]
    data = np.column_stack([out[k] for k in cols])
    header = ",".join(cols)
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def main():
    vp = Vehicle3DOF()
    s0 = State3DOF(vx=19.44)
    T = 10.0
    dt = 0.01

    out_dir = os.path.join("examples", "out")
    os.makedirs(out_dir, exist_ok=True)

    # 场景1：阶跃转向 + 制动（滑移率 -0.1）
    step_deg = 30.0
    step_rad = np.deg2rad(step_deg)
    def delta_sw_step(t: float) -> float:
        return step_rad if t >= 1.0 else 0.0
    def lambda_brake(t: float, s: State3DOF) -> float:
        return -0.1 if t >= 1.0 else 0.0

    out_brake = simulate(
        T, dt, vp, s0, delta_sw_step, lambda_brake, lambda_brake
    )
    save_csv(os.path.join(out_dir, "three_dof_brake.csv"), out_brake)

    # 场景2：阶跃转向 + 驱动（滑移率 +0.05）
    def lambda_drive(t: float, s: State3DOF) -> float:
        return 0.05 if t >= 1.0 else 0.0
    out_drive = simulate(
        T, dt, vp, s0, delta_sw_step, lambda_drive, lambda_drive
    )
    save_csv(os.path.join(out_dir, "three_dof_drive.csv"), out_drive)

    # 场景3：双移线（+30° 0.5s，然后 -30° 0.5s）
    def delta_sw_double(t: float) -> float:
        if 1.0 <= t < 1.5:
            return step_rad
        elif 1.5 <= t < 2.0:
            return -step_rad
        else:
            return 0.0
    out_double = simulate(T, dt, vp, s0, delta_sw_double)
    save_csv(os.path.join(out_dir, "three_dof_double.csv"), out_double)

    # 场景4：蛇行（正弦），从 t0 开始
    amp_deg = 10.0
    amp_rad = np.deg2rad(amp_deg)
    f = 0.5  # Hz
    t0 = 1.0
    def delta_sw_sine(t: float) -> float:
        if t < t0:
            return 0.0
        return amp_rad * np.sin(2 * np.pi * f * (t - t0))
    out_sine = simulate(T, dt, vp, s0, delta_sw_sine)
    save_csv(os.path.join(out_dir, "three_dof_sine.csv"), out_sine)

    print("Saved scenario CSVs to examples/out/")


if __name__ == "__main__":
    main()