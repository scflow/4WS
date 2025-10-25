import os
import sys
import numpy as np

# 让脚本可找到 src 模块
ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
sys.path.append(ROOT)

from src.params import VehicleParams
from src.model import derivatives
from src.strategy import ratio_rear_steer, RatioConfig, ideal_yaw_rate, TrackingConfig


def rk4_step(x, delta_f, compute_delta_r, dt, p: VehicleParams):
    """单步 RK4，compute_delta_r(x) 返回当前后轮转角。"""
    def f(state, df):
        dr = compute_delta_r(state)
        return derivatives(state, df, dr, p)["xdot"]

    k1 = f(x, delta_f)
    k2 = f(x + 0.5 * dt * k1, delta_f)
    k3 = f(x + 0.5 * dt * k2, delta_f)
    k4 = f(x + dt * k3, delta_f)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def run_sim(amplitude=0.05, T=5.0, dt=0.01):
    p = VehicleParams()

    t = np.arange(0.0, T + 1e-9, dt)
    x_ratio = np.array([0.0, 0.0])
    x_track = np.array([0.0, 0.0])

    data = {
        "t": [],
        "beta_ratio": [], "r_ratio": [], "ay_ratio": [], "df_ratio": [], "dr_ratio": [],
        "beta_track": [], "r_track": [], "ay_track": [], "df_track": [], "dr_track": [], "r_cmd": [],
    }

    r_cfg = RatioConfig()
    tr_cfg = TrackingConfig()

    for ti in t:
        # 前轮阶跃输入（0.5s 开始）
        df = amplitude if ti >= 0.5 else 0.0

        # 比例律策略
        dr_ratio = ratio_rear_steer(df, p, r_cfg)
        out_r = derivatives(x_ratio, df, dr_ratio, p)
        x_ratio = rk4_step(x_ratio, df, lambda s: dr_ratio, dt, p)

        # 跟踪策略：随状态更新
        dr_track, diag = ideal_yaw_rate(df, x_track, p, tr_cfg)
        out_t = derivatives(x_track, df, dr_track, p)
        x_track = rk4_step(x_track, df, lambda s: ideal_yaw_rate(df, s, p, tr_cfg)[0], dt, p)

        # 记录
        data["t"].append(ti)
        data["beta_ratio"].append(x_ratio[0])
        data["r_ratio"].append(x_ratio[1])
        data["ay_ratio"].append(out_r["ay"])
        data["df_ratio"].append(df)
        data["dr_ratio"].append(dr_ratio)

        data["beta_track"].append(x_track[0])
        data["r_track"].append(x_track[1])
        data["ay_track"].append(out_t["ay"])
        data["df_track"].append(df)
        data["dr_track"].append(dr_track)
        data["r_cmd"].append(diag["r_cmd"]) if df != 0 else data["r_cmd"].append(0.0)

    for k in list(data.keys()):
        data[k] = np.array(data[k])
    return p, data


def save_outputs(data):
    out_dir = os.path.join(ROOT, "examples", "out")
    os.makedirs(out_dir, exist_ok=True)

    # 仅导出 CSV（移除 Matplotlib 图表）
    arr = np.vstack([
        data["t"], data["df_track"], data["dr_track"], data["r_cmd"],
        data["r_track"], data["beta_track"], data["ay_track"],
        data["r_ratio"], data["beta_ratio"], data["ay_ratio"],
    ]).T
    header = (
        "t,df,dr_track,r_cmd,r_track,beta_track,ay_track,"
        "r_ratio,beta_ratio,ay_ratio"
    )
    csv_path = os.path.join(out_dir, "step_demo.csv")
    np.savetxt(csv_path, arr, delimiter=",", header=header, comments="")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    params, data = run_sim()
    save_outputs(data)
    print("Simulation complete. Outputs saved to examples/out/step_demo.csv")