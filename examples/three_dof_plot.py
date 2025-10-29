"""
读取 CSV 并绘制 r(t) 与 a_y(t) 曲线，用于与论文曲线对比。

用法：
  python3 examples/three_dof_plot.py [path_to_csv]

若不提供参数，默认读取 examples/out/three_dof_demo.csv 并保存 PNG 到同目录。
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = os.path.join("examples", "out", "three_dof_demo.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # 读取数据
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    t = data["t"]
    r = data["r"]
    ay = data["ay"] if "ay" in data.dtype.names else None

    # 如果没有 ay 列，则用有限差分近似 ay = d(vy)/dt + r*vx
    if ay is None:
        vx = data["vx"]
        vy = data["vy"]
        dt = np.mean(np.diff(t))
        dvy_dt = np.gradient(vy, dt)
        ay = dvy_dt + r * vx

    # 作图
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax[0].plot(t, r, label="yaw rate r")
    ax[0].set_ylabel("r [rad/s]")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(t, ay, label="lateral accel a_y")
    ax[1].set_ylabel("a_y [m/s^2]")
    ax[1].set_xlabel("t [s]")
    ax[1].grid(True)
    ax[1].legend()

    out_png = os.path.splitext(csv_path)[0] + ".png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved plot to {out_png}")


if __name__ == "__main__":
    main()