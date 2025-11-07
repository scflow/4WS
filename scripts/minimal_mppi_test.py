import os
import sys
import numpy as np

# 确保项目根目录在 Python 路径中，便于导入 src 包
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.params import VehicleParams
from src.mppi_iface import MPPIController4WS


def straight_plan(n=50, dx=1.0):
    return [{"x": i * dx, "y": 0.0} for i in range(n)]


def main():
    params = VehicleParams(U=15.0)
    dt = 0.05
    delta_max = np.deg2rad(20.0)
    dU_max = 3.0
    U_max = 30.0

    ctrl = MPPIController4WS(
        params=params,
        dt=dt,
        plan_provider=straight_plan,
        delta_max=delta_max,
        dU_max=dU_max,
        U_max=U_max,
        model_type="2dof",
        num_samples=32,
        horizon=25,
        lambda_=30.0,
    )

    # state: [x, y, psi, beta, r, U, df, dr]
    s0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, params.U, 0.0, 0.0], dtype=np.float32)

    u = ctrl.command(s0)
    print("u shape:", np.shape(u))
    print("u:", u)


if __name__ == "__main__":
    main()