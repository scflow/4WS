# MPPI 控制器（4WS）成本函数与动力学模型说明

更新时间：2025-11-05

本文汇总当前项目中 MPPI 控制（四轮转向 4WS）的状态/动作定义、动力学模型、运行代价（成本函数）构成、参考轨迹与误差计算方式，以及主要超参数与后端集成要点，便于调参与问题定位。

## 总览
- 实现位置：`src/mppi_iface.py` 中的 `MPPIController4WS`
- 状态向量：`s = [x, y, psi, beta, r, U, delta_f, delta_r]`
- 动作向量（微分形式）：`u = [d_delta_f, d_delta_r, dU]`
- 时间步长：`dt`（来自 `SimEngine` 的控制周期，默认 `0.02s`）
- 采样与滚动参数：`num_samples=128`，`horizon=20`，`lambda_=120.0`
- 设备选择优先级：优先 `MPS`（macOS），否则 `CUDA`，否则 `CPU`
- 与后端集成：`SimEngine._autop_update_mppi()` 在自动跟踪时调用 `controller.command(s)` 并写回 `df/dr/U`

## 动力学模型（Torch）
实现于 `MPPIController4WS._dynamics(s, u)`：

1. 解包状态与动作（支持批量维度）：
   - 若状态维度为 8，则解出 `df_cur/dr_cur`；否则默认从 0 起始。
2. 速度与转角更新（带边界）：
   - `U_next = clamp(U + dU, 0, U_max)`
   - `df_next = clamp(df_cur + d_df, -delta_max, +delta_max)`
   - `dr_next = clamp(dr_cur + d_dr, -delta_max, +delta_max)`
3. 侧偏角与横向力（复用 2DOF 公式）：
   - `alpha_f, alpha_r = slip_angles_2dof_torch(beta, r, df_next, dr_next, a, b, U_next)`
   - `Fy_f, Fy_r = lateral_forces_2dof_torch(alpha_f, alpha_r, params)`
4. 2DOF 动力学主方程：
   - 质量 `m`、惯量 `Iz` 来自 `VehicleParams`
   - `beta_dot = (Fy_f + Fy_r) / (m * max(U_next, 1e-6)) - r`
   - `r_dot    = (a * Fy_f - b * Fy_r) / Iz`
5. 几何部分：
   - `x_dot = U_next * cos(psi + beta)`
   - `y_dot = U_next * sin(psi + beta)`
   - `psi_dot = r`
6. 欧拉积分：
   - `s_next = [x + dt*x_dot, y + dt*y_dot, psi + dt*psi_dot, beta + dt*beta_dot, r + dt*r_dot, U_next, df_next, dr_next]`

### 动作边界与噪声
- 角速度上限（相对于转角上限）：`delta_rate_max = delta_rate_frac * delta_max`（由 `SimEngine` 传入，当前为 `1.0`）
- 微分动作边界：`d_delta_bound = delta_rate_max * dt`
  - `u_min = [-d_delta_bound, -d_delta_bound, -dU_max*dt]`
  - `u_max = [+d_delta_bound, +d_delta_bound, +dU_max*dt]`
- 动作噪声协方差（对 `d_df/d_dr/dU`）：
  - `diag([0.03, 0.03, 0.15])`

## 参考轨迹与误差计算
- 轨迹来源：`ENGINE.plan`（经 `/api/plan/quintic` 或 `/api/plan/circle` 生成），格式为数组 `[ {t, x, y, psi}, ... ]`
- 计算流程（见 `_running_cost`）：
  1. 提取参考 `px/py` 并计算段航向 `seg_psi = atan2(diff(py), diff(px))`
  2. 就地查找最近点索引 `di`，基于两点距离最小化
  3. 横向误差 `e_y`：将当前位姿相对段航向旋转到横向坐标，`e_y = -ex*sin(psi_base) + ey*cos(psi_base)`
  4. 航向误差 `e_psi`：`e_psi = wrap(psi_base - psi)` 到 `[-pi, pi]`
  5. 参考曲率 `kappa_ref`：对邻近段航向差 `dpsi_seg` 除以均值步长 `ds_avg` 的近似
  6. 期望速度 `U_des`：基于横向加速度上限 `ay_limit ≈ mu*g*ay_limit_coeff`，`U_des = sqrt(ay_limit / |kappa_ref|)`（直线则保持当前速度）

## 成本函数（运行代价）
实现于 `MPPIController4WS._running_cost(s, u)`，当前构成为：

1. 轨迹跟踪与几何一致性
   - 横向误差：`w_lat * e_y^2`，`w_lat = 2000.0`
   - 航向误差：`w_head * e_psi^2`，`w_head = 4000.0`
   - 曲率跟踪：`24.0 * (kappa_cur - kappa_ref)^2`，其中 `kappa_cur ≈ |r|/max(U,1e-6)`
   - 速度跟踪：`w_speed * (U - U_des)^2`，`w_speed = 20.0`
   - 横摆率抑制：`0.03 * r^2`（权重较轻，避免低速“横移”趋势）
   - 侧偏角抑制：`8.0 * beta^2`

2. 控制代价与平滑
   - 控制努力：`w_u * ||u||^2`，`w_u = 1.0`
   - 控制变化：`w_du * ||u - u_prev||^2`，`w_du = 2.0`（使用上一时刻动作缓存）

3. 前后轮相位约束（低速强反相，高速弱同相）
   - 速度插值系数：
     - 阈值：`U1 = 5.0`，`U2 = 20.0`
     - 插值：`s_lin = clamp((U - U1)/(U2 - U1), 0, 1)`
     - 比例律：`k(U) = k_low*(1 - s_lin) + k_high*s_lin`，`k_low = -0.30`，`k_high = 0.10`
   - 相位误差项：`w_phase * (dr - k(U)*df)^2`
     - 权重：`w_phase = w_phase_base*(1 - s_lin) + 0.25*w_phase_base*s_lin`
     - 基值：`w_phase_base = 300.0`

4. 同向惩罚（直接压制蟹行）
   - 定义：`same_sign_pos = clamp(df * dr, min=0)`（同号且同向为正）
   - 成本：`w_phase_sign * (1 - s_lin) * (same_sign_pos)^2`
   - 权重：`w_phase_sign = 120.0`（低速更强，高速弱化）

5. 理想横摆率前馈一致性
   - 参考：`dr_ff = ideal_yaw_rate(df, [beta, r], params)`（见 `src/strategy.py`）
   - 成本：`w_yaw_ff * (dr - dr_ff)^2`，`w_yaw_ff = 80.0`

> 注：当参考轨迹不足（点数 < 2）时，成本退化为仅包含控制努力项，以避免无参考下的异常行为。

## 超参数与默认值（关键项）
- `delta_max`：转角幅度上限（引擎中默认 30°）
- `delta_rate_frac`：角速度占比（相对 `delta_max`），当前 `SimEngine` 设为 `1.0`，允许更快到达目标角度
- `U_max`：速度上限（默认取 `U_cruise`）
- `dU_max`：每秒速度变化上限（默认 `1.5 m/s`）
- `mu/g/ay_limit_coeff`：横向加速度限的近似项（速度期望推导用）
- `num_samples/horizon/lambda_`：MPPI 的采样与滚动窗口参数

## 与后端 SimEngine 的集成
- 入口：`SimEngine._autop_update_mppi()`
  - 延迟初始化控制器，按设备可用性选择 `mps/cuda/cpu`
  - 打包当前状态为 `s_np` 并调用 `controller.command(s_np)`
  - 将返回的微分动作积分为下一时刻 `df/dr/U` 并写回 `ENGINE.ctrl`
  - 初始化失败时回退到 `simple` 模式
- 相关参数传递：
  - `delta_max/dU_max/U_max/delta_rate_frac` 等由引擎传入控制器构造函数
  - `plan_provider` 为 `ENGINE.plan` 的轻量引用，成本函数每次从该引用读取参考轨迹
- 重置行为（`/api/sim/reset`）：
  - 停止自动跟踪、清空参考轨迹、将前后轮角归零

## 调参与现象说明
- 低速急弯场景：
  - 建议提高 `w_phase_base` 或 `w_yaw_ff`，抑制同向蟹行并提升“转起来”的倾向
  - 保持较轻的 `r^2` 权重，避免算法优先选择横移
- 高速并线或大半径弯：
  - 相位约束自动弱化；若出现过度反相，可降低 `w_phase_base` 或提高 `k_high`
- 响应速度：
  - 若转角响应过快/过慢，可在配置里把 `delta_rate_frac` 调整到 `0.8–1.0`

## 参考文件
- `src/mppi_iface.py`：MPPI 控制封装、动力学与成本函数
- `src/strategy.py`：`ideal_yaw_rate` 前馈参考的实现
- `src/sim.py`：引擎参数、自动跟踪集成、MPPI 调用
- `app.py`：参考轨迹生成接口（五次多项式、圆弧）

---
如需针对特定场景（小半径弯、S 曲线、泊车入位）输出一套推荐参数与曲线对比，请在文档后补充场景描述，我可以添加相应的调参建议与效果说明。