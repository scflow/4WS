// 车辆与控制相关结构与方法
#pragma once
// 单位约定：SI（m, kg, s, rad, N）
// 目的：提供 2DOF/4WS 仿真所需的基本数据结构与派生方法
#include <string>
#include <unordered_map>

namespace fourws {

// 车体状态（2DOF近似）：位置、航向、侧偏角、横摆率
struct CarState {
    double x{0.0};
    double y{0.0};
    double psi{0.0};
    double beta{0.0};
    double r{0.0};
};

// 控制输入：纵向速度与前/后轮转角（弧度）
struct Control {
    double U{10.0};
    double delta_f{0.0};
    double delta_r{0.0};
};

// 轨迹记录配置：是否启用、保留时长、最大点数
struct TrackSettings {
    bool enabled{true};
    double retention_sec{30.0};
    int max_points{20000};
};

// 车辆参数与派生量
struct VehicleParams {
    double m{1500.0};
    double Iz{2500.0};
    double a{1.2};
    double b{1.6};
    double width{1.8};
    double track{1.5};
    double kf{1.6e5};
    double kr{1.7e5};
    double U{20.0};
    double mu{0.85};
    double g{9.81};
    double U_min{1e-3};
    double U_blend{0.3};
    std::string tire_model{"pacejka"};

    // 轴距：L = a + b
    double L() const;
    // 有效纵向速度幅值：|U| 的下限保护，避免 U→0 奇异
    double U_eff() const;
    // 稳定性因数：K = (m/L) * (b/kr - a/kf)
    double understeer_gradient() const;
    // 稳态横摆率参考：r_ref = U / (L + K U^2) * delta_f
    double r_ref(double delta_f) const;
    // 摩擦限幅横摆率指令：|r| ≤ mu * g / U
    double yaw_rate_cmd(double delta_f) const;

    // 导出数值字段为键值表（不含字符串字段）
    using DictMap = std::unordered_map<std::string, double>;

    // 返回主要参数与派生量键值对：m、Iz、a、b、kf、kr、U、mu、g、U_min、U_blend、L、K 等
    DictMap to_map() const;
};

} // namespace fourws