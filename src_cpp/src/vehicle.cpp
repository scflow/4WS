// 车辆参数派生量与稳态横摆率计算（中文注释）
#include "vehicle.hpp"
#include <cmath>
#include <limits>

namespace fourws {

// 轴距 L = a + b
double VehicleParams::L() const { return a + b; }

// 有效速度幅值：保护近零速度避免 r/U 奇异
double VehicleParams::U_eff() const {
    const double absU = std::abs(U);
    return absU > U_min ? absU : U_min;
}

// 稳定性因数 K = (m/L) * (b/kr - a/kf)
double VehicleParams::understeer_gradient() const {
    const double Lval = L();
    return (m / Lval) * (b / kr - a / kf);
}

// 稳态横摆率 r_ref = U / (L + K U^2) * delta_f
double VehicleParams::r_ref(double delta_f) const {
    const double Ue = U_eff();
    const double K = understeer_gradient();
    return (Ue / (L() + K * Ue * Ue)) * delta_f;
}

// 摩擦限幅横摆率指令：|r| ≤ mu * g / U
double VehicleParams::yaw_rate_cmd(double delta_f) const {
    const double r = r_ref(delta_f);
    const double Ue = U_eff();
    const double r_max = (Ue < 1e-6) ? std::numeric_limits<double>::infinity() : (mu * g / Ue);
    if (r > r_max) return r_max;
    if (r < -r_max) return -r_max;
    return r;
}

// 参数与派生量导出为键值表（不含字符串字段）
VehicleParams::DictMap VehicleParams::to_map() const {
    DictMap out;
    out["m"] = m;
    out["Iz"] = Iz;
    out["a"] = a;
    out["b"] = b;
    out["width"] = width;
    out["track"] = track;
    out["kf"] = kf;
    out["kr"] = kr;
    out["U"] = U;
    out["mu"] = mu;
    out["g"] = g;
    out["U_min"] = U_min;
    out["U_blend"] = U_blend;
    // 字符串字段不在 double map 中，如 tire_model；可在需要时另行提供接口
    out["L"] = L();
    out["K"] = understeer_gradient();
    return out;
}

} // namespace fourws