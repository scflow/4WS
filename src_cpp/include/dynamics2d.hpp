#pragma once
#include "dynamics.hpp"
#include "vehicle.hpp"

namespace fourws {

// 面向对象封装：提供 2DOF 动力学计算的类接口，避免函数堆叠
class Dynamics2DOF {
public:
    explicit Dynamics2DOF(const VehicleParams& params) : p_(params) {}

    // 更新车辆参数（例如速度、胎模选择等）
    void set_params(const VehicleParams& params) { p_ = params; }
    const VehicleParams& params() const { return p_; }

    // 计算侧偏角（前/后）
    std::pair<double,double> slip_angles(double beta, double r, double df, double dr) const;

    // 计算横向力（前/后）
    std::pair<double,double> lateral_forces(double alpha_f, double alpha_r) const;

    // 计算状态导数与观测量
    Derivatives2D derivatives(double beta, double r, double df, double dr) const;

private:
    VehicleParams p_;
};

} // namespace fourws