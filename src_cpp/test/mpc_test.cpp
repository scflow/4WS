#include "mpc.hpp"
#include "vehicle.hpp"
#include <cassert>
#include <cstdio>
#include <cmath>
#include <vector>

using namespace fourws;

int main() {
    VehicleParams p;
    p.U = 15.0;
    // 简单直线路径
    std::vector<PlanPoint> plan;
    for (int i = 0; i < 20; ++i) {
        plan.push_back(PlanPoint{double(i)*0.1, double(i)*0.5, 0.0, 0.0});
    }

    // 初始误差状态：稍微偏航与横摆率
    std::array<double,4> x_aug{0.2, 0.1, 0.05, 0.02};
    std::array<double,2> u0{0.0, 0.0};

    auto [df, dr] = solve_mpc_kin_dyn(x_aug, u0, p, plan, 0.05, 10);
    std::printf("df=%.6f dr=%.6f\n", df, dr);
    // 输出应有限幅且非 NaN
    assert(std::isfinite(df));
    assert(std::isfinite(dr));
    // 简单边界：不应过大
    assert(std::fabs(df) < 1.0);
    assert(std::fabs(dr) < 1.0);

    return 0;
}