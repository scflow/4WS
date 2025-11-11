// 轨迹规划（中文注释）：
// - 五次多项式边界条件解算与采样
// - 单圆弧插值（基于起终点位置与航向）
// 与 Python src/planner.py 对齐
#pragma once
#include <vector>
#include <string>

namespace fourws {

struct Pose2D {
    double x{0.0};
    double y{0.0};
    double psi{0.0}; // 弧度
};

struct PlanPoint {
    double t{0.0};
    double x{0.0};
    double y{0.0};
    double psi{0.0};
};

// 求解五次多项式系数（p, v, a 在 t=0 与 t=T 的边界）
std::vector<double> solve_quintic(double p0, double v0, double a0,
                                  double pT, double vT, double aT,
                                  double T);

// 在给定时间序列上采样（位置、速度、一阶加速度）
void sample_poly(const std::vector<double>& coeffs,
                 const std::vector<double>& t,
                 std::vector<double>& p,
                 std::vector<double>& v,
                 std::vector<double>& a);

// 独立 x/y 五次多项式规划，返回 {t,x,y,psi}
std::vector<PlanPoint> plan_quintic_xy(const Pose2D& start,
                                       const Pose2D& end,
                                       double T,
                                       int N,
                                       double U_start);

// 圆弧规划：起点/终点位置与航向，选择更贴近终点航向的左右圆心方向
std::vector<PlanPoint> plan_circle_arc(const Pose2D& start,
                                       const Pose2D& end,
                                       double T,
                                       int N,
                                       double U_start);

} // namespace fourws