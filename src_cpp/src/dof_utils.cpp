#include "dof_utils.hpp"
#include <cmath>
#include <limits>

namespace fourws {

double yaw_rate_limit(double mu_y, double g, double vx_eff, double eps) {
    const double vmag = std::abs(vx_eff);
    if (vmag < eps) return std::numeric_limits<double>::infinity();
    return mu_y * g / vmag;
}

double apply_yaw_saturation(double r, double r_dot, double mu_y, double g, double vx_eff, double gain) {
    const double r_max = yaw_rate_limit(mu_y, g, vx_eff);
    if (std::abs(r) > r_max) {
        r_dot -= gain * (std::abs(r) - r_max) * ((r >= 0.0) ? 1.0 : -1.0);
    }
    return r_dot;
}

std::pair<double,double> body_to_world_2dof(double U, double beta, double psi) {
    const double c = std::cos(psi + beta);
    const double s = std::sin(psi + beta);
    return {U * c, U * s};
}

double curvature_4ws(double df, double dr, double L) {
    const double L_eff = (std::abs(L) < 1e-9) ? 1e-9 : L;
    return (std::tan(df) - std::tan(dr)) / L_eff;
}

} // namespace fourws