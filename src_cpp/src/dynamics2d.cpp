#include "dynamics2d.hpp"

namespace fourws {

std::pair<double,double> Dynamics2DOF::slip_angles(double beta, double r, double df, double dr) const {
    return slip_angles_2dof(beta, r, df, dr, p_);
}

std::pair<double,double> Dynamics2DOF::lateral_forces(double alpha_f, double alpha_r) const {
    return lateral_forces_2dof(alpha_f, alpha_r, p_);
}

Derivatives2D Dynamics2DOF::derivatives(double beta, double r, double df, double dr) const {
    return derivatives_2dof(beta, r, df, dr, p_);
}

} // namespace fourws