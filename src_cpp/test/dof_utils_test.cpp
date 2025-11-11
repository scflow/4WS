#include <cassert>
#include <cmath>
#include <cstdio>
#include "dof_utils.hpp"

using namespace fourws;

static inline bool near(double a, double b, double tol=1e-6) { return std::abs(a-b) <= tol; }

int main() {
    // yaw_rate_limit
    {
        double mu=0.85, g=9.81, vx=10.0;
        double rmax = yaw_rate_limit(mu,g,vx);
        assert(near(rmax, mu*g/vx, 1e-12));
        double rmax_inf = yaw_rate_limit(mu,g,0.0);
        assert(std::isinf(rmax_inf));
    }

    // apply_yaw_saturation
    {
        double mu=0.85, g=9.81, vx=0.5, gain=2.0;
        double r_dot = 0.05;
        double rmax = yaw_rate_limit(mu,g,vx);
        double r = rmax * 1.5; // exceed limit to trigger penalty
        double r_dot2 = apply_yaw_saturation(r, r_dot, mu,g,vx,gain);
        assert(r_dot2 < r_dot); // penalized
        // If below limit, unchanged
        double r_small = rmax * 0.5;
        double r_dot3 = apply_yaw_saturation(r_small, r_dot, mu,g,vx,gain);
        assert(near(r_dot3, r_dot, 1e-12));
    }

    // body_to_world_2dof
    {
        double U=10.0, beta=0.1, psi=0.2;
        auto [xd, yd] = body_to_world_2dof(U,beta,psi);
        assert(near(xd, U*std::cos(psi+beta), 1e-12));
        assert(near(yd, U*std::sin(psi+beta), 1e-12));
    }

    // curvature_4ws
    {
        double df=0.05, dr=-0.02, L=2.8;
        double k = curvature_4ws(df,dr,L);
        assert(near(k, (std::tan(df)-std::tan(dr))/L, 1e-12));
    }

    std::printf("PASS: dof_utils basic checks\n");
    return 0;
}