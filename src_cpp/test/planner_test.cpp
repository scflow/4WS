#include <cassert>
#include <cstdio>
#include <cmath>
#include <vector>
#include "planner.hpp"

using namespace fourws;

static inline bool near(double a, double b, double tol=1e-6) {
    return std::abs(a-b) <= tol;
}

int main() {
    // Test: solve_quintic + sample_poly
    {
        double p0=0.0, v0=1.0, a0=0.0;
        double pT=10.0, vT=1.0, aT=0.0;
        double T=2.0;
        auto c = solve_quintic(p0,v0,a0,pT,vT,aT,T);
        std::vector<double> tt = {0.0, T};
        std::vector<double> p, v, a;
        sample_poly(c, tt, p, v, a);
        assert(near(p.front(), p0, 1e-6));
        assert(near(v.front(), v0, 1e-6));
        assert(near(a.front(), a0, 1e-6));
        assert(near(p.back(), pT, 1e-6));
        assert(near(v.back(), vT, 1e-6));
        assert(near(a.back(), aT, 1e-6));
    }

    // Test: plan_quintic_xy basic
    {
        Pose2D s{0.0, 0.0, 0.0};
        Pose2D e{10.0, 10.0, M_PI/2};
        double T=2.0; int N=21; double U=3.0;
        auto path = plan_quintic_xy(s, e, T, N, U);
        assert(path.size() == (size_t)N);
        assert(near(path.front().x, s.x, 1e-6));
        assert(near(path.front().y, s.y, 1e-6));
        assert(near(path.back().x, e.x, 1e-3));
        assert(near(path.back().y, e.y, 1e-3));
        // 航向接近终端期望
        assert(std::abs(path.back().psi - e.psi) < 0.1);
    }

    // Test: plan_circle_arc simple left/right selection
    {
        Pose2D s{0.0, 0.0, 0.0};
        Pose2D e{10.0, 10.0, M_PI/2};
        auto arc = plan_circle_arc(s, e, 2.0, 21, 3.0);
        assert(arc.size() == 21u);
        assert(near(arc.front().x, s.x, 1e-6));
        assert(near(arc.front().y, s.y, 1e-6));
        assert(near(arc.back().x, e.x, 1e-3));
        assert(near(arc.back().y, e.y, 1e-3));
        assert(std::abs(arc.back().psi - e.psi) < 0.2);
    }

    // Test: plan_circle_arc degeneracy -> line fallback
    {
        // 令 denom=dx*nx+dy*ny≈0：起点朝向与位移正交
        Pose2D s{0.0, 0.0, 0.0}; // nx=0, ny=1 for left; nx=0, ny=-1 for right
        Pose2D e{10.0, 0.0, 0.0};
        auto arc = plan_circle_arc(s, e, 2.0, 11, 3.0);
        assert(arc.size() == 11u);
        // 应退化为直线插值，psi=atan2(dy,dx)=0
        for (const auto& p : arc) {
            assert(std::abs(p.psi - 0.0) < 1e-6);
        }
        assert(near(arc.front().x, s.x, 1e-6));
        assert(near(arc.front().y, s.y, 1e-6));
        assert(near(arc.back().x, e.x, 1e-6));
        assert(near(arc.back().y, e.y, 1e-6));
    }

    // Test: identical start/end
    {
        Pose2D s{1.0, 2.0, 0.3};
        Pose2D e=s;
        auto arc = plan_circle_arc(s, e, 1.0, 5, 2.0);
        assert(arc.size() == 5u);
        for (const auto& p : arc) {
            assert(near(p.x, s.x, 1e-6));
            assert(near(p.y, s.y, 1e-6));
            assert(near(p.psi, s.psi, 1e-6));
        }
    }

    std::printf("PASS: planner basic checks\n");
    return 0;
}

