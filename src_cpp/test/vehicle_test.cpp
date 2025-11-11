#include <iostream>
#include "vehicle.hpp"

using namespace fourws;

int main() {
    VehicleParams vp;
    vp.U = 20.0;
    vp.mu = 0.85;

    const double df = 0.05; // 约 2.86°
    std::cout << "L=" << vp.L() << "\n";
    std::cout << "K=" << vp.understeer_gradient() << "\n";
    std::cout << "r_ref=" << vp.r_ref(df) << "\n";
    std::cout << "r_cmd=" << vp.yaw_rate_cmd(df) << "\n";

    auto m = vp.to_map();
    std::cout << "map[U]=" << m["U"] << ", map[L]=" << m["L"] << ", map[K]=" << m["K"] << "\n";

    return 0;
}