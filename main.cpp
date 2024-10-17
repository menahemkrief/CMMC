#include "src/tau_matrix_monte_carlo.hpp"
#include "src/units/units.hpp"

int main(){
    
    Vector energy_groups_boundries = {
        0.001,
        0.00197172,
        0.00388768,
        0.00766542,
        0.0151141,
        0.0298007,
        0.0587586,
        0.115856,
        0.228435,
        0.450409,
        0.888081,
        1.75105,
        3.45258,
        6.80751,
        13.4225,
        26.4654,
        52.1824,
        102.889,
        202.869,
        400,
    };

    for(auto& e : energy_groups_boundries) e *= units::kev;

    Vector energy_groups_center;
    for(std::size_t g=0; g<energy_groups_boundries.size()-1;++g) energy_groups_center.push_back(0.5*(energy_groups_boundries[g]+energy_groups_boundries[g+1]));
    
    auto tau_engine = tau_matrix_monte_carlo_engine(energy_groups_center, energy_groups_boundries, 200000, true, 0);

    // double const T = 2.0*units::me_c2 / units::k_boltz;
    double constexpr T = 10.0*units::kev_kelvin;
    Matrix m = tau_engine.generate_S_matrix(T, false);

    Vector tmp_grid = {1e-2, 1., 3., 4., 6., 10., 20., 30., 40., 60., 80., 100.};
    for(auto& temp : tmp_grid){
        temp *= units::kev_kelvin;
    }

    tau_engine.generate_tables(tmp_grid);
    auto const num_energy_groups = energy_groups_center.size();
    Matrix tau(num_energy_groups, Vector(num_energy_groups, 0.0));
    Matrix dtau(num_energy_groups, Vector(num_energy_groups, 0.0));

    tau_engine.generate_tau_matrix(15.*units::kev_kelvin, 1., 1., 1., tau, dtau);

}