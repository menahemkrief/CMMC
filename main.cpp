#include "src/cmmc.hpp"
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

    Vector energy_groups_centers;
    for(std::size_t g=0; g<energy_groups_boundries.size()-1;++g) energy_groups_centers.push_back(0.5*(energy_groups_boundries[g]+energy_groups_boundries[g+1]));
    
    auto compton_engine = ComptonMatrixMC(energy_groups_centers, energy_groups_boundries, 200000, true, 0);

    // double const T = 2.0*units::me_c2 / units::k_boltz;
    double constexpr T = 10.0*units::kev_kelvin;
    Matrix m = compton_engine.calculate_S_matrix(T);

    Vector temperature_grid = {1e-2, 1., 3., 4., 6., 10., 20., 30., 40., 60., 80., 100.};
    for(auto& temp : temperature_grid){
        temp *= units::kev_kelvin;
    }

    compton_engine.set_tables(temperature_grid);
    auto const num_energy_groups = energy_groups_centers.size();
    Matrix tau(num_energy_groups, Vector(num_energy_groups, 0.0));
    Matrix dtau(num_energy_groups, Vector(num_energy_groups, 0.0));

    compton_engine.get_tau_matrix(15.*units::kev_kelvin, 1., 1., 1., tau, dtau);

}