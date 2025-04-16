#include <limits>
#include <ctime>
#include <cmath>

#include "compton_matrix_mc.hpp"
#include "units/units.hpp"
#include "planck_integral/planck_integral.hpp"

#include <boost/math/special_functions/pow.hpp>

namespace compton_matrix_mc {

namespace machine_limits {
static double constexpr signaling_NaN = std::numeric_limits<double>::signaling_NaN();
static double constexpr min_double = 1024.*std::numeric_limits<double>::min();
}

ComptonMatrixMC::ComptonMatrixMC(
    Vector const compton_temperatures_,
    Vector const energy_groups_centers_,
    Vector const energy_groups_boundries_,
    std::size_t const num_of_samples_,
    std::optional<unsigned int> const seed_) :

    compton_temperatures(compton_temperatures_),
    energy_groups_centers(energy_groups_centers_),
    energy_groups_boundries(energy_groups_boundries_),
    num_energy_groups(energy_groups_centers.size()),
    num_of_samples(num_of_samples_),
    seed(seed_ ? seed_.value() : static_cast<unsigned int>(std::time(0))),
    sample_uniform_01(
        boost::random::mt19937(seed),
        boost::random::uniform_01<>()
    ),
    S_tables(),
    dSdT_tables(),
    S_temp(num_energy_groups, Vector(num_energy_groups, machine_limits::signaling_NaN)),
    n_eq(num_energy_groups, machine_limits::signaling_NaN),
    B(num_energy_groups, machine_limits::signaling_NaN) {

    printf("Generating a ComptonMatrixMC object... seed=%d\n", seed);
    if (num_energy_groups + 1 != energy_groups_boundries.size()) {
        printf("ComptonMatrixMC fatal - inconsistent number of energy group boundaries and centers\n");
        exit(1);
    }


    if (std::any_of(
        energy_groups_boundries.begin(), energy_groups_boundries.end(),
        [](double const e) { return e < 0.0; }
    )) {
        printf("ComptonMatrixMC fatal - negative energy group boundaries");
        exit(1);
    }

    for (std::size_t g=0; g<num_energy_groups; ++g) {
        if (energy_groups_boundries[g] >= energy_groups_boundries[g+1]) {
            printf("ComptonMatrixMC fatal - energy group boundaries are not strictly increasing\n");
            exit(1);
        }

        if (energy_groups_boundries[g] >= energy_groups_centers[g] or energy_groups_boundries[g+1] <= energy_groups_centers[g]) {
            printf("ComptonMatrixMC fatal - energy group center is not enclosed in its corresponding energy group boundaries\n");
            exit(1);
        }
    }

    printf("Compton matrices defined on %ld groups.\nPhoton energy group boundaries (in KeV) \n", num_energy_groups);
    for (auto const e : energy_groups_boundries) {
        printf("%g KeV, ", e/units::kev);
    }
    printf("\n");

    set_tables(compton_temperatures);
}

double ComptonMatrixMC::sample_gamma(double const temperature) const {

    double const theta = units::k_boltz * temperature / units::me_c2;
    double const sum_1_bt = 1.0 + 1.0 / theta;
    double const Sb = sum_1_bt + 0.5/(theta*theta);

    double const r0Sb = sample_uniform_01()*Sb;

    double const r1 = sample_uniform_01();

    if (r0Sb <= 1.0) {
        double const r2 = sample_uniform_01();
        double const r3 = sample_uniform_01();

        return 1.0 - theta*std::log(r1*r2*r3);
    }

    if (r0Sb <= sum_1_bt) {
        double const r2 = sample_uniform_01();

        return 1.0 - theta*std::log(r1*r2);
    }

    return 1.0 - theta*std::log(r1);
}

void ComptonMatrixMC::calculate_Bg_ng(double const temperature) const {
    using boost::math::pow;
    double constexpr fac = pow<3>(units::clight) / (8.0*M_PI*units::planck_constant);
    for (std::size_t g=0; g < num_energy_groups; ++g) {
        double const Bg = planck_integral::planck_energy_density_group_integral(energy_groups_boundries[g], energy_groups_boundries[g+1], temperature);

        double const nu = energy_groups_centers[g] / units::planck_constant;
        double const dnu = (energy_groups_boundries[g+1] - energy_groups_boundries[g])/units::planck_constant;

        n_eq[g] = fac*Bg/(pow<3>(nu)*dnu);
        B[g] = Bg;
    }
}

Matrix ComptonMatrixMC::calculate_S_matrix(double const temperature) {

    for (std::size_t i=0; i < num_energy_groups; ++i) {
        for (std::size_t j=0; j < num_energy_groups; ++j) {
            S_temp[i][j] = 0.0;
        }
    }

    std::vector<double> const Omega_0{ 0.0, 0.0, 1.0 }; // Photon direction before scattering in the lab frame, assume to be along the z-axis
    std::vector<double> Omega_0_tag(3., 0.);          // Photon direction before scattering in the electron rest frame
    std::vector<double> Omega_e(3., 0.);              // Electron direction in the lab frame
    std::vector<double> Omega_s_tag(3., 0.);          // Scattering angle in the electron rest frame
    std::vector<double> Omega_tag(3., 0.);            // Photon direction after scattering in the electron rest frame

    std::vector<double> weight(num_energy_groups, 0.);// Monte Carlo weight of samples for each group
    double sum_beta = 0.0;                            // total weight of electron velocity sample 

    for (std::size_t sample_i=0; sample_i < num_of_samples; ++sample_i) {

        if ((sample_i+1) % (num_of_samples/4) == 0 or sample_i==0) {
            printf("Compton matrix T=%gkev sample %ld/%ld [%d%%]\n", temperature/units::kev_kelvin, sample_i+1, num_of_samples, int(100*double(sample_i+1.)/double(num_of_samples)));
        }
        // Before the scattering the photon is assumed to be moving on the z-axis

        // step 1: sample electron velocity from the `gamma^2*exp(-gamma/kT)` distribution 
        double const gamma = sample_gamma(temperature);

        // to get the Maxwell Jutter distribution we weight the sample by `beta`
        double const beta = std::sqrt(1.0 - 1.0 / (gamma*gamma));
        sum_beta += beta;

        // step 2: sample electorn direction in the lab frame `mu_e`
        double const mu_e = 1.0 - 2.0*sample_uniform_01();
        Omega_e[0] = std::sqrt(1. - mu_e*mu_e);
        Omega_e[2] = mu_e;

        // step 3: Define the boost factor to the electron rest frame 
        double const D0 = gamma * (1.0 - beta*mu_e);

        // step 4: apply the boost to the photon direction (before the scattering), In the elector rest frame of the electron the photon moves in the omega_0_tag direction
        double const mu_0_tag = 1. / D0 * (1. - gamma/(1.+gamma)*(D0+1.)*beta*mu_e);
        double const sin_0_tag = std::sqrt(1. - mu_0_tag*mu_0_tag);
        Omega_0_tag[0] = -sin_0_tag;
        Omega_0_tag[2] = mu_0_tag;

        // step 5: sample the scattering angle of the photon
        double const mu_s_tag = 1.0 - 2.0 * sample_uniform_01();
        double const sin_s_tag = std::sqrt(1.0 - mu_s_tag * mu_s_tag);
        double const psi_s_tag = sample_uniform_01() * 2. * M_PI;
        Omega_s_tag[0] = sin_s_tag * std::cos(psi_s_tag);
        Omega_s_tag[1] = sin_s_tag * std::sin(psi_s_tag);
        Omega_s_tag[2] = mu_s_tag;

        // step 6 : rotate Omega_p by -theta_0 to get the scattered photon direction in the electorn frame
        Omega_tag[0] = Omega_0_tag[2] * Omega_s_tag[0] + Omega_0_tag[0] * Omega_s_tag[2];
        Omega_tag[1] = Omega_s_tag[1];
        Omega_tag[2] = -Omega_0_tag[0] * Omega_s_tag[0] + Omega_0_tag[2] * Omega_s_tag[2];

        // step 7 : boost factor to the lab frame
        double const D_tag = gamma*(1. + beta*(Omega_tag[0]*Omega_e[0] + Omega_tag[2]*Omega_e[2]));

        // step 8: sample the energy groups 
        double const interp = sample_uniform_01();
        for (std::size_t g0=0; g0<num_energy_groups; ++g0) {
            // step 8a: sample energy
            double const boundry_g0 = energy_groups_boundries[g0];
            double width = energy_groups_boundries[g0+1]-boundry_g0;

            double const E0 = boundry_g0 + interp*width; // photon energy before scattering in the lab frame

            // weight of energy sample, sample is weighted using the Wein Distribution
            double const a = (E0-boundry_g0)/(units::k_boltz*temperature);
            double const w_E0 = (E0*E0)/(boundry_g0*boundry_g0)*std::exp(-a);

            weight[g0] += w_E0;

            // step 8b: calculate E
            double const E0_tag = D0*E0; // energy of photon before scattering in the electron rest frame
            double const A = 1. / (1. + (1. - mu_s_tag)*E0_tag / units::me_c2);

            double const E_tag = A*E0_tag; // energy of photon after scattering in the electron rest frame
            double const E = D_tag*E_tag; // energy of photon after scattering in the lab frame

            // step 8c: find the out energy group
            auto g_iterator = std::lower_bound(energy_groups_boundries.begin(), energy_groups_boundries.end(), E);
            auto g = std::distance(energy_groups_boundries.begin(), g_iterator)-1; // gives the index of the energy group

            g = std::max(0L, g); // if the out energy is less then the first boundary we add to the first group
            g = std::min(static_cast<long>(energy_groups_centers.size())-1, g); // if the out energy is greater then the last boundary we add to the last group

            // step 8d: calcualte the cross section contribution
            double const sigma = 0.75 * D0/gamma * A*A*(A + 1./A - sin_s_tag*sin_s_tag)*w_E0*beta;

            if (g0 == static_cast<std::size_t>(g)) {
                S_temp[g0][g] += sigma;
            } else {
                // make sure the energy change due to compton is the average energy change calculated using the Monte-Carlo integration, add the change to the in g0->g0 cross section
                double const fac = (E-E0)/(energy_groups_centers[g]-energy_groups_centers[g0]);
                
                S_temp[g0][g] += sigma*fac;
                S_temp[g0][g0] += sigma*(1.0-fac);
            }
        }
    }

    // total weight
    double const beta_avg = sum_beta / num_of_samples;

    // multiply by sigma_thomson and normalization factors
    for (std::size_t g0=0; g0 < num_energy_groups; ++g0) {
        for (std::size_t g=0; g < num_energy_groups; ++g) {
            double const weight_avg = weight[g0]/num_of_samples;
            S_temp[g0][g] *= units::sigma_thomson/(num_of_samples*beta_avg*weight_avg);
        }
    }

    enforce_detailed_balance(temperature, S_temp);

    return S_temp;
}

void ComptonMatrixMC::set_tables(std::vector<double> const& temperature_grid) {

    if (temperature_grid.size()<2) {
        printf("Compton temperature grid has less than two temperature points - %ld\n", temperature_grid.size());
        exit(1);
    }
    printf("Setting Compton matrix tables for %ld temperatures (in KeV):\n", temperature_grid.size());
    for (std::size_t i=0; i<temperature_grid.size(); ++i) {
        printf("%g KeV, ", temperature_grid[i]/units::kev_kelvin);
        if (i>0 and temperature_grid[i]<=temperature_grid[i-1]) {
            printf("fatal - Compton temperature grid is not monotonic\n");
            exit(1);
        }
    }
    printf("\n");

    S_tables = std::vector<Matrix>(temperature_grid.size(), Matrix(num_energy_groups, Vector(num_energy_groups, 0.0)));
    dSdT_tables = std::vector<Matrix>(temperature_grid.size(), Matrix(num_energy_groups, Vector(num_energy_groups, 0.0)));

    for (std::size_t i=0; i < temperature_grid.size(); ++i) {
        S_tables[i] = calculate_S_matrix(temperature_grid[i]);
    }

    for (std::size_t i=0; i+1 < temperature_grid.size(); ++i) {
        using boost::math::pow;

        double const T1 = temperature_grid[i];
        double const T2 = temperature_grid[i+1];

        double const dT = T2 - T1;

        for (std::size_t g=0; g < num_energy_groups; ++g) {
            for (std::size_t gt=0; gt<num_energy_groups; ++gt) {
                dSdT_tables[i][g][gt] = (S_tables[i+1][g][gt] - S_tables[i][g][gt])/dT;
            }
        }
    }
}

void ComptonMatrixMC::get_tau_matrix(double const temperature, double const density, double const A, double const Z, Matrix& tau) const {
    auto const tmp_iterator = std::lower_bound(compton_temperatures.cbegin(), compton_temperatures.cend(), temperature);
    auto const tmp_i = std::distance(compton_temperatures.cbegin(), tmp_iterator) - 1; //  gives the index of lower bound of the temperature in the temperature grid

    if (tmp_i+1 == static_cast<int>(compton_temperatures.size())) {
        printf("temperature T=%gkev given to get_tau_matrix is too high (maximal table temperature=%gkev)\n", temperature/units::kev_kelvin, compton_temperatures.back()/units::kev_kelvin);
        exit(1);
    }

    if (tmp_i == -1) {
        printf("temperature T=%gkev given to get_tau_matrix is too low (minimal table temperature=%gkev)\n", temperature/units::kev_kelvin, compton_temperatures[0]/units::kev_kelvin);
        exit(1);
    }

    if (tau.size() != num_energy_groups) {
        std::cout << "tau given to get_tau_matrix has less then `num_energy_groups` rows" << std::endl;
        exit(1);
    }

    for (std::size_t g=0; g < num_energy_groups; ++g) {
        if (tau[g].size() != num_energy_groups) {
            std::cout << "tau given to get_tau_matrix has less then `num_energy_groups` columns" << std::endl;
            exit(1);
        }
    }

    double const x = (temperature-compton_temperatures[tmp_i])/(compton_temperatures[tmp_i+1]-compton_temperatures[tmp_i]);
    for (std::size_t i = 0; i < num_energy_groups; ++i) {
        for (std::size_t j=0; j < num_energy_groups; ++j) {
            tau[i][j] = S_tables[tmp_i][i][j]*(1. - x) + S_tables[tmp_i+1][i][j]*x;
        }
    }

    double const Nelectron = density*units::Navogadro/A*Z;
    for (std::size_t i=0; i<num_energy_groups; ++i) {
        for (std::size_t j=0; j<num_energy_groups; ++j) {
            tau[i][j] *= Nelectron;
        }
    }

    enforce_detailed_balance(temperature, tau);
}

Matrix ComptonMatrixMC::get_tau_matrix(double const temperature, double const density, double const A, double const Z) const {
    Matrix tau(num_energy_groups, Vector(num_energy_groups, 0.0));

    get_tau_matrix(temperature, density, A, Z, tau);

    return tau;
}

void ComptonMatrixMC::get_dtau_matrix(double const temperature, double const density, double const A, double const Z, Matrix& dtau_dT) const {
    auto const tmp_iterator = std::lower_bound(compton_temperatures.cbegin(), compton_temperatures.cend(), temperature);
    auto const tmp_i = std::distance(compton_temperatures.cbegin(), tmp_iterator) - 1; //  gives the index of lower bound of the temperature in the temperature grid

    if (tmp_i+1 == static_cast<int>(compton_temperatures.size())) {
        printf("temperature T=%gkev given to get_tau_matrix is too high (maximal table temperature=%gkev)\n", temperature/units::kev_kelvin, compton_temperatures.back()/units::kev_kelvin);
        exit(1);
    }

    if (tmp_i == -1) {
        printf("temperature T=%gkev given to get_tau_matrix is too low (minimal table temperature=%gkev)\n", temperature/units::kev_kelvin, compton_temperatures[0]/units::kev_kelvin);
        exit(1);
    }

    if (dtau_dT.size() != num_energy_groups) {
        std::cout << "dtau_dT given to get_dtau_matrix has less then `num_energy_groups` rows" << std::endl;
        exit(1);
    }

    for (std::size_t g=0; g < num_energy_groups; ++g) {
        if (dtau_dT[g].size() != num_energy_groups) {
            std::cout << "dtau_dT given to get_dtau_matrix has less then `num_energy_groups` columns" << std::endl;
            exit(1);
        }
    }

    double const Nelectron = density*units::Navogadro/A*Z;
    dtau_dT = dSdT_tables[tmp_i];
    for (std::size_t i=0; i<num_energy_groups; ++i) {
        for (std::size_t j=0; j<num_energy_groups; ++j) {
            dtau_dT[i][j] *= Nelectron;
        }
    }

    enforce_detailed_balance(temperature, dtau_dT);
}

Matrix ComptonMatrixMC::get_dtau_matrix(double const temperature, double const density, double const A, double const Z) const {
    Matrix dtau(num_energy_groups, Vector(num_energy_groups, 0.0));

    get_dtau_matrix(temperature, density, A, Z, dtau);

    return dtau;
}

void ComptonMatrixMC::enforce_detailed_balance(double const temperature, Matrix& mat) const {
    calculate_Bg_ng(temperature);
    for (std::size_t g=0; g<num_energy_groups; ++g) {
        double const E_g = energy_groups_centers[g];

        for (std::size_t gt=0; gt < g; ++gt) { // notice it is gt < *g*

            if (B[gt] < machine_limits::min_double) {
                mat[g][gt] = 0.0;
                mat[gt][g] = 0.0;
            } else {
                double const E_gt = energy_groups_centers[gt];
                double const detailed_balance_factor = (1.0+n_eq[gt])*B[g]*E_gt / ((1.0+n_eq[g])*B[gt]*E_g);

                if (detailed_balance_factor < 1.0) {
                    mat[gt][g] = mat[g][gt]*detailed_balance_factor;
                } else {
                    mat[g][gt] = mat[gt][g]/detailed_balance_factor;
                }
            }
        }
    }
}

} // namespace compton_matrix_mc