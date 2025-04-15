#include <limits>
#include <ctime>
#include <cmath>
#include <algorithm>

#include "compton_matrix_mc.hpp"
#include "units/units.hpp"
#include "planck_integral/planck_integral.hpp"

#include <boost/math/special_functions/pow.hpp>

#ifdef RICH_MPI
#include <mpi.h>
#endif

namespace compton_matrix_mc {

namespace machine_limits {
static double constexpr signaling_NaN = std::numeric_limits<double>::signaling_NaN();
static double constexpr min_double = 1024. * std::numeric_limits<double>::min();
}

#ifdef RICH_MPI
namespace
{
/**
* @brief Reduces a matrix sum across MPI processes.
*
* This function reduces a matrix sum across MPI processes. It takes a matrix `S` and performs an all-reduce sum operation on each row of the matrix.
*
* @param[in,out] S The matrix to be reduced. The matrix is modified in-place.
* @note There is an implicit assumption that the matrix is a square matrix.
*/
void ReduceMatrixSum(Matrix& S) {
    std::size_t num_energy_groups = S.size();

    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    std::vector<double> send_vector(num_energy_groups * num_energy_groups, 0);

    for (std::size_t g0 = 0; g0 < num_energy_groups; ++g0) {
        for (std::size_t g = 0; g < num_energy_groups; ++g) {
            send_vector[g0 * num_energy_groups + g] = S[g0][g];
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, send_vector.data(), num_energy_groups * num_energy_groups, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for (std::size_t g0 = 0; g0 < num_energy_groups; ++g0) {
        for (std::size_t g = 0; g < num_energy_groups; ++g) {
            S[g0][g] = send_vector[g0 * num_energy_groups + g] / world_size;
        }
    }
}
}
#endif

ComptonMatrixMC::ComptonMatrixMC(Vector const compton_temperatures_,
    Vector const energy_groups_centers_,
    Vector const energy_groups_boundries_,
    std::size_t const num_of_samples_,
    std::optional<unsigned int> const seed_) :

    compton_temperatures(compton_temperatures_),
    energy_groups_centers(energy_groups_centers_),
    energy_groups_boundries(energy_groups_boundries_),
    energy_groups_width(energy_groups_centers_.size(), 0.0),
    num_energy_groups(energy_groups_centers.size()),
    num_of_samples(num_of_samples_),
    seed(seed_ ? seed_.value() : static_cast<unsigned int>(std::time(0))),
    sample_uniform_01(
        boost::random::mt19937_64(seed),
        boost::random::uniform_01<>()
    ),
    S_tables(),
    dSdT_tables(),
    up_scattering_last_table(),
    down_scattering_last_table(),
    n_eq(num_energy_groups, machine_limits::signaling_NaN),
    B(num_energy_groups, machine_limits::signaling_NaN),
    up_scattering_last(machine_limits::signaling_NaN),
    down_scattering_last(machine_limits::signaling_NaN) {

    int rank = 0;

#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    sample_uniform_01.engine().seed(seed + rank); // set different seeds for each MPI process
    if (rank == 0) printf("ComptonMatrixMC initialized with MPI. Seed=%d\n", seed);
#else
    printf("ComptonMatrixMC initialized without MPI. Seed=%d\n", seed);
#endif

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

    for (std::size_t g = 0; g < num_energy_groups; ++g) {
        if (energy_groups_boundries[g] >= energy_groups_boundries[g + 1]) {
            printf("ComptonMatrixMC fatal - energy group boundaries are not strictly increasing\n");
            exit(1);
        }

        if (energy_groups_boundries[g] >= energy_groups_centers[g] or energy_groups_boundries[g + 1] <= energy_groups_centers[g]) {
            printf("ComptonMatrixMC fatal - energy group center is not enclosed in its corresponding energy group boundaries\n");
            exit(1);
        }
    }

    if (rank == 0) {
        printf("Compton matrices defined on %ld groups.\nPhoton energy group boundaries (in KeV) \n", num_energy_groups);
        for (auto const e : energy_groups_boundries) {
            printf("%g KeV, ", e / units::kev);
        }
        printf("\n");
    }

    for (std::size_t g = 0; g < num_energy_groups; ++g) {
        energy_groups_width[g] = energy_groups_boundries[g + 1] - energy_groups_boundries[g];
    }

    set_tables(compton_temperatures);
}

double ComptonMatrixMC::sample_gamma(double const temperature) {

    double const theta = units::k_boltz * temperature / units::me_c2;
    double const sum_1_bt = 1.0 + 1.0 / theta;
    double const Sb = sum_1_bt + 0.5 / (theta * theta);

    double const r0Sb = sample_uniform_01() * Sb;

    double const r1 = sample_uniform_01();

    if (r0Sb <= 1.0) {
        double const r2 = sample_uniform_01();
        double const r3 = sample_uniform_01();

        return 1.0 - theta * std::log(r1 * r2 * r3);
    }

    if (r0Sb <= sum_1_bt) {
        double const r2 = sample_uniform_01();

        return 1.0 - theta * std::log(r1 * r2);
    }

    return 1.0 - theta * std::log(r1);
}

void ComptonMatrixMC::calculate_Bg_ng(double const temperature) {
    using boost::math::pow;
    double constexpr fac = pow<3>(units::clight) / (8.0 * M_PI * units::planck_constant);
    for (std::size_t g = 0; g < num_energy_groups; ++g) {
        double const Bg = planck_integral::planck_energy_density_group_integral(energy_groups_boundries[g], energy_groups_boundries[g + 1], temperature);

        double const nu = energy_groups_centers[g] / units::planck_constant;
        double const dnu = (energy_groups_boundries[g + 1] - energy_groups_boundries[g]) / units::planck_constant;

        n_eq[g] = fac * Bg / (pow<3>(nu) * dnu);
        B[g] = Bg;
    }
}

void ComptonMatrixMC::calculate_S_matrix(double const temperature, Matrix& S) {

    int rank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    for (std::size_t i = 0; i < num_energy_groups; ++i) {
        for (std::size_t j = 0; j < num_energy_groups; ++j) {
            S[i][j] = 0.0;
        }
    }

    up_scattering_last = 0.0;
    down_scattering_last = 0.0;

    std::vector<double> const Omega_0{0.0, 0.0, 1.0}; // Photon direction before scattering in the lab frame, assume to be along the z-axis
    std::vector<double> Omega_0_tag(3., 0.);          // Photon direction before scattering in the electron rest frame
    std::vector<double> Omega_e(3., 0.);              // Electron direction in the lab frame
    std::vector<double> Omega_s_tag(3., 0.);          // Scattering angle in the electron rest frame
    std::vector<double> Omega_tag(3., 0.);            // Photon direction after scattering in the electron rest frame
    
    std::vector<double> weight(num_energy_groups, 0.);// Monte Carlo weight of samples for each group
    double sum_beta = 0.0;                            // total weight of electron velocity sample 

    for (std::size_t sample_i = 0; sample_i < num_of_samples; ++sample_i) {

        if (rank == 0) {
            if ((sample_i + 1) % (num_of_samples / 4) == 0 or sample_i == 0) {
                printf("Compton matrix T=%gkev sample %ld/%ld [%d%%]\n", temperature / units::kev_kelvin, sample_i + 1, num_of_samples, int(100 * double(sample_i + 1.) / double(num_of_samples)));
            }
        }

        // Before the scattering the photon is assumed to be moving on the z-axis

        // step 1: sample electron velocity from the `gamma^2*exp(-gamma/kT)` distribution
        double const gamma = sample_gamma(temperature);

        // to get the Maxwell Jutter distribution we weight the sample by `beta`
        double const beta = std::sqrt(1.0 - 1.0 / (gamma * gamma));
        sum_beta += beta;

        // step 2: sample electorn direction in the lab frame `mu_e`
        double const mu_e = 1.0 - 2.0 * sample_uniform_01();
        Omega_e[0] = std::sqrt(1. - mu_e * mu_e);
        Omega_e[2] = mu_e;

        // step 3: Define the boost factor to the electron rest frame 
        double const D0 = gamma * (1.0 - beta * mu_e);

        // step 4: apply the boost to the photon direction (before the scattering), In the elector rest frame of the electron the photon moves in the omega_0_tag direction
        double const mu_0_tag = 1. / D0 * (1. - gamma / (1. + gamma) * (D0 + 1.) * beta * mu_e);
        double const sin_0_tag = std::sqrt(1. - mu_0_tag * mu_0_tag);
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
        double const D_tag = gamma * (1. + beta * (Omega_tag[0] * Omega_e[0] + Omega_tag[2] * Omega_e[2]));

        // step 8: sample the energy groups 
        double const interp = sample_uniform_01();
        for (std::size_t g0 = 0; g0 < num_energy_groups; ++g0) {
            // step 8a: sample energy
            double const boundry_g0 = energy_groups_boundries[g0];
            double width = std::min(energy_groups_boundries[g0 + 1] - boundry_g0, 30 * units::k_boltz * temperature);

            double const E0 = boundry_g0 + interp * width; // photon energy before scattering in the lab frame
            
            // weight of energy sample, sample is weighted using the Wein Distribution
            double const a = (E0 - boundry_g0) / (units::k_boltz * temperature);
            double const w_E0 = (E0 * E0) / (boundry_g0 * boundry_g0) * std::exp(-a);

            weight[g0] += w_E0;

            // step 8b: calculate E
            double const E0_tag = D0 * E0; // energy of photon before scattering in the electron rest frame
            double const A = 1. / (1. + (1. - mu_s_tag) * E0_tag / units::me_c2);

            double const E_tag = A * E0_tag; // energy of photon after scattering in the electron rest frame
            double const E = D_tag * E_tag;  // energy of photon after scattering in the lab frame

            // step 8c: find the out energy group
            auto g_iterator = std::lower_bound(energy_groups_boundries.begin(), energy_groups_boundries.end(), E);
            auto g = std::distance(energy_groups_boundries.begin(), g_iterator) - 1; // gives the index of the energy group

            g = std::max(0L, g); // if the out energy is less then the first boundary we add to the first group
            g = std::min(static_cast<long>(energy_groups_centers.size()) - 1, g); // if the out energy is greater then the last boundary we add to the last group

            // step 8d: calcualte the cross section contribution
            double const sigma = 0.75 * D0 / gamma * A * A * (A + 1. / A - sin_s_tag * sin_s_tag) * w_E0 * beta;

            if (g0 == static_cast<std::size_t>(g)) {
                S[g0][g] += sigma;

                if (g0 + 1 == num_energy_groups) {
                    // for the last group, we explicitly calculate the change in energy due to up and down scattering and incorporate it inside the group cross section (similarly to what we do between different groups)
                    if (E0 < E) {
                        up_scattering_last += (E - E0) / energy_groups_centers[g0] * sigma;
                    }
                    else {
                        down_scattering_last += (E0 - E) / energy_groups_centers[g0] * sigma;
                    }
                }
            }
            else {
                // make sure the energy change due to compton is the average energy change calculated using the Monte-Carlo integration, add the change to the in g0->g0 cross section
                double const fac = (E - E0) / (energy_groups_centers[g] - energy_groups_centers[g0]);
                
                S[g0][g] += sigma * fac;
                S[g0][g0] += sigma * (1.0 - fac);
            }
        }
    }

#ifdef RICH_MPI
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ReduceMatrixSum(S);

    MPI_Allreduce(MPI_IN_PLACE, &sum_beta, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &up_scattering_last, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &down_scattering_last, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    sum_beta /= world_size;
    up_scattering_last /= world_size;
    down_scattering_last /= world_size;

    for (std::size_t g0 = 0; g0 < num_energy_groups; ++g0) {
        weight[g0] /= world_size;
    }

    MPI_Allreduce(MPI_IN_PLACE, weight.data(), num_energy_groups, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

    // total weight
    double const beta_avg = sum_beta / num_of_samples;

    // multiply by sigma_thomson and normalization factors
    for (std::size_t g0 = 0; g0 < num_energy_groups; ++g0) {
        double const weight_avg = weight[g0] / num_of_samples;
        for (std::size_t g = 0; g < num_energy_groups; ++g) {
            if (weight_avg < std::numeric_limits<double>::min() * 1e40)
            {
                S[g0][g] = std::numeric_limits<double>::min() * 1e40;
                continue;
            }

            S[g0][g] *= units::sigma_thomson / (num_of_samples * beta_avg * weight_avg);
            S[g0][g] = std::max(S[g0][g], std::numeric_limits<double>::min() * 1e40);
        }

        if (g0 + 1 == num_energy_groups) {
            up_scattering_last *= units::sigma_thomson / (num_of_samples * beta_avg * weight_avg);
            down_scattering_last *= units::sigma_thomson / (num_of_samples * beta_avg * weight_avg);
        }
    }

    enforce_detailed_balance(temperature, S);
}

void ComptonMatrixMC::set_tables(std::vector<double> const& temperature_grid) {

    if (temperature_grid.size() < 2) {
        printf("Compton temperature grid has less than two temperature points - %ld\n", temperature_grid.size());
        exit(1);
    }

    int rank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif   

    if (rank == 0) printf("Setting Compton matrix tables for %ld temperatures (in KeV):\n", temperature_grid.size());

    for (std::size_t i = 0; i < temperature_grid.size(); ++i) {
        if (rank == 0)
            printf("%g KeV, ", temperature_grid[i] / units::kev_kelvin);
        if (i > 0 and temperature_grid[i] <= temperature_grid[i - 1]) {
            printf("fatal - Compton temperature grid is not monotonic\n");
            exit(1);
        }
    }

    if (rank == 0) printf("\n");

    S_tables = std::vector<Matrix>(temperature_grid.size(), Matrix(num_energy_groups, Vector(num_energy_groups, 0.0)));
    dSdT_tables = std::vector<Matrix>(temperature_grid.size(), Matrix(num_energy_groups, Vector(num_energy_groups, 0.0)));

    up_scattering_last_table = std::vector<double>(temperature_grid.size(), 0.0);
    down_scattering_last_table = std::vector<double>(temperature_grid.size(), 0.0);

    for (std::size_t i = 0; i < temperature_grid.size(); ++i) {
        calculate_S_matrix(temperature_grid[i], S_tables[i]);
        for (std::size_t g0 = 0; g0 < num_energy_groups; ++g0) {
            if (g0 + 1 == num_energy_groups) {
                up_scattering_last_table[i] = up_scattering_last;
                down_scattering_last_table[i] = down_scattering_last;
            }
        }
    }

    for (std::size_t i = 0; i < temperature_grid.size(); ++i) {
        size_t lower = i > 0 ? i - 1 : 0;
        size_t upper = i < temperature_grid.size() - 1 ? i + 1 : temperature_grid.size() - 1;

        // to calculate dSdT numerically at `T` we find the limits around the point s.t. T in [T_lower, T_upper] and T_upper - T_lower > max(1e5, temperature_grid[i] * 0.2)
        while ((temperature_grid[upper] - temperature_grid[lower]) < std::max(1e5, temperature_grid[i] * 0.2))
        {
            if (lower > 0) --lower;
            if (upper < (temperature_grid.size() - 1)) ++upper;

            if (lower == 0 && upper == (temperature_grid.size() - 1)) break;
        }

        double const dT = temperature_grid[upper] - temperature_grid[lower];

        for (std::size_t g0 = 0; g0 < num_energy_groups; ++g0) {
            for (std::size_t g = 0; g < num_energy_groups; ++g) {
                dSdT_tables[i][g0][g] = (S_tables[upper][g0][g] - S_tables[lower][g0][g]) / dT;
            }

            if (g0 + 1 == num_energy_groups) {
                double const sigma_upper = up_scattering_last_table[upper] - down_scattering_last_table[upper];
                double const sigma_lower = up_scattering_last_table[lower] - down_scattering_last_table[lower];
                double const sigma = up_scattering_last_table[i] - down_scattering_last_table[i];

                /*Since for the 'last group -> last group' we account for the up/down scattering difference we can have "negative" cross section (i.e energy is lost to the material because of scattering inside the group itself) this condition ensures that the sign is the same across temperatures and that we have no sign change because of sampling accuracy */
                if ((sigma < 0 && sigma_lower < 0 && sigma_upper < 0) || (sigma > 0 && sigma_lower > 0 && sigma_upper > 0)) {
                    dSdT_tables[i][g0][g0] = (sigma_upper - sigma_lower) / dT;
                }
                else {
                    dSdT_tables[i][g0][g0] = 0;
                }
            }
        }
    }

    if (rank == 0) std::cout << "Done compton matrix tables" << std::endl;
}

void ComptonMatrixMC::get_tau_matrix(double const temperature, double const density, double const A, double const Z, Matrix& tau) {
    auto const tmp_iterator = std::lower_bound(compton_temperatures.cbegin(), compton_temperatures.cend(), temperature);
    auto const tmp_i = std::distance(compton_temperatures.cbegin(), tmp_iterator) - 1; //  gives the index of lower bound of the temperature in the temperature grid

    if (tmp_i + 1 == static_cast<int>(compton_temperatures.size())) {
        printf("temperature T=%gkev given to get_tau_matrix is too high (maximal table temperature=%gkev)\n", temperature / units::kev_kelvin, compton_temperatures.back() / units::kev_kelvin);
        exit(1);
    }

    if (tmp_i == -1) {
        printf("temperature T=%gkev given to get_tau_matrix is too low (minimal table temperature=%gkev)\n", temperature / units::kev_kelvin, compton_temperatures[0] / units::kev_kelvin);
        exit(1);
    }

    if (tau.size() != num_energy_groups) {
        std::cout << "tau given to get_tau_matrix has less then `num_energy_groups` rows" << std::endl;
        exit(1);
    }

    for (std::size_t g = 0; g < num_energy_groups; ++g) {
        if (tau[g].size() != num_energy_groups) {
            std::cout << "tau given to get_tau_matrix has less then `num_energy_groups` columns" << std::endl;
            exit(1);
        }
    }

    double const x = (temperature - compton_temperatures[tmp_i]) / (compton_temperatures[tmp_i + 1] - compton_temperatures[tmp_i]);
    for (std::size_t i = 0; i < num_energy_groups; ++i) {
        for (std::size_t j = 0; j < num_energy_groups; ++j) {
            tau[i][j] = S_tables[tmp_i][i][j] * (1. - x) + S_tables[tmp_i + 1][i][j] * x;
        }
    }

    double const Nelectron = density * units::Navogadro / A * Z;
    for (std::size_t i = 0; i < num_energy_groups; ++i) {
        for (std::size_t j = 0; j < num_energy_groups; ++j) {
            tau[i][j] *= Nelectron;
        }
    }

    enforce_detailed_balance(temperature, tau);
}

void ComptonMatrixMC::get_dtau_matrix(double const temperature, double const density, double const A, double const Z, Matrix& dtau_dT) {
    auto const tmp_iterator = std::lower_bound(compton_temperatures.cbegin(), compton_temperatures.cend(), temperature);
    auto const tmp_i = std::distance(compton_temperatures.cbegin(), tmp_iterator) - 1; //  gives the index of lower bound of the temperature in the temperature grid

    if (tmp_i + 1 == static_cast<int>(compton_temperatures.size())) {
        printf("temperature T=%gkev given to get_dtau_matrix is too high (maximal table temperature=%gkev)\n", temperature / units::kev_kelvin, compton_temperatures.back() / units::kev_kelvin);
        exit(1);
    }

    if (tmp_i == -1) {
        printf("temperature T=%gkev given to get_dtau_matrix is too low (minimal table temperature=%gkev)\n", temperature / units::kev_kelvin, compton_temperatures[0] / units::kev_kelvin);
        exit(1);
    }

    if (dtau_dT.size() != num_energy_groups) {
        std::cout << " dtau_dT given to get_dtau_matrix has less then `num_energy_groups` rows" << std::endl;
        exit(1);
    }

    for (std::size_t g = 0; g < num_energy_groups; ++g) {
        if (dtau_dT[g].size() != num_energy_groups) {
            std::cout << "dtau_dT given to get_dtau_matrix has less then `num_energy_groups` columns" << std::endl;
            exit(1);
        }
    }

    double const x = (temperature - compton_temperatures[tmp_i]) / (compton_temperatures[tmp_i + 1] - compton_temperatures[tmp_i]);
    for (std::size_t i = 0; i < num_energy_groups; ++i) {
        for (std::size_t j = 0; j < num_energy_groups; ++j) {
            dtau_dT[i][j] = dSdT_tables[tmp_i][i][j] * (1. - x) + dSdT_tables[tmp_i + 1][i][j] * x;
        }
    }

    double const Nelectron = density * units::Navogadro / A * Z;
    for (std::size_t i = 0; i < num_energy_groups; ++i) {
        for (std::size_t j = 0; j < num_energy_groups; ++j) {
            dtau_dT[i][j] *= Nelectron;
        }
    }

    enforce_detailed_balance(temperature, dtau_dT);
}

Matrix ComptonMatrixMC::get_tau_matrix(double const temperature, double const density, double const A, double const Z) {
    Matrix tau(num_energy_groups, Vector(num_energy_groups, 0.0));

    get_tau_matrix(temperature, density, A, Z, tau);

    return tau;
}

Matrix ComptonMatrixMC::get_S_matrix(double const temperature) {
    Matrix S(num_energy_groups, Vector(num_energy_groups, 0.0));

    calculate_S_matrix(temperature, S);
    return S;
}

std::pair<double, double> ComptonMatrixMC::get_last_group_upscattering_and_downscattering(double const temperature, double const density, double const A, double const Z) {
    auto const tmp_iterator = std::lower_bound(compton_temperatures.cbegin(), compton_temperatures.cend(), temperature);
    auto const tmp_i = std::distance(compton_temperatures.cbegin(), tmp_iterator) - 1; //  gives the index of lower bound of the temperature in the temperature grid

    if (tmp_i + 1 == static_cast<int>(compton_temperatures.size())) {
        printf("temperature T=%gkev given to get_tau_matrix is too high (maximal table temperature=%gkev)\n", temperature / units::kev_kelvin, compton_temperatures.back() / units::kev_kelvin);
        exit(1);
    }

    if (tmp_i == -1) {
        printf("temperature T=%gkev given to get_tau_matrix is too low (minimal table temperature=%gkev)\n", temperature / units::kev_kelvin, compton_temperatures[0] / units::kev_kelvin);
        exit(1);
    }

    double const x = (temperature - compton_temperatures[tmp_i]) / (compton_temperatures[tmp_i + 1] - compton_temperatures[tmp_i]);

    double upscattering_interp = up_scattering_last_table[tmp_i] * (1. - x) + up_scattering_last_table[tmp_i + 1] * x;
    double downscattering_interp = down_scattering_last_table[tmp_i] * (1. - x) + down_scattering_last_table[tmp_i + 1] * x;

    double const Nelectron = density * units::Navogadro / A * Z;

    upscattering_interp *= Nelectron;
    downscattering_interp *= Nelectron;

    return std::pair(upscattering_interp, downscattering_interp);
}

void ComptonMatrixMC::enforce_detailed_balance(double const temperature, Matrix& mat) {
    calculate_Bg_ng(temperature);
    for (std::size_t g = 0; g < num_energy_groups; ++g) {
        double const E_g = energy_groups_centers[g];

        for (std::size_t gt = 0; gt < g; ++gt) { // notice it is gt < *g*

            if (B[gt] < machine_limits::min_double) {
                mat[g][gt] = 0.0;
                mat[gt][g] = 0.0;
            }
            else {
                double const E_gt = energy_groups_centers[gt];
                double const detailed_balance_factor = (1.0 + n_eq[gt]) * B[g] * E_gt / ((1.0 + n_eq[g]) * B[gt] * E_g);

                if (detailed_balance_factor < 1.0) {
                    mat[gt][g] = mat[g][gt] * detailed_balance_factor;
                }
                else {
                    mat[g][gt] = mat[gt][g] / detailed_balance_factor;
                }
            }
        }
    }
}

} // namepsace compton_matrix_mc