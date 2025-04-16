#include <limits>
#include <ctime>
#include <cmath>

#include "compton_matrix_mc.hpp"
#include "units/units.hpp"
#include "planck_integral/planck_integral.hpp"

#include <boost/math/special_functions/pow.hpp>

static double constexpr signaling_NaN = std::numeric_limits<double>::signaling_NaN();

ComptonMatrixMC::ComptonMatrixMC(
    Vector const energy_groups_centers_,
    Vector const energy_groups_boundries_,
    std::size_t const num_of_samples_,
    bool const force_detailed_balance_,
    std::optional<unsigned int> const seed_) :

    energy_groups_centers(energy_groups_centers_),
    energy_groups_boundries(energy_groups_boundries_),
    num_energy_groups(energy_groups_centers.size()),
    num_of_samples(num_of_samples_),
    seed(seed_ ? seed_.value() : static_cast<unsigned int>(std::time(0))),
    sample_uniform_01(
        boost::random::mt19937(seed),
        boost::random::uniform_01<>()
    ),
    force_detailed_balance(force_detailed_balance_),
    temperature_grid(),
    S_log_tables(),
    dSdUm_tables(),
    S_temp(num_energy_groups, Vector(num_energy_groups, signaling_NaN)),
    n_eq(num_energy_groups, signaling_NaN),
    B(num_energy_groups, signaling_NaN) {
    
    printf("Generating a ComptonMatrixMC object... seed=%d\n", seed);
    if (num_energy_groups + 1 != energy_groups_boundries.size()) {
        printf("ComptonMatrixMC fatal - inconsistent number of energy group boundaries and centers\n");
        exit(1);
    }
    
    for (std::size_t g=0; g<num_energy_groups; ++g) {
        if (energy_groups_boundries[g]   < 0. or
            energy_groups_boundries[g]   >= energy_groups_boundries[g+1] or
            energy_groups_boundries[g]   >= energy_groups_centers[g] or
            energy_groups_boundries[g+1] <= energy_groups_centers[g]) {
            printf("ComptonMatrixMC fatal - inconsistent energy group boundaries/centers\n");
            exit(1);
        }
    }

    printf("Compton matrices defined on %ld groups.\nPhoton energy group boundaries (in kev) \n", num_energy_groups);
    for (auto const e : energy_groups_boundries) {
        printf("%g ", e/units::kev);
    }
    printf("\n");
}

double ComptonMatrixMC::sample_gamma(double const temperature) {

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

void ComptonMatrixMC::set_Bg_ng(double const temperature) {
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

    double sum_beta = 0.0;
    std::vector<double> const Omega_0(3., 0.);
    std::vector<double> Omega_0_tag(3., 0.), Omega_e(3., 0.), Omega_tag(3., 0.), Omega_p_tag(3., 0.);
    std::vector<double> weight(num_energy_groups, 0.);
    for (std::size_t sample_i=0; sample_i < num_of_samples; ++sample_i) {

        if ((sample_i+1) % (num_of_samples/4) == 0 or sample_i==0) {
            printf("Compton matrix T=%gkev sample %ld/%ld [%d%%]\n", temperature/units::kev_kelvin, sample_i+1, num_of_samples, int(100*double(sample_i+1.)/double(num_of_samples)));
        }

        // step 1: sample electron velocity from a weighted Maxwell Juttner distribution 
        double const gamma = sample_gamma(temperature);

        // weight of sample
        double const beta = std::sqrt(1.0 - 1.0 / (gamma*gamma));
        sum_beta += beta;

        // step 2: sample mu_e
        double const mu_e = 1.0 - 2.0*sample_uniform_01();
        Omega_e[0] = std::sqrt(1. - mu_e*mu_e);
        Omega_e[2] = mu_e;

        // step 3:
        double const D0 = gamma * (1.0 - beta*mu_e);

        // step 4: the direction of the photon before the scattering in the rest frame of the electron
        double const mu_0_tag = 1. / D0 * (1. - gamma/(1.+gamma)*(D0+1.)*beta*mu_e);
        double const sin_0_tag = std::sqrt(1. - mu_0_tag*mu_0_tag);
        Omega_0_tag[0] = -sin_0_tag;
        Omega_0_tag[2] = mu_0_tag;

        // step 5: sample the scattering angle of the photon
        double const mu_p_tag = 1.0 - 2.0*sample_uniform_01();
        double const sin_p_tag = std::sqrt(1.0 - mu_p_tag*mu_p_tag);
        double const psi_p_tag = sample_uniform_01()*2.*M_PI;
        Omega_p_tag[0] = sin_p_tag * std::cos(psi_p_tag);
        Omega_p_tag[1] = sin_p_tag * std::sin(psi_p_tag);
        Omega_p_tag[2] = mu_p_tag;

        // step 6 : rotate Omega_p by -theta_0 to get the angle in the electorn frame
        Omega_tag[0] = Omega_0_tag[2]*Omega_p_tag[0] + Omega_0_tag[0]*Omega_p_tag[2];
        Omega_tag[1] = Omega_p_tag[1];
        Omega_tag[2] = -Omega_0_tag[0]*Omega_p_tag[0] + Omega_0_tag[2]*Omega_p_tag[2];

        // step 7 
        double const D_tag = gamma*(1. + beta*(Omega_tag[0]*Omega_e[0] + Omega_tag[2]*Omega_e[2]));

        // step 8: sample the energy groups 
        double const interp = sample_uniform_01();
        for (std::size_t g0=0; g0<num_energy_groups; ++g0) {
            // step 8a: sample energy
            double const boundry_g0 = energy_groups_boundries[g0];
            double width = energy_groups_boundries[g0+1]-boundry_g0;

            double const E0 = boundry_g0 + interp*width;
            // weight of energy sample
            double const a = (E0-boundry_g0)/(units::k_boltz*temperature);
            double const w_E0 = (E0*E0)/(boundry_g0*boundry_g0)*std::exp(-a);

            weight[g0] += w_E0;

            // step 8b: calculate E
            double const E0_tag = D0*E0;
            double const A = 1. / (1. + (1. - mu_p_tag)*E0_tag / units::me_c2);
            double const E_tag = A*E0_tag;
            double const E = D_tag*E_tag;

            // step 8c: find the out energy group
            auto g_iterator = std::lower_bound(energy_groups_boundries.begin(), energy_groups_boundries.end(), E);
            auto g = std::distance(energy_groups_boundries.begin(), g_iterator)-1; // gives the index of the energy group

            g = std::max(0L, g);
            g = std::min(static_cast<long>(energy_groups_centers.size())-1, g);

            // step 8d: calcualte the cross section contribution
            double const sigma = 0.75 * D0/gamma * A*A*(A + 1./A - sin_p_tag*sin_p_tag)*w_E0*beta;

            if (g0 == static_cast<std::size_t>(g)) {
                S_temp[g0][g] += sigma;
            } else {
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

    if (force_detailed_balance) {
        set_Bg_ng(temperature);
        double constexpr thresh = units::sigma_thomson*std::numeric_limits<double>::epsilon()*1e3;

        for (std::size_t g=0; g < num_energy_groups; ++g) {
            double const E_g = energy_groups_centers[g];

            for (std::size_t gt=g+1; gt<num_energy_groups; ++gt) {
                if (S_temp[gt][g] < thresh and S_temp[g][gt] < thresh) continue;

                if (B[g]<std::numeric_limits<double>::min()*1e40 or B[gt]<std::numeric_limits<double>::min()*1e40) {
                    S_temp[g][gt] = S_temp[gt][g] = 0.;
                    continue;
                }

                double const E_gt = energy_groups_centers[gt];
                double const detailed_balance_factor = (1.0+n_eq[gt])*B[g]*E_gt / ((1.0+n_eq[g])*B[gt]*E_g);

                if (detailed_balance_factor < 1.0) {
                    S_temp[gt][g] = S_temp[g][gt]*detailed_balance_factor;
                } else {
                    S_temp[g][gt] = S_temp[gt][g]/detailed_balance_factor;
                }
            }
        }
    }

    return S_temp;
}

void ComptonMatrixMC::set_tables(std::vector<double> const& temperature_grid_) {

    if (temperature_grid_.size()<2) {
        printf("Compton temperature grid has less than two temperature points - %ld\n", temperature_grid_.size());
        exit(1);
    }
    printf("setting Compton matrix tables for %ld temperatures (in kev):\n", temperature_grid_.size());
    for (std::size_t i=0; i<temperature_grid_.size(); ++i) {
        printf("%g ", temperature_grid_[i]/units::kev_kelvin);
        if (i>0 and temperature_grid_[i]<=temperature_grid_[i-1]) {
            printf("fatal - Compton temperature grid is not monotonic\n");
            exit(1);
        }
    }
    printf("\n");

    temperature_grid = temperature_grid_;
    S_log_tables = std::vector<Matrix>(temperature_grid.size(), Matrix(num_energy_groups, Vector(num_energy_groups, 0.0)));
    dSdUm_tables = std::vector<Matrix>(temperature_grid.size(), Matrix(num_energy_groups, Vector(num_energy_groups, 0.0)));

    for (std::size_t i=0; i < temperature_grid.size(); ++i) {
        S_log_tables[i] = calculate_S_matrix(temperature_grid[i]);
        for (std::size_t g0=0; g0 < num_energy_groups; ++g0) {
            for (std::size_t g=0; g < num_energy_groups; ++g) {
                S_log_tables[i][g0][g] = std::log(S_log_tables[i][g0][g]);
            }
        }
    }

    for (std::size_t i=0; i+1 < temperature_grid.size(); ++i) {
        using boost::math::pow;

        double const T1 = temperature_grid[i];
        double const T2 = temperature_grid[i+1];

        double const dUm = units::arad*(pow<4>(T2) - pow<4>(T1));

        for (std::size_t g=0; g < num_energy_groups; ++g) {
            for (std::size_t gt=0; gt<num_energy_groups; ++gt) {
                dSdUm_tables[i][g][gt] = (std::exp(S_log_tables[i+1][g][gt]) - std::exp(S_log_tables[i][g][gt]))/dUm;
            }
        }
    }
}

void ComptonMatrixMC::get_tau_matrix(double const temperature, double const density, double const A, double const Z, Matrix& tau, Matrix& dtau_dUm) {
    auto const tmp_iterator = std::lower_bound(temperature_grid.cbegin(), temperature_grid.cend(), temperature);
    auto const tmp_i = std::distance(temperature_grid.cbegin(), tmp_iterator) - 1; //  gives the index of lower bound of the temperature in the temperature grid

    if (tmp_i+1 == static_cast<int>(temperature_grid.size())) {
        printf("temperature T=%gkev given to get_tau_matrix is too high (maximal table temperature=%gkev)\n", temperature/units::kev_kelvin, temperature_grid.back()/units::kev_kelvin);
        exit(1);
    }

    if (tmp_i == -1) {
        printf("temperature T=%gkev given to get_tau_matrix is too low (minimal table temperature=%gkev)\n", temperature/units::kev_kelvin, temperature_grid[0]/units::kev_kelvin);
        exit(1);
    }

    if (tau.size() != num_energy_groups or dtau_dUm.size() != num_energy_groups) {
        std::cout << "tau or dtau_dUm given to get_tau_matrix has less then `num_energy_groups` rows" << std::endl;
        exit(1);
    }

    for (std::size_t g=0; g < num_energy_groups; ++g) {
        if (tau[g].size() != num_energy_groups or dtau_dUm[g].size() != num_energy_groups) {
            std::cout << "tau or dtau_dUm given to get_tau_matrix has less then `num_energy_groups` columns" << std::endl;
            exit(1);
        }
    }

    if (force_detailed_balance) set_Bg_ng(temperature);

    double const x = (temperature-temperature_grid[tmp_i])/(temperature_grid[tmp_i+1]-temperature_grid[tmp_i]);
    for (std::size_t i = 0; i < num_energy_groups; ++i) {
        double const E_i = energy_groups_centers[i];

        for (std::size_t j=i; j < num_energy_groups; ++j) {
            tau[i][j] = std::exp(S_log_tables[tmp_i][i][j])*(1. - x) + std::exp(S_log_tables[tmp_i+1][i][j])*x;

            if (i == j) continue;

            tau[j][i] = std::exp(S_log_tables[tmp_i][j][i])*(1. - x) + std::exp(S_log_tables[tmp_i+1][j][i])*x;

            // enforce detailed balance on the interpolated matrix
            if (force_detailed_balance) {

                if (B[i]<std::numeric_limits<double>::min()*1e40 or B[j]<std::numeric_limits<double>::min()*1e40) {
                    tau[i][j] = tau[j][i] = 0.;
                    continue;
                }

                double const E_j = energy_groups_centers[j];
                double const detailed_balance_factor = (1.0+n_eq[j])*B[i]*E_j / ((1.0+n_eq[i])*B[j]*E_i);

                if (detailed_balance_factor < 1.0) {
                    tau[j][i] = tau[i][j]*detailed_balance_factor;
                } else {
                    tau[i][j] = tau[j][i]/detailed_balance_factor;
                }
            } else {
                tau[j][i] = std::exp(S_log_tables[tmp_i][j][i])*(1. - x) + std::exp(S_log_tables[tmp_i+1][j][i])*x;
            }
        }
    }

    double const Nelectron = density*units::Navogadro/A*Z;
    dtau_dUm = dSdUm_tables[tmp_i];
    for (std::size_t i=0; i<num_energy_groups; ++i) {
        for (std::size_t j=0; j<num_energy_groups; ++j) {
            tau[i][j] *= Nelectron;
            dtau_dUm[i][j] *= Nelectron;
        }
    }
}

Matrix ComptonMatrixMC::get_tau_matrix(double const temperature, double const density, double const A, double const Z) {
    Matrix tau(num_energy_groups, Vector(num_energy_groups, 0.0));
    Matrix dtau(num_energy_groups, Vector(num_energy_groups, 0.0));

    get_tau_matrix(temperature, density, A, Z, tau, dtau);

    return tau;
}