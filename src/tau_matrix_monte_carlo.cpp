#include <limits>
#include <ctime>
#include <cmath>

#include "tau_matrix_monte_carlo.hpp"
#include "units/units.hpp"
#include "planck_integral/planck_integral.hpp"

#include <boost/math/special_functions/pow.hpp>

static double constexpr signaling_NaN = std::numeric_limits<double>::signaling_NaN();

tau_matrix_monte_carlo_engine::tau_matrix_monte_carlo_engine(Vector const energy_groups_center_, 
                                                             Vector const energy_groups_boundries_, 
                                                             std::size_t const num_of_samples_, 
                                                             bool const force_detailed_balance_,
                                                             std::size_t const seed) :
                            energy_groups_center(energy_groups_center_),
                            energy_groups_boundries(energy_groups_boundries_),
                            num_energy_groups(energy_groups_center.size()),
                            S_temp(num_energy_groups, Vector(num_energy_groups, signaling_NaN)),
                            num_of_samples(num_of_samples_), 
                            sample_uniform_01(
                                boost::random::mt19937(static_cast<unsigned int>(seed != 0 ? seed : std::time(0))),
                                boost::random::uniform_01<>()
                            ),
                            force_detailed_balance(force_detailed_balance_),
                            temperature_grid(),
                            S_log_tables(),
                            dSdUm_tables(),
                            n_eq(num_energy_groups, signaling_NaN),
                            B(num_energy_groups, signaling_NaN) {}

double tau_matrix_monte_carlo_engine::sample_gamma(double const temperature){
    
    double const theta = units::k_boltz * temperature / units::me_c2;
    double const sum_1_bt = 1.0 + 1.0 / theta;
    double const Sb = sum_1_bt + 0.5/(theta*theta); 

    double const r0Sb = sample_uniform_01()*Sb;
    
    double const r1 = sample_uniform_01();

    if(r0Sb <= 1.0){
        double const r2 = sample_uniform_01();
        double const r3 = sample_uniform_01();

        return 1.0 - theta*std::log(r1*r2*r3);
    }

    if(r0Sb <= sum_1_bt){
        double const r2 = sample_uniform_01();

        return 1.0 - theta*std::log(r1*r2);
    }

    return 1.0 - theta*std::log(r1);
}

Matrix tau_matrix_monte_carlo_engine::generate_S_matrix(double const temperature, bool const log_grid){

    for(std::size_t i=0; i < num_energy_groups; ++i){
        for(std::size_t j=0; j < num_energy_groups; ++j){
            S_temp[i][j] = 0.0;
        }
    }

    double sum_beta = 0.0;
    std::vector<double> const Omega_0(3., 0.);
    std::vector<double> Omega_0_tag(3., 0.), Omega_e(3., 0.), Omega_tag(3., 0.), Omega_p_tag(3., 0.);
    std::vector<double> weight(num_energy_groups, 0.);
    for(std::size_t sample_i=0; sample_i < num_of_samples; ++sample_i){

        if((sample_i+1) % (num_of_samples/4) == 0 or sample_i==0){
            printf("Compton matrix T=%gkev sample %d/%d [%d %]\n", temperature/units::kev_kelvin, sample_i+1, num_of_samples, int(100*double(sample_i+1.)/double(num_of_samples)));
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
        for(std::size_t g0=0; g0<num_energy_groups; ++g0){
            // step 8a: sample energy
            double const E0 = energy_groups_boundries[g0] + interp*(energy_groups_boundries[g0+1]-energy_groups_boundries[g0]);
            // weight of energy sample
            double const w_E0 = E0*E0*std::exp(-E0/(units::k_boltz*temperature));
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
            g = std::min(static_cast<long>(energy_groups_center.size())-1, g);

            // step 8d: calcualte the cross section contribution
            double const sigma = 0.75 * D0/gamma * A*A*(A + 1./A - sin_p_tag*sin_p_tag)*w_E0*beta;
            
            S_temp[g0][g] += sigma;
        }    
    }

    // total weight
    double const beta_avg = sum_beta / num_of_samples;
    
    // multiply by sigma_thomson and normalization factors
    for(std::size_t g0=0; g0 < num_energy_groups; ++g0){
        for(std::size_t g=0; g < num_energy_groups; ++g){
            double const weight_avg = weight[g0]/num_of_samples;
            if(weight_avg > std::numeric_limits<double>::min()*1e40 and S_temp[g0][g] > std::numeric_limits<double>::min()*1e40){
                S_temp[g0][g] *= units::sigma_thomson/(num_of_samples*beta_avg*weight_avg);
            } else {
                S_temp[g0][g] = 1e-200;
            }
        }
    }

    if(force_detailed_balance) {
        double constexpr thresh = units::sigma_thomson*std::numeric_limits<double>::epsilon()*1e3;

        using boost::math::pow;
        double constexpr fac = pow<3>(units::clight) / (8.0*M_PI*units::planck_constant);
        for(std::size_t g=0; g < num_energy_groups; ++g){
            double const Bg = planck_integral::planck_energy_density_group_integral(energy_groups_boundries[g], energy_groups_boundries[g+1], temperature);

            double const nu = energy_groups_center[g] / units::planck_constant;
            double const dnu = (energy_groups_boundries[g+1] - energy_groups_boundries[g])/units::planck_constant;

            n_eq[g] = fac*Bg/(pow<3>(nu)*dnu);
            B[g] = Bg;
        }

        for(std::size_t g=0; g < num_energy_groups; ++g){
            double const E_g = energy_groups_center[g];
            
            for(std::size_t gt=g+1; gt<num_energy_groups; ++gt){
                if(S_temp[gt][g] < thresh and S_temp[g][gt] < thresh) continue;
                
                double const E_gt = energy_groups_center[gt];

                double const detailed_balance_factor = (1.0+n_eq[gt])*B[g]*E_gt / ((1.0+n_eq[g])*B[gt]*E_g);
                
                if(S_temp[gt][g] < S_temp[g][gt]){
                    S_temp[gt][g] = S_temp[g][gt]*detailed_balance_factor;
                }
                else{
                    S_temp[g][gt] = S_temp[gt][g]/detailed_balance_factor;
                }
            }
        }
    }

    if(log_grid){
        for(std::size_t g0=0; g0 < num_energy_groups; ++g0){
            for(std::size_t g=0; g < num_energy_groups; ++g){
                S_temp[g0][g] = std::log(S_temp[g0][g]);
            }
        }
    }
    return S_temp;
}

void tau_matrix_monte_carlo_engine::generate_tables(std::vector<double> const& tmp_grid){
    temperature_grid = tmp_grid;
    S_log_tables = std::vector<Matrix>(temperature_grid.size(), Matrix(num_energy_groups, Vector(num_energy_groups, 0.0)));
    dSdUm_tables = std::vector<Matrix>(temperature_grid.size(), Matrix(num_energy_groups, Vector(num_energy_groups, 0.0)));
    
    for(std::size_t i=0; i < temperature_grid.size(); ++i){
        S_log_tables[i] = generate_S_matrix(temperature_grid[i], true);
    }

    for(std::size_t i=0; i+1 < temperature_grid.size(); ++i){
        using boost::math::pow;

        double const T1 = temperature_grid[i];
        double const T2 = temperature_grid[i+1];
        
        double const dUm = units::arad*(pow<4>(T2) - pow<4>(T1));

        for(std::size_t g=0; g < num_energy_groups; ++g){
            for(std::size_t gt=0; gt<num_energy_groups; ++gt){
                dSdUm_tables[i][g][gt] = (std::exp(S_log_tables[i+1][g][gt]) - std::exp(S_log_tables[i][g][gt]))/dUm;
            }
        }
    }
}

void tau_matrix_monte_carlo_engine::generate_tau_matrix(double const temperature, double const density, double const A, double const Z, Matrix& tau, Matrix& dtau_dUm){
    auto const tmp_iterator = std::lower_bound(temperature_grid.cbegin(), temperature_grid.cend(), temperature);
    auto const tmp_i = std::distance(temperature_grid.cbegin(), tmp_iterator) - 1; //  gives the index of lower bound of the temperature in the temperature grid

    if(tmp_i+1 == temperature_grid.size()){
        std::cout << "temperature given to generate_tau_matrix is too high" << std::endl;
        exit(1);
    }

    if(tmp_i == -1){
        std::cout << "temperature given to generate_tau_matrix is too low" << std::endl;
        exit(1);
    }

    if(tau.size() != num_energy_groups or dtau_dUm.size() != num_energy_groups){
        std::cout << "tau or dtau_dUm given to generate_tau_matrix has less then `num_energy_groups` rows" << std::endl;
        exit(1);
    }

    for(std::size_t g=0; g < num_energy_groups; ++g){
        if(tau[g].size() != num_energy_groups or dtau_dUm[g].size() != num_energy_groups){
            std::cout << "tau or dtau_dUm given to generate_tau_matrix has less then `num_energy_groups` columns" << std::endl;
            exit(1);
        }
    }

    double const k_bT = units::k_boltz*temperature;

    using boost::math::pow;
    double constexpr fac = pow<3>(units::clight) / (8.0*M_PI*units::planck_constant);
    for(std::size_t g=0; g < num_energy_groups; ++g){
        double const Bg = planck_integral::planck_energy_density_group_integral(energy_groups_boundries[g], energy_groups_boundries[g+1], temperature);

        double const nu = energy_groups_center[g] / units::planck_constant;
        double const dnu = (energy_groups_boundries[g+1] - energy_groups_boundries[g])/units::planck_constant;

        n_eq[g] = fac*Bg/(pow<3>(nu)*dnu);
        B[g] = Bg;
    }

    double const Nelectron = density*units::Navogadro/A*Z;
    double const x = (temperature-temperature_grid[tmp_i])/(temperature_grid[tmp_i+1]-temperature_grid[tmp_i]);
    for(std::size_t i = 0; i < num_energy_groups; ++i){
        double const E_i = energy_groups_center[i];

        for(std::size_t j=i; j < num_energy_groups; ++j){
            // double const interp_value_log = S_log_tables[tmp_i][i][j]*(1. - x) + S_log_tables[tmp_i+1][i][j]*x;
            // tau_temp[i][j] = std::exp(interp_value_log);
            
            double const interp_value = std::exp(S_log_tables[tmp_i][i][j])*(1. - x) + std::exp(S_log_tables[tmp_i+1][i][j])*x;
            tau[i][j] = interp_value;

            if(i == j) continue;

            double const E_j = energy_groups_center[j];
            // double const w_i = energy_groups_boundries[i+1] - energy_groups_boundries[i];
            // double const w_j = energy_groups_boundries[j+1] - energy_groups_boundries[j];

            // double const detailed_balance_factor = (E_i*E_i*w_i)/(E_j*E_j*w_j)*std::exp((E_j-E_i)/k_bT);

            double const detailed_balance_factor = (1.0+n_eq[j])*B[i]*E_j / ((1.0+n_eq[i])*B[j]*E_i);

            tau[j][i] = tau[i][j]*detailed_balance_factor;
        }
    }


    dtau_dUm = dSdUm_tables[tmp_i];
    for(std::size_t i=0; i<num_energy_groups; ++i){
        for(std::size_t j=0; j<num_energy_groups; ++j){
            tau[i][j] *= Nelectron;
            dtau_dUm[i][j] *= Nelectron;
        }
    }
}


Matrix tau_matrix_monte_carlo_engine::return_tau_matrix(double const temperature, double const density, double const A, double const Z){
    Matrix tau(num_energy_groups, Vector(num_energy_groups, 0.0));
    Matrix dtau(num_energy_groups, Vector(num_energy_groups, 0.0));

    generate_tau_matrix(temperature, density, A, Z, tau, dtau);

    return tau;
}
