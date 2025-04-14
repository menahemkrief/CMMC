#include <limits>
#include <ctime>
#include <cmath>

#include "compton_matrix_mc.hpp"
#include "units/units.hpp"
#include "planck_integral/planck_integral.hpp"

#include <boost/math/special_functions/pow.hpp>
#ifdef RICH_MPI
    #include <mpi.h>
#endif

static double constexpr signaling_NaN = std::numeric_limits<double>::signaling_NaN();

#ifdef RICH_MPI
namespace
{
    void ReduceMatrixSum(Matrix &S, size_t const num_energy_groups)
    {
        int world_size = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        std::vector<double> send_vector(num_energy_groups * num_energy_groups, 0);

        for(std::size_t g0=0; g0 < num_energy_groups; ++g0){
            for(std::size_t g=0; g < num_energy_groups; ++g){
                send_vector[g0*num_energy_groups + g] = S[g0][g];
            }
        }
        
        MPI_Allreduce(MPI_IN_PLACE, send_vector.data(), num_energy_groups * num_energy_groups, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        for(std::size_t g0=0; g0 < num_energy_groups; ++g0){
            for(std::size_t g=0; g < num_energy_groups; ++g){
                S[g0][g] = send_vector[g0*num_energy_groups + g] / world_size;
            }
        }
    }
}
#endif

ComptonMatrixMC::ComptonMatrixMC(Vector const energy_groups_centers_, 
                                 Vector const energy_groups_boundries_, 
                                 std::size_t const num_of_samples_, 
                                 bool const force_detailed_balance_,
                                 std::optional<unsigned int> const seed_) :
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
                            force_detailed_balance(force_detailed_balance_),
                            temperature_grid(),
                            S_log_tables(),
                            dSdUm_tables(),
                            n_eq(num_energy_groups, signaling_NaN),
                            B(num_energy_groups, signaling_NaN),
                            up_scattering_last(signaling_NaN),
                            down_scattering_last(signaling_NaN),
                            up_scattering_last_table(),
                            down_scattering_last_table() {
    int rank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    sample_uniform_01.engine().seed(seed + rank);
    if(rank == 0) printf("ComptonMatrixMC initialized with MPI. Seed=%d\n", seed);
#else
    printf("ComptonMatrixMC initialized without MPI. Seed=%d\n", seed);
#endif

    if(num_energy_groups + 1 != energy_groups_boundries.size()){
        printf("ComptonMatrixMC fatal - inconsistent number of energy group boundaries and centers\n");
        exit(1);
    }
    for(std::size_t g=0; g<num_energy_groups; ++g){
        if(energy_groups_boundries[g]   < 0. or
           energy_groups_boundries[g]   >= energy_groups_boundries[g+1] or
           energy_groups_boundries[g]   >= energy_groups_centers[g] or
           energy_groups_boundries[g+1] <= energy_groups_centers[g]){
            printf("ComptonMatrixMC fatal - inconsistent energy group boundaries/centers\n");
            exit(1);
        }
    }
    if(rank == 0)
    {
    printf("Compton matrices defined on %ld groups.\nPhoton energy group boundaries (in kev) \n", num_energy_groups);
    for(auto const e : energy_groups_boundries){
        printf("%g ", e/units::kev);
    }
    printf("\n");    
    }

    for(std::size_t g=0; g < num_energy_groups; ++g){
        energy_groups_width[g] = energy_groups_boundries[g+1] - energy_groups_boundries[g];
    }
}

double ComptonMatrixMC::sample_gamma(double const temperature){
    
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

void ComptonMatrixMC::set_Bg_ng(double const temperature){
    using boost::math::pow;
    double constexpr fac = pow<3>(units::clight) / (8.0*M_PI*units::planck_constant);
    for(std::size_t g=0; g < num_energy_groups; ++g){
        double const Bg = planck_integral::planck_energy_density_group_integral(energy_groups_boundries[g], energy_groups_boundries[g+1], temperature);

        double const nu = energy_groups_centers[g] / units::planck_constant;
        double const dnu = (energy_groups_boundries[g+1] - energy_groups_boundries[g])/units::planck_constant;

        n_eq[g] = fac*Bg/(pow<3>(nu)*dnu);
        B[g] = Bg;
    }
}

void ComptonMatrixMC::calculate_S_and_dSdUm_matrices(double const temperature, Matrix& S, Matrix& dSdUm){

    int rank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    for(std::size_t i=0; i < num_energy_groups; ++i){
        for(std::size_t j=0; j < num_energy_groups; ++j){
            S[i][j] = 0.0;
            dSdUm[i][j] = 0.0;
        }
    }

    up_scattering_last = 0.0;
    down_scattering_last = 0.0;

    double sum_beta = 0.0;
    std::vector<double> const Omega_0(3., 0.);
    std::vector<double> Omega_0_tag(3., 0.), Omega_e(3., 0.), Omega_tag(3., 0.), Omega_p_tag(3., 0.);
    std::vector<double> weight(num_energy_groups, 0.);
    for(std::size_t sample_i=0; sample_i < num_of_samples; ++sample_i){

        if(rank == 0)
        {
        if((sample_i+1) % (num_of_samples/4) == 0 or sample_i==0){
            printf("Compton matrix T=%gkev sample %ld/%ld [%d%%]\n", temperature/units::kev_kelvin, sample_i+1, num_of_samples, int(100*double(sample_i+1.)/double(num_of_samples)));
                }
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
            double const boundry_g0 = energy_groups_boundries[g0];
            double width = std::min(energy_groups_boundries[g0+1]-boundry_g0, 30 * units::k_boltz*temperature);

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
            
            if(g0 == static_cast<std::size_t>(g)){
                S[g0][g] += sigma;
                dSdUm[g0][g] += sigma*gamma;

                if(g0+1 == num_energy_groups){
                    if(E0 < E){
                        up_scattering_last += (E-E0)/energy_groups_centers[g0] * sigma;
                    } else {
                        down_scattering_last += (E0-E)/energy_groups_centers[g0] * sigma;
                    }
                }
            } else {
                double const fac = (E-E0)/(energy_groups_centers[g]-energy_groups_centers[g0]);
                S[g0][g] += sigma*fac;
                dSdUm[g0][g] += sigma*gamma*fac;

                S[g0][g0] += sigma*(1.0-fac);
                dSdUm[g0][g0] += sigma*gamma*(1.0-fac);
            }
        }    
    }
 
#ifdef RICH_MPI
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // 
    ReduceMatrixSum(S, num_energy_groups);
    ReduceMatrixSum(dSdUm, num_energy_groups);
    
    MPI_Allreduce(MPI_IN_PLACE, &sum_beta, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &up_scattering_last, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &down_scattering_last, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    sum_beta /= world_size;
    up_scattering_last /= world_size;
    down_scattering_last /= world_size;
    
    for(std::size_t g0=0; g0 < num_energy_groups; ++g0){
        weight[g0] /= world_size;
    }

    MPI_Allreduce(MPI_IN_PLACE, weight.data(), num_energy_groups, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

    // total weight
    double const beta_avg = sum_beta / num_of_samples;
    
    // multiply by sigma_thomson and normalization factors
    for(std::size_t g0=0; g0 < num_energy_groups; ++g0){
        double const weight_avg = weight[g0]/num_of_samples;
        for(std::size_t g=0; g < num_energy_groups; ++g){
            if(weight_avg < std::numeric_limits<double>::min() * 1e40)
            {
                S[g0][g] = std::numeric_limits<double>::min()*1e40;
                dSdUm[g0][g] = 0.0;
                continue;
            }
            S[g0][g] *= units::sigma_thomson/(num_of_samples*beta_avg*weight_avg);
            dSdUm[g0][g] *= units::sigma_thomson/(num_of_samples*beta_avg*weight_avg);
            S[g0][g] = std::max(S[g0][g], std::numeric_limits<double>::min()*1e40);

        }
        if(g0+1 == num_energy_groups){
            up_scattering_last *= units::sigma_thomson/(num_of_samples*beta_avg*weight_avg);
            down_scattering_last *= units::sigma_thomson/(num_of_samples*beta_avg*weight_avg);
        }
    }

    using boost::math::pow;
    
    Matrix detailed_balance_factors(num_energy_groups, Vector(num_energy_groups, 1));
    if(force_detailed_balance){
        set_Bg_ng(temperature);
        double constexpr thresh = units::sigma_thomson*std::numeric_limits<double>::epsilon()*1e3;

        for(std::size_t g=0; g < num_energy_groups; ++g){
            double const E_g = energy_groups_centers[g];
            
            for(std::size_t gt=g+1; gt<num_energy_groups; ++gt){
                if(S[gt][g] < thresh and S[g][gt] < thresh) continue;
                
                if(B[gt]*E_g < std::numeric_limits<double>::min() * 1e40)
                    continue;
                double const E_gt = energy_groups_centers[gt];
                double const detailed_balance_factor = (1.0+n_eq[gt])*B[g]*E_gt / ((1.0+n_eq[g])*B[gt]*E_g);
                
                if(std::isnan(detailed_balance_factor)) continue;

                if(detailed_balance_factor < 1.0){
                    detailed_balance_factors[gt][g] = S[g][gt]*detailed_balance_factor/S[gt][g];
                    S[gt][g] = S[g][gt]*detailed_balance_factor;
                }
                else{
                    detailed_balance_factors[g][gt] = S[gt][g]/(S[g][gt]*detailed_balance_factor);
                    S[g][gt] = S[gt][g]/detailed_balance_factor;
                }
            }
        }
    }
}

void ComptonMatrixMC::set_tables(std::vector<double> const& temperature_grid_){

    if(temperature_grid_.size()<2){
        printf("Compton temperature grid has less than two temperature points - %ld\n", temperature_grid_.size());
        exit(1);
    }
    int rank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif   
    if(rank == 0)
    printf("setting Compton matrix tables for %ld temperatures (in kev):\n", temperature_grid_.size());
    for(std::size_t i=0; i<temperature_grid_.size(); ++i){
        if(rank == 0)
        printf("%g ", temperature_grid_[i]/units::kev_kelvin);
        if(i>0 and temperature_grid_[i]<=temperature_grid_[i-1]){
            printf("fatal - Compton temperature grid is not monotonic\n");
            exit(1);
        }
    }
    if(rank == 0)
    printf("\n");

    temperature_grid = temperature_grid_;
    S_log_tables = std::vector<Matrix>(temperature_grid.size(), Matrix(num_energy_groups, Vector(num_energy_groups, 0.0)));
    dSdUm_tables = std::vector<Matrix>(temperature_grid.size(), Matrix(num_energy_groups, Vector(num_energy_groups, 0.0)));

    up_scattering_last_table = std::vector<double>(temperature_grid.size(), 0.0);
    down_scattering_last_table = std::vector<double>(temperature_grid.size(), 0.0);

    for(std::size_t i=0; i < temperature_grid.size(); ++i){
        calculate_S_and_dSdUm_matrices(temperature_grid[i], S_log_tables[i], dSdUm_tables[i]);
        for(std::size_t g0=0; g0 < num_energy_groups; ++g0){
            for(std::size_t g=0; g < num_energy_groups; ++g){
                S_log_tables[i][g0][g] = std::log(S_log_tables[i][g0][g]);

            }

            if(g0+1 == num_energy_groups){
                up_scattering_last_table[i] = up_scattering_last;
                down_scattering_last_table[i] = down_scattering_last;
            }
        }
    }

    for(std::size_t i=0; i < temperature_grid.size(); ++i){
        size_t lower = i > 0 ? i - 1 : 0;
        size_t upper = i < temperature_grid.size() - 1 ? i + 1 : temperature_grid.size() - 1;
        while((temperature_grid[upper] - temperature_grid[lower]) < std::max(1e5, temperature_grid[i] * 0.2))
        {
            if(lower > 0) --lower;
            if(upper < (temperature_grid.size() - 1)) ++upper;
            if(lower == 0 && upper == (temperature_grid.size() - 1))
                break;
        }
        double dUm = temperature_grid[upper] -temperature_grid[lower];
        for (std::size_t g0=0; g0 < num_energy_groups; ++g0){
            for (std::size_t g=0; g < num_energy_groups; ++g){
                dSdUm_tables[i][g0][g] = (std::exp(S_log_tables[upper][g0][g]) - std::exp(S_log_tables[lower][g0][g])) / dUm;
            }

            if(g0+1 == num_energy_groups){
                double const dS_upper = up_scattering_last_table[upper] - down_scattering_last_table[upper];
                double const dS_lower = up_scattering_last_table[lower] - down_scattering_last_table[lower];
                double const dS = up_scattering_last_table[i] - down_scattering_last_table[i];
                if((dS < 0 && dS_lower < 0 && dS_upper < 0) || (dS > 0 && dS_lower > 0 && dS_upper > 0))
                    dSdUm_tables[i][g0][g0] = (dS_upper - dS_lower)/dUm;
                else
                    dSdUm_tables[i][g0][g0] = 0;
            }
        }
    }
    if(rank == 0)
        std::cout<<"Done compton matrix tables"<<std::endl;

  
}

void ComptonMatrixMC::get_tau_matrix(double const temperature, double const density, double const A, double const Z, Matrix& tau, Matrix& dtau_dUm){
    auto const tmp_iterator = std::lower_bound(temperature_grid.cbegin(), temperature_grid.cend(), temperature);
    auto const tmp_i = std::distance(temperature_grid.cbegin(), tmp_iterator) - 1; //  gives the index of lower bound of the temperature in the temperature grid

    if(tmp_i+1 == static_cast<int>(temperature_grid.size())){
        printf("temperature T=%gkev given to get_tau_matrix is too high (maximal table temperature=%gkev)\n", temperature/units::kev_kelvin, temperature_grid.back()/units::kev_kelvin);
        exit(1);
    }

    if(tmp_i == -1){
        printf("temperature T=%gkev given to get_tau_matrix is too low (minimal table temperature=%gkev)\n", temperature/units::kev_kelvin, temperature_grid[0]/units::kev_kelvin);
        exit(1);
    }

    if(tau.size() != num_energy_groups or dtau_dUm.size() != num_energy_groups){
        std::cout << "tau or dtau_dUm given to get_tau_matrix has less then `num_energy_groups` rows" << std::endl;
        exit(1);
    }

    for(std::size_t g=0; g < num_energy_groups; ++g){
        if(tau[g].size() != num_energy_groups or dtau_dUm[g].size() != num_energy_groups){
            std::cout << "tau or dtau_dUm given to get_tau_matrix has less then `num_energy_groups` columns" << std::endl;
            exit(1);
        }
    }

    if(force_detailed_balance) set_Bg_ng(temperature);

    double const x = (temperature-temperature_grid[tmp_i])/(temperature_grid[tmp_i+1]-temperature_grid[tmp_i]);
    for(std::size_t i = 0; i < num_energy_groups; ++i){
        double const E_i = energy_groups_centers[i];

        for(std::size_t j=i; j < num_energy_groups; ++j){
            tau[i][j] = std::exp(S_log_tables[tmp_i][i][j])*(1. - x) + std::exp(S_log_tables[tmp_i+1][i][j])*x;
            dtau_dUm[i][j] = dSdUm_tables[tmp_i][i][j]*(1. - x) + dSdUm_tables[tmp_i+1][i][j]*x;

            if(i == j) continue;
            
            tau[j][i] = std::exp(S_log_tables[tmp_i][j][i])*(1. - x) + std::exp(S_log_tables[tmp_i+1][j][i])*x;
            dtau_dUm[j][i] = dSdUm_tables[tmp_i][j][i]*(1. - x) + dSdUm_tables[tmp_i+1][j][i]*x;

            // enforce detailed balance on the interpolated matrix
            if(force_detailed_balance){
                if(B[j]*E_i < std::numeric_limits<double>::min() * 1e40)
                    continue;
                double const E_j = energy_groups_centers[j];
                double const detailed_balance_factor = (1.0+n_eq[j])*B[i]*E_j / ((1.0+n_eq[i])*B[j]*E_i);
                
                if(std::isnan(detailed_balance_factor)) continue;
                
                if(detailed_balance_factor < 1.0) {
                    tau[j][i] = tau[i][j]*detailed_balance_factor;
                } else {
                    tau[i][j] = tau[j][i]/detailed_balance_factor;
                }
            } 
            else{
                tau[j][i] = std::exp(S_log_tables[tmp_i][j][i])*(1. - x) + std::exp(S_log_tables[tmp_i+1][j][i])*x;
            }
        }
    }

    double const Nelectron = density*units::Navogadro/A*Z;
    for(std::size_t i=0; i<num_energy_groups; ++i){
        for(std::size_t j=0; j<num_energy_groups; ++j){
            tau[i][j] *= Nelectron;
            dtau_dUm[i][j] *= Nelectron;
        }
    }
}

Matrix ComptonMatrixMC::get_tau_matrix(double const temperature, double const density, double const A, double const Z){
    Matrix tau(num_energy_groups, Vector(num_energy_groups, 0.0));
    Matrix dtau(num_energy_groups, Vector(num_energy_groups, 0.0));

    get_tau_matrix(temperature, density, A, Z, tau, dtau);

    return tau;
}

std::pair<Matrix, Matrix> ComptonMatrixMC::get_S_and_dSdUm_matrices(double const temperature){
    Matrix S(num_energy_groups, Vector(num_energy_groups, 0.0));
    Matrix dSdUm(num_energy_groups, Vector(num_energy_groups, 0.0));

    calculate_S_and_dSdUm_matrices(temperature, S, dSdUm);
    return std::pair(S, dSdUm);
}

std::pair<double, double> ComptonMatrixMC::get_last_group_upscattering_and_downscattering(double const temperature, double const density, double const A, double const Z){
    auto const tmp_iterator = std::lower_bound(temperature_grid.cbegin(), temperature_grid.cend(), temperature);
    auto const tmp_i = std::distance(temperature_grid.cbegin(), tmp_iterator) - 1; //  gives the index of lower bound of the temperature in the temperature grid

    if(tmp_i+1 == static_cast<int>(temperature_grid.size())){
        printf("temperature T=%gkev given to get_tau_matrix is too high (maximal table temperature=%gkev)\n", temperature/units::kev_kelvin, temperature_grid.back()/units::kev_kelvin);
        exit(1);
    }

    if(tmp_i == -1){
        printf("temperature T=%gkev given to get_tau_matrix is too low (minimal table temperature=%gkev)\n", temperature/units::kev_kelvin, temperature_grid[0]/units::kev_kelvin);
        exit(1);
    }

    double const x = (temperature-temperature_grid[tmp_i])/(temperature_grid[tmp_i+1]-temperature_grid[tmp_i]);

    double upscattering_interp = up_scattering_last_table[tmp_i]*(1. - x) + up_scattering_last_table[tmp_i+1]*x;
    double downscattering_interp = down_scattering_last_table[tmp_i]*(1. - x) + down_scattering_last_table[tmp_i+1]*x;

    double const Nelectron = density*units::Navogadro/A*Z;
    
    upscattering_interp *= Nelectron;
    downscattering_interp *= Nelectron;

    return std::pair(upscattering_interp, downscattering_interp);
}  