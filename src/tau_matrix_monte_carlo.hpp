#ifndef TAU_MATRIX_MONTE_CARLO
#define TAU_MATRIX_MONTE_CARLO

#include <boost/random.hpp>

using Vector = std::vector<double>;
using Matrix = std::vector<std::vector<double>>;

class tau_matrix_monte_carlo_engine {
    public:
        tau_matrix_monte_carlo_engine(Vector const energy_groups_center_, 
                                      Vector const energy_groups_boundaries_, 
                                      std::size_t const num_of_samples_,
                                      bool const force_detailed_balance_);
        
        // for debugging enable to set the seed for the random number generator
        tau_matrix_monte_carlo_engine(Vector const energy_groups_center_, 
                                      Vector const energy_groups_boundries_, 
                                      std::size_t const num_of_samples_, 
                                      bool const force_detailed_balance_,
                                      std::size_t const seed);

        Matrix generate_S_matrix(double const temperature, bool const log_grid);
        
        void generate_tables(std::vector<double> const& tmp_grid);
        
        void generate_tau_matrix(double const temperature, double const density, double const A, double const Z, Matrix& tau, Matrix& dtau_dUm);

        Matrix return_tau_matrix(double const temperature, double const density, double const A, double const Z);

        /*
            \brief Sample from the distribution A*gamma^2e^(-gamma/theta)
        */
        double sample_gamma(double const temperature);
    
    private:
        Vector energy_groups_center;
        Vector energy_groups_boundries;
        std::size_t num_energy_groups;
        
        Matrix S_temp;

        std::size_t num_of_samples;
        boost::random::variate_generator<boost::random::mt19937, boost::random::uniform_01<>> sample_uniform_01;

        bool force_detailed_balance;

        Vector temperature_grid;
        std::vector<Matrix> S_log_tables;

        std::vector<Matrix> dSdUm_tables;

        std::vector<double> n_eq;
        std::vector<double> B;
};

#endif