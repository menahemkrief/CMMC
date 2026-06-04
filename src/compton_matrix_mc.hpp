#ifndef __COMPTON_MATRIX_MC__
#define __COMPTON_MATRIX_MC__

#include <boost/random.hpp>

using Vector = std::vector<double>;
using Matrix = std::vector<std::vector<double>>;
using AngleCdf = std::vector<double>;
using AngleMatrix = std::vector<std::vector<AngleCdf>>;

class ComptonMatrixMC {
    public:
    /**
     * @brief Constructs a new Compton Matrix Monte Carlo object for the calculation of
     * Compton scattering matrix on given photon energy groups.
     * 
     * All quantities are always given in c.g.s. units
     * 
     * @param energy_groups_centers_ - centers of energy groups [erg] (does not have to be the arithmetic center).
     * @param energy_groups_boundries_ - groups boundaries [erg]
     * @param num_of_samples_ - number of Monet carlo samples for the integration.
     * @param force_detailed_balance_ - whether or not force detailed balance.
     * @param seed_ - if given non-negative value - sets the seed of the random number generator - to enable bit-by-bit reproducible results.
     * @param use_energy_redistribution_ - when true, redistribute off-diagonal MC contributions using linear energy-transfer weighting.
     */
        ComptonMatrixMC(
            Vector const energy_groups_centers_, 
            Vector const energy_groups_boundries_, 
            std::size_t const num_of_samples_, 
            bool const force_detailed_balance_,
            int const seed_=-1,
            bool const use_energy_redistribution_=false);

        /**
         * @brief Calculates the *microscopic* Compton scattering matrix at
         * the given temperature (without interpolation on temperature). 
         * The calculation is perfoemed using a Monte-Carlo integration.
         * 
         * @param temperature the given temeprature [K]
         * @return Matrix the microscopic Compton scattering matrix [cm^2]
         */
        Matrix calculate_S_matrix(double const temperature);
        
        /**
         * @brief Given a set of temperatures, calculates and store the S matrices, so 
         * that tau matrices can be calculated for other temperatures using interpolation between 
         * the given temperatures (using the `get_tau_matrix` functions).
         * @param temperature_grid - a set of temperatures [K]
         * @return * void 
         */
        void set_tables(std::vector<double> const& temperature_grid_);
        
        /**
         * @brief Get the Compton tau matrix - which is the macroscopic cross secion for Compton scattering
         * The result is calculated for the given temperature by interpolation over the set of temperatures
         * given to `set_tables`.
         * 
         * @param temperature - the given electron temperature [K]
         * @param density - the given mass density
         * @param A  - the given atomic wieght
         * @param Z - the average number of free electrons per nucleous (ionization)
         * @param tau - the resulting Compton tau marix (units of 1/cm).
         * @param dtau_dUm the derivative of tau with respect to aT^4
         */
        void get_tau_matrix(double const temperature, double const density, double const A, double const Z, Matrix& tau, Matrix& dtau_dUm);

        Matrix get_tau_matrix(double const temperature, double const density, double const A, double const Z);

       /**
        * @brief Sample from the distribution A*gamma^2e^(-gamma/theta)
        * where theta=kB T / m_e * c^2
        * @param temperature the given temperature [K]
        * @return double the sampled value of gamma
        */
       double sample_gamma(double const temperature);

       /**
        * @brief Get the angular CDF for a Compton scattering transition g0 -> g
        * at the given temperature, by interpolation on the temperature grid.
        * The CDF has NUM_ANGLE_BINS+1 entries at uniformly spaced cos(theta) bin edges
        * from -1 to +1. CDF[i] = P(cos_theta_scat < -1 + 2*i/NUM_ANGLE_BINS).
        *
        * @param temperature electron temperature [K]
        * @param g0 incoming energy group index
        * @param g outgoing energy group index
        * @return std::vector<double> CDF vector of size NUM_ANGLE_BINS+1
        */
       std::vector<double> get_angle_cdf(double temperature, std::size_t g0, std::size_t g) const;

       static constexpr std::size_t NUM_ANGLE_BINS = 32;

    private:
        void set_Bg_ng(double const);
        
        Vector const energy_groups_centers;
        Vector const energy_groups_boundries;
        Vector energy_groups_width;
        std::size_t const num_energy_groups;
        
        std::size_t const num_of_samples;
        unsigned int const seed;
        boost::random::variate_generator<boost::random::mt19937_64, boost::random::uniform_01<>> sample_uniform_01;

        bool const force_detailed_balance;
        bool const use_energy_redistribution;

        // tabulation
        Vector temperature_grid;
        std::vector<Matrix> S_log_tables;
        std::vector<Matrix> dSdUm_tables;

        // auxiliary arrays
        Matrix S_temp;
        std::vector<double> n_eq;
        std::vector<double> B;

        // angular CDF tables: [temp_idx][g0][g] = CDF with NUM_ANGLE_BINS+1 values
        std::vector<AngleMatrix> angle_cdf_tables;
        AngleMatrix angle_histogram_temp;
};

#endif