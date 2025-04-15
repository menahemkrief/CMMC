#ifndef __COMPTON_MATRIX_MC__
#define __COMPTON_MATRIX_MC__

#include <boost/random.hpp>
#include <optional>

using Vector = std::vector<double>;
using Matrix = std::vector<std::vector<double>>;

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
     * @param seed_ - if given non-negative value - sets the seed of the random number generator - to enable bit-by-bit reproducible results.
     */
        ComptonMatrixMC(
            Vector const compton_temperatures_,
            Vector const energy_groups_centers_, 
            Vector const energy_groups_boundries_, 
            std::size_t const num_of_samples_, 
            std::optional<unsigned int> const seed_=std::nullopt);

        /**
         * @brief Calculates the *microscopic* Compton scattering matrix at
         * the given temperature (without interpolation on temperature). 
         * The calculation is perfoemed using a Monte-Carlo integration.
         * 
         * @param temperature the given temeprature [K]
         * @param S the S matrix 
         * @return Matrix the microscopic Compton scattering matrix [cm^2]
         */
        void calculate_S_matrix(double const temperature, Matrix& S);
        
        Matrix get_S_matrix(double const temperature);

        
        /**
         * @brief Get the Compton tau matrix - which is the macroscopic cross secion for Compton scattering
         * The result is calculated for the given temperature by interpolation over the set of temperatures
         * given to `set_tables`.
         * 
         * @param temperature - the given electron temperature [K]
         * @param density - the given mass density
         * @param A  - the given atomic wieght
         * @param Z - the average number of free electrons per nucleous (ionization)
         * @param[in out] tau - the resulting Compton tau marix (units of 1/cm).
         */
        void get_tau_matrix(double const temperature, double const density, double const A, double const Z, Matrix& tau);

        Matrix get_tau_matrix(double const temperature, double const density, double const A, double const Z);
        
        /**
         * @brief Get the Compton tau matrix - which is the macroscopic cross secion for Compton scattering
         * The result is calculated for the given temperature by interpolation over the set of temperatures
         * given to `set_tables`.
         * 
         * @param temperature - the given electron temperature [K]
         * @param density - the given mass density
         * @param A  - the given atomic wieght
         * @param Z - the average number of free electrons per nucleous (ionization)
         * @param[in out] dtau_dT the derivative of tau with respect to T
         */
        void get_dtau_matrix(double const temperature, double const density, double const A, double const Z, Matrix& dtau_dT);

        Matrix get_dtau_matrix(double const temperature, double const density, double const A, double const Z);

       /**
        * @brief Sample from the distribution A*gamma^2e^(-gamma/theta)
        * where theta=kB T / m_e * c^2
        * @param temperature the given temperature [K]
        * @return double the sampled value of gamma
        */
       double sample_gamma(double const temperature);

       /**
        * @brief Retrieves the maximum temperature from the temperature grid.
        * 
        * This function returns the last element of the temperature grid, 
        * which represents the highest temperature value stored in the grid.
        * 
        * @return double The maximum temperature value in the temperature grid.
        */
       double get_maximum_compton_temperature() const { return compton_temperatures.back();}
        
        std::pair<double, double> get_last_group_upscattering_and_downscattering(double const temperature, double const density, double const A, double const Z);
        
        Vector get_compton_temperatures() const { return compton_temperatures; }
        Vector get_energy_groups_centers() const { return energy_groups_centers; }
        Vector get_energy_groups_width() const { return energy_groups_width; }
        Vector get_energy_groups_boundries() const { return energy_groups_boundries; }
        std::size_t get_num_energy_groups() const { return num_energy_groups; }
        std::size_t get_num_of_samples() const { return num_of_samples; }
        unsigned int get_seed() const { return seed; }
        std::vector<Matrix> get_S_tables() const { return S_tables; }
        std::vector<Matrix> get_dSdT_tables() const { return dSdT_tables; }
      private:
        /**
         * @brief Given a set of temperatures, calculates and store the S matrices, so 
         * that tau matrices can be calculated for other temperatures using interpolation between 
         * the given temperatures (using the `get_tau_matrix` functions).
         * @param temperature_grid - a set of temperatures [K]
         * @return * void 
         */
        void set_tables(std::vector<double> const& temperature_grid_);
        /*!
        @brief force detailed balance at `temperature` for `mat`
        @param temperature at which to force detailed balance
        @param mat matrix on which to force detailed balance
        */
        void enforce_detailed_balance(double const temperature, Matrix& mat);
        
        /*!
        @brief calculates the planck integrals per group `Bg` and the equilibrium occupancy number `n_eq`
        @param temperature the temperature for the planckian
        */
        void calculate_Bg_ng(double const temperature);
        
        Vector const compton_temperatures; // temperature grid for the compston tables

        Vector const energy_groups_centers;
        Vector const energy_groups_boundries;
        Vector energy_groups_width;
        std::size_t const num_energy_groups;
        
        std::size_t const num_of_samples; // number of samples of the Monte Carlo integration
        
        unsigned int const seed;
        boost::random::variate_generator<boost::random::mt19937_64, boost::random::uniform_01<>> sample_uniform_01;

        // tabulation
        std::vector<Matrix> S_tables;
        std::vector<Matrix> dSdT_tables;

        // auxiliary arrays
        std::vector<double> n_eq; // temporary array to avoid repeated allocations, occupancy number at equilibrium
        std::vector<double> B; // temporary array to avoid repeated allocations, group energy density at equilibrium (i.e. the integral on the planck function)

        double up_scattering_last;
        double down_scattering_last;

        std::vector<double> up_scattering_last_table;
        std::vector<double> down_scattering_last_table;
};

#endif