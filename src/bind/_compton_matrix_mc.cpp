#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../compton_matrix_mc.hpp"

void bind_compton_matrix_mc(pybind11::module& m){
    using namespace pybind11::literals;

    pybind11::class_<ComptonMatrixMC>(m, "ComptonMatrixMC")
    .def(pybind11::init<std::vector<double> const,
                        std::vector<double> const,
                        std::size_t const,
                        bool const,
                        int const>(),
                        pybind11::kw_only(),
                        "energy_groups_centers"_a,
                        "energy_groups_boundaries"_a,
                        "num_of_samples"_a,
                        "force_detailed_balance"_a,
                        "seed"_a=-1)
    .def("sample_gamma",       &ComptonMatrixMC::sample_gamma,       pybind11::kw_only(), "temperature"_a)
    .def("calculate_S_matrix", &ComptonMatrixMC::calculate_S_matrix, pybind11::kw_only(), "temperature"_a)
    .def("set_tables",         &ComptonMatrixMC::set_tables,         pybind11::kw_only(), "temperature_grid"_a)
    .def("get_tau_matrix",     pybind11::overload_cast<double const, double const, double const, double const>(&ComptonMatrixMC::get_tau_matrix), pybind11::kw_only(), "temperature"_a, "density"_a, "A"_a, "Z"_a)
    ;
}

PYBIND11_MODULE(_compton_matrix_mc, m){
    m.doc() = "Compton Matrix Monte Carlo c++ module";

    bind_compton_matrix_mc(m);
}