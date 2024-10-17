#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../tau_matrix_monte_carlo.hpp"

namespace tau_matrix {
    void bind_tau_matrix_monte_carlo(pybind11::module& m){
        using namespace pybind11::literals;

        pybind11::class_<tau_matrix_monte_carlo_engine>(m, "tau_matrix_monte_carlo_engine")
        .def(pybind11::init<std::vector<double> const,
                            std::vector<double> const,
                            std::size_t const,
                            bool const,
                            std::size_t const>(),
                            pybind11::kw_only(),
                            "energy_groups_center"_a,
                            "energy_groups_boundaries"_a,
                            "num_of_samples"_a,
                            "force_detailed_balance"_a,
                            "seed"_a=0)
        .def("sample_gamma",      &tau_matrix_monte_carlo_engine::sample_gamma,      pybind11::kw_only(), "temperature"_a)
        .def("generate_S_matrix", &tau_matrix_monte_carlo_engine::generate_S_matrix, pybind11::kw_only(), "temperature"_a, "log_grid"_a=false)
        .def("generate_tables",   &tau_matrix_monte_carlo_engine::generate_tables,   pybind11::kw_only(), "temperature_grid"_a)
        .def("return_tau_matrix", &tau_matrix_monte_carlo_engine::return_tau_matrix, pybind11::kw_only(), "temperature"_a, "density"_a, "A"_a, "Z"_a)
        ;
    }
}


PYBIND11_MODULE(_tau_matrix_monte_carlo, m){
    m.doc() = "tau matrix monte carlo c++ module";

    tau_matrix::bind_tau_matrix_monte_carlo(m);
}