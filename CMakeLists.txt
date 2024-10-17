cmake_minimum_required(VERSION 3.22)
project(CMMC LANGUAGES CXX)

file(GLOB_RECURSE SRC_CPP "src/*.cpp")
file(GLOB SRC_NO_PYBIND "src/*.cpp" "main.cpp")

find_package(pybind11 REQUIRED)


message("pybind11 directory: " ${pybind11_DIR})

message("SRC FILES:")
foreach(file IN LISTS SRC_CPP)
    message(${file})
endforeach()

pybind11_add_module(_tau_matrix_monte_carlo ${SRC_CPP})
target_link_libraries(_tau_matrix_monte_carlo PRIVATE pybind11::module)
set_target_properties(_tau_matrix_monte_carlo PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}" LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/../cpp_modules")

pybind11_add_module(_units ${SRC_CPP})
target_link_libraries(_units PRIVATE pybind11::module)
set_target_properties(_units PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}" LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/../cpp_modules")

pybind11_add_module(_planck_integral ${SRC_CPP})
target_link_libraries(_planck_integral PRIVATE pybind11::module)
set_target_properties(_planck_integral PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}" LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/../cpp_modules")

message("SRC FILES NO PYBIND:")
foreach(file IN LISTS SRC_NO_PYBIND)
    message(${file})
endforeach()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
add_executable(main ${SRC_NO_PYBIND}) 
set_target_properties(main PROPERTIES SUFFIX ".out" RUNTIME_OUTPUT_DIRECTORY  "${PROJECT_BINARY_DIR}/../")


                                             
