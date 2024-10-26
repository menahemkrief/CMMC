cmake_minimum_required(VERSION 3.22)
project(CMMC LANGUAGES CXX)

file(GLOB_RECURSE SRC_CPP "src/*.cpp")
file(GLOB SRC_AND_MAIN "src/*.cpp" "main.cpp")

find_package(pybind11 REQUIRED)

set(COMPILATION_FLAGS "-Wall -Wextra -Werror -Wshadow -fno-common -fstack-protector-all -rdynamic -Wdouble-promotion -fstrict-aliasing")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMPILATION_FLAGS}") 
set(CMAKE_CXX_FLAGS_DEBUG "${COMPILATION_FLAGS} -DDEBUG -O0 -g3 -gdwarf-3 -D_GLIBCXX_DEBUG") 
set(CMAKE_CXX_FLAGS_RELEASE "${COMPILATION_FLAGS} -DNDEBUG -O2")

message("pybind11 directory: " ${pybind11_DIR})

message("SRC FILES:")
foreach(file IN LISTS SRC_CPP)
    message(${file})
endforeach()

pybind11_add_module(_cmmc ${SRC_CPP})
target_link_libraries(_cmmc PRIVATE pybind11::module)
set_target_properties(_cmmc PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}" LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/../cpp_modules")

pybind11_add_module(_units ${SRC_CPP})
target_link_libraries(_units PRIVATE pybind11::module)
set_target_properties(_units PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}" LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/../cpp_modules")

pybind11_add_module(_planck_integral ${SRC_CPP})
target_link_libraries(_planck_integral PRIVATE pybind11::module)
set_target_properties(_planck_integral PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}" LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/../cpp_modules")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
add_executable(main ${SRC_AND_MAIN}) 
set_target_properties(main PROPERTIES SUFFIX ".out" RUNTIME_OUTPUT_DIRECTORY  "${PROJECT_BINARY_DIR}")


                                             
