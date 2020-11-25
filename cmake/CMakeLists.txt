cmake_minimum_required(VERSION 3.0.0)
project(CameraStabilization VERSION 0.1.0)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

############################################################################
# main
############################################################################

add_executable(CameraStabilization main.cpp curve_fitting_data.hpp)
target_include_directories(CameraStabilization PUBLIC
    "${PROJECT_BINARY_DIR}"
)

############################################################################
# lib
############################################################################

add_subdirectory(lib)
list(APPEND EXTRA_LIBS OptimizationProblems)
target_link_libraries(CameraStabilization OptimizationProblems)

############################################################################
# tests
############################################################################

include(CTest)
enable_testing()

add_test(NAME Runs COMMAND CameraStabilization)
add_test(NAME HelloWorld COMMAND CameraStabilization)
set_tests_properties(HelloWorld
    PROPERTIES PASS_REGULAR_EXPRESSION "Ceres Solver Report: Iterations: 3, Initial cost: 4[.]512500e[+]01, Final cost: 5[.]012552e[-]16, Termination: CONVERGENCE"
)

add_test(NAME PowellsFunction COMMAND CameraStabilization)
set_tests_properties(PowellsFunction
    PROPERTIES PASS_REGULAR_EXPRESSION "Final x1 = 0.000146222, x2 = -1.46222e-05, x3 = 2.40957e-05, x4 = 2.40957e-05"
)

add_test(NAME ExponentialFunction COMMAND CameraStabilization)
set_tests_properties(ExponentialFunction
    PROPERTIES PASS_REGULAR_EXPRESSION "Final   m: 0.291861 c: 0.131439"
)
  
############################################################################
# install
############################################################################

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
