cmake_minimum_required(VERSION 3.19)

project(cuda_test_project LANGUAGES CUDA CXX)

find_package(Ceres REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

###

enable_language(CUDA)
include(CheckLanguage)
check_language(CUDA)
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

###

add_executable(cuda_test src/add.cu)
set_target_properties(cuda_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
                        
target_link_libraries(cuda_test Ceres::ceres)
target_link_libraries (cuda_test Eigen3::Eigen)
