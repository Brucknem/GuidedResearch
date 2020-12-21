cmake_minimum_required(VERSION 3.10)

# set the project name
project(DynamicStabilization)

# packages
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# library
add_subdirectory(lib/CameraStabilization)
add_subdirectory(lib/ImageUtils)

# add the executable
add_executable(DynamicStabilization
        main.cpp
        DynamicCalibration.hpp
        Utils.hpp
        OpticalFlow.h)

target_link_libraries(DynamicStabilization PUBLIC CameraStabilization ImageUtils ${OpenCV_LIBS})
target_include_directories(DynamicStabilization PUBLIC "${PROJECT_BINARY_DIR}")
