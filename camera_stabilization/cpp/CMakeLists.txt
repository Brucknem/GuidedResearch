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
add_executable(Main
        main.cpp
        DynamicCalibration.hpp
        Utils.hpp
        OpticalFlow.h
        BackgroundSegmentation.h
        )

target_link_libraries(Main PUBLIC CameraStabilization ImageUtils ${OpenCV_LIBS})
target_include_directories(Main PUBLIC "${PROJECT_BINARY_DIR}")

# add the executable
add_executable(CompareScaleFactors
        compare_scale_factors.cpp
        DynamicCalibration.hpp
        Utils.hpp
        OpticalFlow.h
        BackgroundSegmentation.h
        )

target_link_libraries(CompareScaleFactors PUBLIC CameraStabilization ImageUtils ${OpenCV_LIBS})
target_include_directories(CompareScaleFactors PUBLIC "${PROJECT_BINARY_DIR}")
