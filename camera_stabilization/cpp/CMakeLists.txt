cmake_minimum_required(VERSION 3.10)

# set the project name
project(DynamicStabilization)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

# packages
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# library
add_subdirectory(lib/CameraStabilization)
add_subdirectory(lib/ImageUtils)

# add the Main testing
add_executable(Main
        main.cpp
        DynamicCalibration.hpp
        Utils.hpp
        OpticalFlow.h
        BackgroundSegmentation.h
        )

target_link_libraries(Main PUBLIC CameraStabilization ImageUtils ${OpenCV_LIBS})
target_include_directories(Main PUBLIC "${PROJECT_BINARY_DIR}")

# add the CompareScaleFactors
add_executable(CompareScaleFactors
        compare_scale_factors.cpp
        DynamicCalibration.hpp
        Utils.hpp
        OpticalFlow.h
        BackgroundSegmentation.h
        )

target_link_libraries(CompareScaleFactors PUBLIC CameraStabilization ImageUtils ${OpenCV_LIBS})
target_include_directories(CompareScaleFactors PUBLIC "${PROJECT_BINARY_DIR}")

# add the CompareScaleFactors
add_executable(VisualizeKeyframeUpdate
        visualize_keyframe_update.cpp
        DynamicCalibration.hpp
        Utils.hpp
        OpticalFlow.h
        BackgroundSegmentation.h
        )

target_link_libraries(VisualizeKeyframeUpdate PUBLIC CameraStabilization ImageUtils ${OpenCV_LIBS})
target_include_directories(VisualizeKeyframeUpdate PUBLIC "${PROJECT_BINARY_DIR}")
