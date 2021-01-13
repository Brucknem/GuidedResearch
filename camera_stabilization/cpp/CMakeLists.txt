cmake_minimum_required(VERSION 3.10)

# set the project name
project(CameraStabilization C CXX)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

# packages
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


# add the Main testing
add_executable(Main
        main.cpp
        DynamicCalibration.hpp
        Utils.hpp
        OpticalFlow.h
        BackgroundSegmentation.h
        )

target_link_libraries(Main PUBLIC ${OpenCV_LIBS})
target_include_directories(Main PUBLIC "${PROJECT_BINARY_DIR}")

# add the CompareScaleFactors
add_executable(CompareScaleFactors
        compare_scale_factors.cpp
        DynamicCalibration.hpp
        Utils.hpp
        OpticalFlow.h
        BackgroundSegmentation.h
        )

target_link_libraries(CompareScaleFactors PUBLIC ${OpenCV_LIBS})
target_include_directories(CompareScaleFactors PUBLIC "${PROJECT_BINARY_DIR}")

# add the VisualizeKeyframeUpdate
add_executable(VisualizeKeyframeUpdate
        visualize_keyframe_update.cpp
        DynamicCalibration.hpp
        Utils.hpp
        OpticalFlow.h
        BackgroundSegmentation.h
        )

target_link_libraries(VisualizeKeyframeUpdate PUBLIC ${OpenCV_LIBS})
target_include_directories(VisualizeKeyframeUpdate PUBLIC "${PROJECT_BINARY_DIR}")

# add the VisualizeKeyframeUpdate
add_executable(VisualizeFeatureMatching
        visualize_feature_matching.cpp
        DynamicCalibration.hpp
        Utils.hpp
        OpticalFlow.h
        BackgroundSegmentation.h
        )

target_link_libraries(VisualizeFeatureMatching PUBLIC ${OpenCV_LIBS})
target_include_directories(VisualizeFeatureMatching PUBLIC "${PROJECT_BINARY_DIR}")

########################################################################################################################
### TESTING ###
########################################################################################################################
enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

add_executable(Tests tests/tests.cpp)
target_link_libraries(Tests PUBLIC libCameraStabilization ${GTEST_BOTH_LIBRARIES} ${OpenCV_LIBS})
target_include_directories(Tests PUBLIC "${PROJECT_BINARY_DIR}")

add_test(AllTestsInFoo Tests)
