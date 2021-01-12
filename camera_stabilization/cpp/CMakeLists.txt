cmake_minimum_required(VERSION 3.10)

# Download and unpack googletest at configure time
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download)
if (result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif ()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download)
if (result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif ()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
        ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
        EXCLUDE_FROM_ALL)

# The gtest/gtest_main targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
    include_directories("${gtest_SOURCE_DIR}/include")
endif ()


# set the project name
project(CameraStabilization C CXX)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

# packages
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# library
add_subdirectory(lib/CameraStabilization)
add_subdirectory(lib/ImageUtils)
add_subdirectory(lib)

# add the Main testing
add_executable(Main
        main.cpp
        DynamicCalibration.hpp
        lib/Utils.hpp
        lib/OpticalFlow.h
        lib/BackgroundSegmentation.h
        )

target_link_libraries(Main PUBLIC libCameraStabilization CameraStabilization ImageUtils ${OpenCV_LIBS})
target_include_directories(Main PUBLIC "${PROJECT_BINARY_DIR}")

# add the CompareScaleFactors
add_executable(CompareScaleFactors
        compare_scale_factors.cpp
        DynamicCalibration.hpp
        lib/Utils.hpp
        lib/OpticalFlow.h
        lib/BackgroundSegmentation.h
        )

target_link_libraries(CompareScaleFactors PUBLIC libCameraStabilization CameraStabilization ImageUtils ${OpenCV_LIBS})
target_include_directories(CompareScaleFactors PUBLIC "${PROJECT_BINARY_DIR}")

# add the VisualizeKeyframeUpdate
add_executable(VisualizeKeyframeUpdate
        visualize_keyframe_update.cpp
        DynamicCalibration.hpp
        lib/Utils.hpp
        lib/OpticalFlow.h
        lib/BackgroundSegmentation.h
        )

target_link_libraries(VisualizeKeyframeUpdate PUBLIC libCameraStabilization CameraStabilization ImageUtils ${OpenCV_LIBS})
target_include_directories(VisualizeKeyframeUpdate PUBLIC "${PROJECT_BINARY_DIR}")

# add the VisualizeKeyframeUpdate
add_executable(VisualizeFeatureMatching
        visualize_feature_matching.cpp
        DynamicCalibration.hpp
        lib/Utils.hpp
        lib/OpticalFlow.h
        lib/BackgroundSegmentation.h
        )

target_link_libraries(VisualizeFeatureMatching PUBLIC libCameraStabilization CameraStabilization ImageUtils ${OpenCV_LIBS})
target_include_directories(VisualizeFeatureMatching PUBLIC "${PROJECT_BINARY_DIR}")

########################################################################################################################
### TESTING ###
########################################################################################################################
enable_testing()

add_executable(Tests tests/tests.cpp)
target_link_libraries(Tests PUBLIC libCameraStabilization gtest_main ${OpenCV_LIBS})
target_include_directories(Tests PUBLIC "${PROJECT_BINARY_DIR}")

add_test(AllTestsInFoo Tests)
configure_file(tests/feature_detection_test_image.png tests/feature_detection_test_image.png COPYONLY)
