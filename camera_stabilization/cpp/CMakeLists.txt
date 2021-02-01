cmake_minimum_required(VERSION 3.10)

########################################################################################################################
### Google Test ###
########################################################################################################################

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


########################################################################################################################
### Project definition ###
########################################################################################################################
project(CameraStabilization C CXX)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

########################################################################################################################
### Packages ###
########################################################################################################################
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Boost REQUIRED COMPONENTS program_options)
include_directories(${Boost_INCLUDE_DIRS})


########################################################################################################################
### Own libraries ###
########################################################################################################################
add_subdirectory(library)
add_subdirectory(runnables)

########################################################################################################################
### Executables ###
########################################################################################################################
add_executable(VisualizeFeatureDetection runnables/VisualizeFeatureDetection.cpp)
target_link_libraries(VisualizeFeatureDetection PUBLIC libCameraStabilization libRunnables ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_include_directories(VisualizeFeatureDetection PUBLIC "${PROJECT_BINARY_DIR}")

add_executable(VisualizeFeatureMatching runnables/VisualizeFeatureMatching.cpp)
target_link_libraries(VisualizeFeatureMatching PUBLIC libCameraStabilization libRunnables ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_include_directories(VisualizeFeatureMatching PUBLIC "${PROJECT_BINARY_DIR}")

add_executable(VisualizeBackgroundSegmentation runnables/VisualizeBackgroundSegmentation.cpp)
target_link_libraries(VisualizeBackgroundSegmentation PUBLIC libCameraStabilization libRunnables ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_include_directories(VisualizeBackgroundSegmentation PUBLIC "${PROJECT_BINARY_DIR}")

add_executable(VisualizeDynamicStabilization runnables/VisualizeDynamicStabilization.cpp)
target_link_libraries(VisualizeDynamicStabilization PUBLIC libCameraStabilization libRunnables ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_include_directories(VisualizeDynamicStabilization PUBLIC "${PROJECT_BINARY_DIR}")

add_executable(VisualizeOpticalFlow runnables/VisualizeOpticalFlow.cpp)
target_link_libraries(VisualizeOpticalFlow PUBLIC libCameraStabilization libRunnables ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_include_directories(VisualizeOpticalFlow PUBLIC "${PROJECT_BINARY_DIR}")

add_executable(VisualizeStaticCalibration runnables/VisualizeStaticCalibration.cpp)
target_link_libraries(VisualizeStaticCalibration PUBLIC libCameraStabilization libRunnables ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_include_directories(VisualizeStaticCalibration PUBLIC "${PROJECT_BINARY_DIR}")

add_executable(VisualizeWatershed runnables/VisualizeWatershed.cpp)
target_link_libraries(VisualizeWatershed PUBLIC libCameraStabilization libRunnables ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_include_directories(VisualizeWatershed PUBLIC "${PROJECT_BINARY_DIR}")

########################################################################################################################
### TESTING ###
########################################################################################################################
enable_testing()

add_executable(Tests tests/ImageTestBase.cpp tests/CameraTestBase.cpp
        tests/FeatureDetectorTests.cpp
        tests/FeatureMatcherTests.cpp
        tests/DynamicStabiliationTests.cpp
        tests/BackgroundSubtractorTests.cpp
        tests/CameraTests.cpp
        tests/PerspectiveProjectionTests.cpp
        )
target_link_libraries(Tests PUBLIC libCameraStabilization gtest_main ${OpenCV_LIBS})
target_include_directories(Tests PUBLIC tests "${PROJECT_BINARY_DIR}")

add_test(AllTests Tests)
configure_file(tests/feature_detection_test_image.png tests/feature_detection_test_image.png COPYONLY)
