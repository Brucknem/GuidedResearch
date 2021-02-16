########################################################################################################################
### Tiny XML 2 ###
########################################################################################################################

# Download and unpack tinyxml2 at configure time
configure_file(CMakeLists.txt.in tinyxml2-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tinyxml2-download)
if (result)
    message(FATAL_ERROR "CMake step for tinyxml2 failed: ${result}")
endif ()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tinyxml2-download)
if (result)
    message(FATAL_ERROR "Build step for tinyxml2 failed: ${result}")
endif ()

# Add tinyxml2 directly to our build.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/tinyxml2-src
        ${CMAKE_CURRENT_BINARY_DIR}/tinyxml2-build
        EXCLUDE_FROM_ALL)

########################################################################################################################
### Library ###
########################################################################################################################

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Ceres REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

find_package(Boost REQUIRED COMPONENTS program_options)
include_directories(${Boost_INCLUDE_DIRS})

find_package(PROJ)


add_library(libCameraStabilization
        FeatureDetection.cpp
        DynamicStabilization.cpp
        FeatureMatching.cpp
        BackgroundSegmentation.cpp
        FrameWarping.cpp
        OpticalFlow.cpp
        TimeMeasurable.cpp
        CSVWriter.cpp
        Intrinsics.cpp
        CameraPoseEstimation.cpp
        RenderingPipeline.cpp
        Residuals.cpp
        WorldObjects.cpp
        Watersheder.cpp
        HDMap.cpp)

target_include_directories(libCameraStabilization
        INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
        PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
        )

target_link_libraries(libCameraStabilization
        Eigen3::Eigen
        Ceres::ceres
        ${OpenCV_LIBS}
        ${Boost_LIBRARIES}
        ${PROJ_LIBRARIES}
        tinyxml2
        )
