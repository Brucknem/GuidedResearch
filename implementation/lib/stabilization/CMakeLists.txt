########################################################################################################################
### Library ###
########################################################################################################################

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Ceres REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

find_package(Boost REQUIRED COMPONENTS program_options)
include_directories(${Boost_INCLUDE_DIRS})

########################################################################################################################
### Dynamic Stabilization ###
########################################################################################################################
add_library(DynamicStabilization
        FeatureDetection.cpp
        DynamicStabilizationBase.cpp
        SURFBFDynamicStabilizer.cpp
        ORBBFDynamicStabilizer.cpp
        FastFREAKBFDynamicStabilizer.cpp
        FeatureMatching.cpp
        MOG2BackgroundSegmentation.cpp
        BackgroundSegmentionBase.cpp
        FrameWarping.cpp
        TimeMeasurable.cpp
        )

target_include_directories(DynamicStabilization
        INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
        PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
        )

target_link_libraries(DynamicStabilization
        ${OpenCV_LIBS}
        ${Boost_LIBRARIES}
        )
