# add the main lib

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Ceres REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

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
        Residuals.cpp)

target_include_directories(libCameraStabilization
        INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
        PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
        )

target_link_libraries(libCameraStabilization Eigen3::Eigen Ceres::ceres ${OpenCV_LIBS})
