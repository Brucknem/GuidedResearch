# add the main library
add_library(libCameraStabilization
        FeatureDetection.cpp
        DynamicStabilization.cpp
        FeatureMatching.cpp
        BackgroundSegmentation.cpp
        FrameWarping.cpp
        OpticalFlow.cpp
        TimeMeasurable.cpp
        CSVWriter.cpp
        KeyframeHistory.cpp
        CameraMatrix.cpp
        Camera.cpp)

target_include_directories(libCameraStabilization
        INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
        PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
        )

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries(libCameraStabilization Eigen3::Eigen)