cmake_minimum_required(VERSION 3.10)

########################################################################################################################
### Packages ###
########################################################################################################################
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Boost REQUIRED COMPONENTS program_options REQUIRED filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})


########################################################################################################################
### Utils lib ###
########################################################################################################################

add_library(Utils OpticalFlow.cpp CSVWriter.cpp ObjectTracking.cpp Commons.cpp)

target_include_directories(Utils
        INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
        PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
        )

target_link_libraries(Utils ${OpenCV_LIBS} ${Boost_LIBRARIES} DynamicStabilization StaticCalibration)

########################################################################################################################
### Visualize ###
########################################################################################################################

add_executable(VisualizeFeatureDetection VisualizeFeatureDetection.cpp)
target_link_libraries(VisualizeFeatureDetection DynamicStabilization StaticCalibration Util ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_include_directories(VisualizeFeatureDetection PUBLIC "${PROJECT_BINARY_DIR}")

add_executable(VisualizeFeatureMatching VisualizeFeatureMatching.cpp)
target_link_libraries(VisualizeFeatureMatching DynamicStabilization StaticCalibration Util ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_include_directories(VisualizeFeatureMatching PUBLIC "${PROJECT_BINARY_DIR}")

add_executable(VisualizeBackgroundSegmentation VisualizeBackgroundSegmentation.cpp)
target_link_libraries(VisualizeBackgroundSegmentation DynamicStabilization StaticCalibration Util ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_include_directories(VisualizeBackgroundSegmentation PUBLIC "${PROJECT_BINARY_DIR}")

add_executable(VisualizeDynamicStabilization VisualizeDynamicStabilization.cpp)
target_link_libraries(VisualizeDynamicStabilization DynamicStabilization StaticCalibration Util ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_include_directories(VisualizeDynamicStabilization PUBLIC "${PROJECT_BINARY_DIR}")

add_executable(VisualizeOpticalFlow VisualizeOpticalFlow.cpp)
target_link_libraries(VisualizeOpticalFlow DynamicStabilization StaticCalibration Util ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_include_directories(VisualizeOpticalFlow PUBLIC "${PROJECT_BINARY_DIR}")

add_executable(VisualizeRenderingPipeline VisualizeRenderingPipeline.cpp)
target_link_libraries(VisualizeRenderingPipeline DynamicStabilization StaticCalibration Util ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_include_directories(VisualizeRenderingPipeline PUBLIC "${PROJECT_BINARY_DIR}")

add_executable(VisualizeStaticCalibration VisualizeStaticCalibration.cpp)
target_link_libraries(VisualizeStaticCalibration DynamicStabilization StaticCalibration Util ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_include_directories(VisualizeStaticCalibration PUBLIC "${PROJECT_BINARY_DIR}")


########################################################################################################################
### Evaluation ###
########################################################################################################################

add_executable(EvaluateDynamicStabilization EvaluateDynamicStabilization.cpp)
target_link_libraries(EvaluateDynamicStabilization DynamicStabilization StaticCalibration Utils ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_include_directories(EvaluateDynamicStabilization PUBLIC "${PROJECT_BINARY_DIR}")

add_executable(EvaluateObjectTracking EvaluateObjectTracking.cpp)
target_link_libraries(EvaluateObjectTracking DynamicStabilization StaticCalibration Utils ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_include_directories(EvaluateObjectTracking PUBLIC "${PROJECT_BINARY_DIR}")
