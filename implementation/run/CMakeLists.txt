cmake_minimum_required(VERSION 3.10)

########################################################################################################################
### Packages ###
########################################################################################################################
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Boost REQUIRED COMPONENTS program_options)
include_directories(${Boost_INCLUDE_DIRS})

########################################################################################################################
### Executables ###
########################################################################################################################

add_executable(VisualizeFeatureDetection VisualizeFeatureDetection.cpp Commons.cpp)
target_link_libraries(VisualizeFeatureDetection PUBLIC libCameraStabilization ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_include_directories(VisualizeFeatureDetection PUBLIC "${PROJECT_BINARY_DIR}")

add_executable(VisualizeFeatureMatching VisualizeFeatureMatching.cpp Commons.cpp)
target_link_libraries(VisualizeFeatureMatching PUBLIC libCameraStabilization ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_include_directories(VisualizeFeatureMatching PUBLIC "${PROJECT_BINARY_DIR}")

add_executable(VisualizeBackgroundSegmentation VisualizeBackgroundSegmentation.cpp Commons.cpp)
target_link_libraries(VisualizeBackgroundSegmentation PUBLIC libCameraStabilization ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_include_directories(VisualizeBackgroundSegmentation PUBLIC "${PROJECT_BINARY_DIR}")

add_executable(VisualizeDynamicStabilization VisualizeDynamicStabilization.cpp Commons.cpp)
target_link_libraries(VisualizeDynamicStabilization PUBLIC libCameraStabilization ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_include_directories(VisualizeDynamicStabilization PUBLIC "${PROJECT_BINARY_DIR}")

add_executable(VisualizeOpticalFlow VisualizeOpticalFlow.cpp Commons.cpp)
target_link_libraries(VisualizeOpticalFlow PUBLIC libCameraStabilization ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_include_directories(VisualizeOpticalFlow PUBLIC "${PROJECT_BINARY_DIR}")

add_executable(VisualizeRenderingPipeline VisualizeRenderingPipeline.cpp Commons.cpp)
target_link_libraries(VisualizeRenderingPipeline PUBLIC libCameraStabilization ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_include_directories(VisualizeRenderingPipeline PUBLIC "${PROJECT_BINARY_DIR}")

add_executable(VisualizeWatersheder VisualizeWatersheder.cpp Commons.cpp)
target_link_libraries(VisualizeWatersheder PUBLIC libCameraStabilization ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_include_directories(VisualizeWatersheder PUBLIC "${PROJECT_BINARY_DIR}")

add_executable(VisualizeHDMapParser VisualizeHDMapParser.cpp Commons.cpp)
target_link_libraries(VisualizeHDMapParser PUBLIC libCameraStabilization ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_include_directories(VisualizeHDMapParser PUBLIC "${PROJECT_BINARY_DIR}")
