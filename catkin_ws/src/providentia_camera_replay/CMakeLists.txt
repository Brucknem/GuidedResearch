cmake_minimum_required(VERSION 3.5)
project(providentia_camera_replay)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  sensor_msgs
  std_msgs
  roscpp
  image_transport
)


#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

catkin_package(
#INCLUDE_DIRS include
#LIBRARIES ${PROJECT_NAME}
#CATKIN_DEPENDS cv_bridge rospy sensor_msgs std_msgs roscpp image_transport
 # DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

add_executable(image_converter src/image_converter.cpp)

# Include Ceres and Eigen
find_package(Ceres REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( image_converter ${OpenCV_LIBS} )

target_link_libraries (image_converter Eigen3::Eigen)
target_link_libraries(image_converter Ceres::ceres)

target_link_libraries(image_converter ${catkin_LIBRARIES})
#gadd_dependencies(talker beginner_tutorials_generate_messages_cpp)

target_link_libraries(image_converter ${OpenCV_LIBRARIES})
