cmake_minimum_required(VERSION 3.0.2)
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

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(image_converter ${catkin_LIBRARIES})
#gadd_dependencies(talker beginner_tutorials_generate_messages_cpp)

target_link_libraries(image_converter ${OpenCV_LIBRARIES})
