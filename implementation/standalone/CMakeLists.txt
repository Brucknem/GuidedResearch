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
add_executable(Watershed Watershed.cpp)
target_link_libraries(Watershed PUBLIC libCameraStabilization ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_include_directories(Watershed PUBLIC "${PROJECT_BINARY_DIR}")
