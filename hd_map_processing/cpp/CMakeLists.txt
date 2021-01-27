cmake_minimum_required(VERSION 3.10)

project(HDMapProcessing C CXX)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

#find_package(PROJ)

add_executable(Main main.cpp)

add_subdirectory(ticpp)

target_link_libraries(Main ${PROJ4_LIBRARIES} ticpp)
include_directories(${PROJ4_INCLUDE_DIRS})
target_include_directories(Main PUBLIC "${PROJECT_BINARY_DIR}")
