# add the main library
add_library(CameraStabilization
            dynamic_stabilization.cpp
            )

target_include_directories(CameraStabilization
          INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
          PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
          )
