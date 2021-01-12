# add the main library
add_library(libCameraStabilization
        FeatureDetection.cpp
        DynamicStabilization.cpp
        FeatureMatching.cpp FeatureMatching.hpp)

target_include_directories(libCameraStabilization
        INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
        PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
        )
