# add the main library
add_library(libRunnables Commons.cpp)

target_include_directories(libRunnables
        INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
        PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
        )
