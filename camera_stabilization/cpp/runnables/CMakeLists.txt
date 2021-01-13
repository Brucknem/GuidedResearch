# add the main library
add_library(libRunnables RunnablesCommons.cpp)

target_include_directories(libRunnables
        INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
        PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
        )
