# packages
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# add the main library
add_library(ImageUtils
            video_capture.cpp
            )

target_include_directories(ImageUtils
          INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
          PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
          )

target_link_libraries(ImageUtils PUBLIC ${OpenCV_LIBS})
