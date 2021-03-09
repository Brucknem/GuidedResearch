cmake_minimum_required(VERSION 3.10)

set(IMAGES
        leitpfosten.png
        leitpfosten_cropped.png
        physical_camera_sensor_highway.png
        test_frame.png
        )

file(COPY ${IMAGES} DESTINATION ./)

configure_file(objects.yaml objects.yaml COPYONLY)
configure_file(pixels.yaml pixels.yaml COPYONLY)
