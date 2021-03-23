cmake_minimum_required(VERSION 3.10)

set(IMAGES
        leitpfosten.png
        leitpfosten_cropped.png
        physical_camera_sensor_highway.png
        test_frame.png
        )

file(COPY ${IMAGES} DESTINATION ./)

configure_file(objects.yaml objects.yaml COPYONLY)
configure_file(pixels_sorted_by_y.yaml pixels_sorted_by_y.yaml COPYONLY)
configure_file(pixels_min_num.yaml pixels_min_num.yaml COPYONLY)
configure_file(pixels_only_bottom.yaml pixels_only_bottom.yaml COPYONLY)
configure_file(pixels_only_top_bottom.yaml pixels_only_top_bottom.yaml COPYONLY)
configure_file(pixels_only_top.yaml pixels_only_top.yaml COPYONLY)
configure_file(pixels_undetermined.yaml pixels_undetermined.yaml COPYONLY)
