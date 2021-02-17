cmake_minimum_required(VERSION 3.10)

set(IMAGES
        feature_detection_test_image.png
        leitpfosten.png
        leitpfosten_cropped.png
        physical_camera_sensor_highway.png
        s40_n_cam_far_calibration_test_image.png
        s40_n_cam_near_calibration_test_image.png
        s40_n_cam_far_calibration_test_image_template.png
        )

file(COPY ${IMAGES} DESTINATION ./)

set(OPENDRIVE
        map_snippet.xodr
        )
file(COPY ${OPENDRIVE} DESTINATION ./)