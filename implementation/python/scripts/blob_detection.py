"""
@file hough_lines.py
@brief This program demonstrates line finding with the Hough transform
"""
import sys
import math
import cv2 as cv
import numpy as np


def main(argv):
    ## [load]
    default_file = '/mnt/local_data/providentia/test_recordings/images/s40_n_cam_far/stamp/1598434182.074549135.png'
    filename = argv[0] if len(argv) > 0 else default_file

    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)
    params = cv.SimpleBlobDetector_Params()

    params.filterByColor = True
    params.blobColor = 2007

    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200

    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.95

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    params.maxInertiaRatio = 0.1

    detector = cv.SimpleBlobDetector_create(params)
    keypoints = detector.detect(src)
    blank = np.zeros((1, 1))
    blobs = cv.drawKeypoints(src, keypoints, blank, (0, 255, 255), cv.DRAW_MATCHES_FLAGS_DEFAULT)

    while True:
        cv.imshow("Source", blobs)
        cv.waitKey()
    return 0

if __name__ == "__main__":
    main(sys.argv[1:])