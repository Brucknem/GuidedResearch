from __future__ import print_function

import os

import cv2 as cv
import numpy as np
import argparse
import random as rng

from src.image_providers import SingleFrameVideoCapture

source_window = 'Image'
maxTrackbar = 1000
rng.seed(12345)


def goodFeaturesToTrack_Demo(val):
    maxCorners = max(val, 1)
    # Parameters for Shi-Tomasi algorithm
    qualityLevel = 0.01
    minDistance = 10
    blockSize = 3
    gradientSize = 3
    useHarrisDetector = False
    k = 0.04
    # Copy the source image
    copy = np.copy(frame)
    # Apply corner detection
    corners = cv.goodFeaturesToTrack(src_gray, maxCorners, qualityLevel, minDistance, None, \
                                     blockSize=blockSize, gradientSize=gradientSize,
                                     useHarrisDetector=useHarrisDetector, k=k)
    # Draw corners detected
    print('** Number of corners detected:', corners.shape[0])
    radius = 4
    for i in range(corners.shape[0]):
        cv.circle(copy, (corners[i, 0, 0], corners[i, 0, 1]), radius,
                  (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256)), cv.FILLED)
    # Show what you got
    cv.namedWindow(source_window)
    cv.imshow(source_window, copy)

if __name__ == '__main__':
    base_path = '/mnt/local_data/providentia/test_recordings/images/s40_n_cam_far/'
    image_path = os.path.join(base_path, 'stamp', '1598434218.634034981.png')

    cap = SingleFrameVideoCapture(image_path)

    ret, frame = cap.read()
    src_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Create a window and a trackbar
    cv.namedWindow(source_window)
    maxCorners = 23  # initial threshold
    cv.createTrackbar('Threshold: ', source_window, maxCorners, maxTrackbar, goodFeaturesToTrack_Demo)
    goodFeaturesToTrack_Demo(maxCorners)
    cv.moveWindow(source_window, 0,0)

    cv.waitKey()
