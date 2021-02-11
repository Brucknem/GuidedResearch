from __future__ import print_function

import os

import cv2 as cv
import numpy as np
import argparse

from src.image_providers import SingleFrameVideoCapture

source_window = 'Source image'
corners_window = 'Corners detected'
max_thresh = 255


def cornerHarris_demo(val):
    thresh = val
    # Detector parameters
    blockSize = 2
    apertureSize = 3
    k = 0.04
    # Detecting corners
    dst = cv.cornerHarris(src_gray, blockSize, apertureSize, k)
    # Normalizing
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    dst_norm_scaled = cv.convertScaleAbs(dst_norm)
    # Drawing a circle around corners
    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if int(dst_norm[i, j]) > thresh:
                cv.circle(dst_norm_scaled, (j, i), 5, (0), 2)
    # Showing the result
    cv.namedWindow(corners_window)
    cv.imshow(corners_window, dst_norm_scaled)


if __name__ == '__main__':
    base_path = '/mnt/local_data/providentia/test_recordings/images/s40_n_cam_far/'
    image_path = os.path.join(base_path, 'stamp', '1598434218.634034981.png')

    cap = SingleFrameVideoCapture(image_path)

    ret, frame = cap.read()
    src_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Create a window and a trackbar
    thresh = 150  # initial threshold
    cornerHarris_demo(thresh)
    cv.createTrackbar('Threshold: ', corners_window, thresh, max_thresh, cornerHarris_demo)

    cv.moveWindow(corners_window, 0,0)

    cv.waitKey()
