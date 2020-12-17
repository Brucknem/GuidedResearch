import numpy as np
import cv2 as cv


def detect_orb_features(frame: np.ndarray):
    orb = cv.ORB_create()

    # find the keypoints with ORB
    kp = orb.detect(frame, None)

    # compute the descriptors with ORB
    return orb.compute(frame, kp)


def detect_harris_corners(frame: np.ndarray):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv.cornerHarris(gray, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    dst = cv.dilate(dst, None)

    result = frame.copy()
    # Threshold for an optimal value, it may vary depending on the image.
    result[dst > 0.01 * dst.max()] = [0, 0, 255]

    return result


def detect_features(frame: np.ndarray):
    kp, _ = detect_orb_features(frame)
    # draw only keypoints location,not size and orientation
    orb_frame = cv.drawKeypoints(frame, kp, None, color=(0, 255, 0), flags=0)
    results = {'orb': orb_frame}

    results['harris'] = detect_harris_corners(frame)

    return results