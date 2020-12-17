from __future__ import print_function
import cv2 as cv
import numpy as np


def flann(img1: np.ndarray, img2: np.ndarray):
    minHessian = 1000
    detector = cv.xfeatures2d_SURF.create(hessianThreshold=minHessian)
    keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(img2, None)
    # -- Step 2: Matching descriptor vectors with a FLANN based matcher
    # Since SURF is a floating-point descriptor NORM_L2 is used
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)
    # -- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.7
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    # -- Draw matches
    img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
    cv.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches,
                   flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imshow('matches', img_matches)
    return img_matches, good_matches, keypoints1, keypoints2


def calculate_homography(img1: np.ndarray, img2: np.ndarray):
    _, good_matches, keypoints1, keypoints2 = flann(img1, img2)

    # -- Localize the object
    obj = np.empty((len(good_matches), 2), dtype=np.float32)
    scene = np.empty((len(good_matches), 2), dtype=np.float32)
    for i in range(len(good_matches)):
        # -- Get the keypoints from the good matches
        obj[i, 0] = keypoints1[good_matches[i].queryIdx].pt[0]
        obj[i, 1] = keypoints1[good_matches[i].queryIdx].pt[1]
        scene[i, 0] = keypoints2[good_matches[i].trainIdx].pt[0]
        scene[i, 1] = keypoints2[good_matches[i].trainIdx].pt[1]
    H, _ = cv.findHomography(obj, scene, cv.RANSAC)
    return H


def contour_matching(contours_1: list, contours_2: list):
    matches = []
    for contour_1 in contours_1:
        for point_1 in contour_1:
            nearest_other = None
            nearest_distance = np.Inf
            perfect_match_found = False
            for contour_2 in contours_2:
                if perfect_match_found:
                    break

                for point_2 in contour_2:
                    thresholded, dist = distance(point_1, point_2)
                    if thresholded < nearest_distance:
                        nearest_distance = dist
                        nearest_other = point_2
                    if thresholded <= 0:
                        perfect_match_found = True
                        break
            if nearest_other is not None:
                matches.append((point_1[0], nearest_other[0]))
    return np.array(matches)


def distance(point_1: np.ndarray, point_2: np.ndarray, threshold: int = 10):
    dist = np.linalg.norm(point_1 - point_2)
    thresholded = dist if dist <= threshold else np.Inf
    return thresholded, dist
