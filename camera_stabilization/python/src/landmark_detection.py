import numpy as np
import cv2 as cv


class FILTER_TYPES:
    SCHARR = 0
    SOBEL = 1
    LAPLACIAN = 2


def is_rectangle(contour: np.ndarray):
    return len(contour) == 4


def is_large_enough(contour: np.ndarray):
    area = cv.contourArea(contour)
    # TODO try different
    threshold = 50
    return area > threshold


def is_convex(contour: np.ndarray):
    return cv.isContourConvex(contour)


def is_centroid_white(contour: np.ndarray, frame: np.ndarray):
    moment = cv.moments(contour)
    cx = int(moment['m10'] / moment['m00'])
    cy = int(moment['m01'] / moment['m00'])
    # TODO try different
    threshold = 150
    pixel = frame[cy, cx]
    return pixel > threshold


def filter_contours(contours: list, frame: np.ndarray):
    wanted_contours = []
    dropout = {'rect': 0, 'large': 0, 'convex': 0, 'white': 0}
    for contour in contours:
        peri = cv.arcLength(contour, True)
        # TODO try different
        approximate_polygon = cv.approxPolyDP(contour, 0.04 * peri, True)

        if not is_rectangle(approximate_polygon):
            dropout['rect'] += 1
            continue

        if not is_large_enough(approximate_polygon):
            dropout['large'] += 1
            continue

        if not is_convex(approximate_polygon):
            dropout['convex'] += 1
            continue

        if not is_centroid_white(approximate_polygon, frame):
            dropout['white'] += 1
            continue

        wanted_contours.append(contour)
    # print(dropout)
    return wanted_contours


def filter_image(filter_input, filter_type: int):
    if filter_type == FILTER_TYPES.LAPLACIAN:
        return cv.convertScaleAbs(cv.Laplacian(filter_input, ddepth=cv.CV_16S, ksize=3, scale=1, delta=0))

    if filter == FILTER_TYPES.SCHARR:
        grad_x = cv.Scharr(filter_input, ddepth=cv.CV_16S, dx=1, dy=0)
        grad_y = cv.Scharr(filter_input, ddepth=cv.CV_16S, dx=0, dy=1)
    else:
        grad_x = cv.Sobel(filter_input, ddepth=cv.CV_16S, dx=1, dy=0, ksize=3, scale=1, delta=0)
        grad_y = cv.Sobel(filter_input, ddepth=cv.CV_16S, dx=0, dy=1, ksize=3, scale=1, delta=0)
    grad_x = cv.convertScaleAbs(grad_x)
    grad_y = cv.convertScaleAbs(grad_y)
    return cv.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)


def detect_road_markings(frame: np.ndarray):
    results = [frame]
    if len(frame.shape) == 3:
        grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    else:
        grayscale = np.array(frame)
    results.append(np.array(grayscale))

    # TODO Try different
    canny = cv.Canny(grayscale, threshold1=50, threshold2=200, edges=None, apertureSize=3)
    results.append(canny)

    filtered = filter_image(canny, FILTER_TYPES.SOBEL)

    contours, hierarchy = cv.findContours(filtered, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours_drawn = cv.drawContours(np.stack((canny, ) * 3, axis=2), contours, -1, (0, 0, 255), 2)
    results.append(contours_drawn)

    contours = filter_contours(contours, grayscale)
    contours_drawn = cv.drawContours(np.stack((canny, ) * 3, axis=2), contours, -1, (0, 255, 0), 2)
    results.append(contours_drawn)

    return results, contours
