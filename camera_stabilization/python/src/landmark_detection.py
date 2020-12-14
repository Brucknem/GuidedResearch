import math

import numpy as np
import cv2 as cv
from enum import Enum


class FilterType(Enum):
    NONE = 0
    SCHARR = 1
    SOBEL = 2
    LAPLACIAN = 3
    AVERAGING = 4
    GAUSSIAN_BLUR = 5
    MEDIAN_BLUR = 6
    BILATERAL_BLUR = 7


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
    threshold = 80
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


def filter_image(filter_input, filter_type: FilterType):
    if filter_type == FilterType.NONE:
        return filter_input
    if filter_type == FilterType.LAPLACIAN:
        return cv.convertScaleAbs(cv.Laplacian(filter_input, ddepth=cv.CV_16S, ksize=3, scale=1, delta=0))
    if filter_type == FilterType.AVERAGING:
        return cv.blur(filter_input, (5, 5))
    if filter_type == FilterType.GAUSSIAN_BLUR:
        return cv.GaussianBlur(filter_input, (5, 5), 0)
    if filter_type == FilterType.MEDIAN_BLUR:
        return cv.medianBlur(filter_input, 5)
    if filter_type == FilterType.BILATERAL_BLUR:
        return cv.bilateralFilter(filter_input, d=9, sigmaColor=75, sigmaSpace=75)

    if filter == FilterType.SCHARR:
        grad_x = cv.Scharr(filter_input, ddepth=cv.CV_16S, dx=1, dy=0)
        grad_y = cv.Scharr(filter_input, ddepth=cv.CV_16S, dx=0, dy=1)
    else:
        grad_x = cv.Sobel(filter_input, ddepth=cv.CV_16S, dx=1, dy=0, ksize=3, scale=1, delta=0)
        grad_y = cv.Sobel(filter_input, ddepth=cv.CV_16S, dx=0, dy=1, ksize=3, scale=1, delta=0)
    grad_x = cv.convertScaleAbs(grad_x)
    grad_y = cv.convertScaleAbs(grad_y)
    return cv.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv.LUT(image, table)


def detect_road_markings(frame: np.ndarray, filter_type: FilterType = FilterType.NONE):
    """
        https://towardsdatascience.com/finding-lane-lines-simple-pipeline-for-lane-detection-d02b62e7572b

    :param frame:
    :param filter_type:
    :return:
    """
    results = {'original': frame}

    if len(frame.shape) == 3:
        grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    else:
        grayscale = np.array(frame)
    # results['grayscale'] = np.array(grayscale)
    blurred = grayscale

    # TODO try different
    gamma = 0.5
    gamma_adjusted = adjust_gamma(grayscale, gamma)
    results['gamma adjusted ({})'.format(gamma)] = gamma_adjusted
    colored_image = gamma_adjusted
    #

    hls_frame = cv.cvtColor(frame, cv.COLOR_RGB2HLS)

    # yellow_mask = cv.inRange(hls_frame, np.array([10, 0, 100], dtype=np.uint8), np.array([40, 255, 255], dtype=np.uint8))

    white_mask = cv.inRange(hls_frame, np.array([0, 200, 0], dtype=np.uint8), np.array([200, 255, 255], dtype=np.uint8))
    results['white markings'] = white_mask

    results['hls'] = hls_frame

    white_mask_shadow_hls = cv.inRange(hls_frame, np.array([0, 87, 0], dtype=np.uint8),
                                       np.array([50, 110, 200], dtype=np.uint8))
    results['white markings (shadow - hls)'] = white_mask_shadow_hls
    # white_mask_shadow_hls_erode = cv.erode(white_mask_shadow_hls, (5,5),iterations=15)
    # results['white markings (shadow - erode)'] = white_mask_shadow_hls_erode

    # white_mask_shadow_hls_close = cv.morphologyEx(white_mask_shadow_hls_erode, cv.MORPH_CLOSE, np.ones((15,15),np.uint8))
    # results['white markings (shadow - close)'] = white_mask_shadow_hls_close

    mask = cv.bitwise_or(white_mask, white_mask_shadow_hls)
    results['white markings (total - hls)'] = mask

    # # -----Converting image to LAB Color model-----------------------------------
    # lab_frame = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
    # results['lab'] = lab_frame
    # white_mask_shadow_lab = cv.inRange(lab_frame, np.array([80, 0, 0], dtype=np.uint8),
    #                                    np.array([130, 255, 255], dtype=np.uint8))
    # results['white markings (shadow - lab)'] = white_mask_shadow_lab
    #
    # # -----Converting image to LAB Color model-----------------------------------
    # hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # results['hsv'] = hsv_frame
    # white_mask_shadow_hsv = cv.inRange(hsv_frame, np.array([0, 0, 90], dtype=np.uint8),
    #                                    np.array([255, 255, 140], dtype=np.uint8))
    # results['white markings (shadow - hsv)'] = white_mask_shadow_hsv
    # mask = cv.bitwise_or(white_mask, white_mask_shadow_hsv)
    # results['white markings (total - hsv)'] = mask

    # return results, []

    # results['yellow markings'] = yellow_mask
    #
    # mask = cv.bitwise_or(yellow_mask, white_mask)
    # # results['total markings'] = mask
    # colored_image = cv.bitwise_and(gamma_adjusted, gamma_adjusted, mask=mask)
    # results['masked'] = colored_image

    # blurred = cv.GaussianBlur(colored_image, (7,7), 0)
    # results['blurred'] = blurred

    # TODO Try different
    canny = cv.Canny(mask, threshold1=50, threshold2=200, edges=None, apertureSize=3)
    results['canny'] = canny

    filtered = canny
    if filter_type:
        filtered = filter_image(canny, filter_type)

    results['filtered ({})'.format(filter_type.name)] = filtered

    canny_stacked = np.stack((canny,) * 3, axis=2)
    contours, hierarchy = cv.findContours(filtered, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours_drawn = cv.drawContours(canny_stacked.copy(), contours, -1, (0, 0, 255), 2)
    results['contours (raw)'] = contours_drawn

    contours = filter_contours(contours, grayscale)
    contours_drawn = cv.drawContours(canny_stacked.copy(), contours, -1, (0, 255, 0), 2)
    results['contours (filtered)'] = contours_drawn

    # lines = cv.HoughLines(filtered, 0.5, np.pi / (180 * 2), 150, None, 0, 0)
    # linesImage = canny_stacked.copy()
    # if lines is not None:
    #     for i in range(0, len(lines)):
    #         rho = lines[i][0][0]
    #         theta = lines[i][0][1]
    #         a = math.cos(theta)
    #         b = math.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    #         pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    #         linesImage = cv.line(linesImage, pt1, pt2, (0,0,255), 2, cv.LINE_AA)
    # results['hough lines'] = linesImage
    #
    # linesP = cv.HoughLinesP(filtered, 1, np.pi / 180, 50, None, 10, 5)
    # linesPimage = canny_stacked.copy()
    # if linesP is not None:
    #         for i in range(0, len(linesP)):
    #             l = linesP[i][0]
    #             linesPimage = cv.line(linesPimage, (l[0], l[1]), (l[2], l[3]), (255,0,0), 2, cv.LINE_AA)
    # results['hough lines (probabilistic)'] = linesPimage
    return results, contours
