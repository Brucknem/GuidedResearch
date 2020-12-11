import cv2 as cv
import numpy as np
from src.filters import CudaScharrFilter, CudaSobelFilter
from src.frame_utils import Frame
from src.landmark_detection import detect_road_markings, FILTER_TYPES
from src.rendering import Renderer
from src.image_providers import load_frame_from_disk
import signal
from src.utils import shutdown_signal_handler


def filter_by_area(contours: list, threshold_area: int = 100):
    filtered_contours = []
    for cnt in contours:
        filtered_contour = []
        for c in cnt:
            area = cv.contourArea(c)
            if area > threshold_area:
                filtered_contour.append(c)
        filtered_contours.append(filtered_contour)
    return filtered_contours


def filter_by_minAreaRect(contours: list):
    filtered_contours = []
    for cnt in contours:
        filtered_contour = []
        for c in cnt:
            if cv.isContourConvex(c):
                filtered_contour.append(c)

        filtered_contours.append(filtered_contour)
    return filtered_contours


def filter_by_rect(contours: list):
    filtered_contours = []
    for cnt in contours:
        filtered_contour = []
        for c in cnt:
            peri = cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, 0.04 * peri, True)
            length = len(approx)
            if length == 4:
                filtered_contour.append(approx)
        filtered_contours.append(filtered_contour)
    return filtered_contours


def get_centroids(contours: list):
    moments = []
    for cnt in contours:
        filtered_contour = []
        for c in cnt:
            moment = cv.moments(c)
            cx = int(moment['m10'] / moment['m00'])
            cy = int(moment['m01'] / moment['m00'])
            filtered_contour.append(np.array([cx, cy]))
        moments.append(filtered_contour)
    return moments


def filter_by_centroids(contours, centroids, frame, threshold):
    filtered_contours = []

    for contour in zip(contours, centroids):
        inner_contours = []
        for c in zip(contour[0], contour[1]):
            centroid = c[1]
            pixel = frame[centroid[1], centroid[0]]
            if pixel > threshold:
                inner_contours.append(c[0])
        filtered_contours.append(inner_contours)
    return filtered_contours


if __name__ == '__main__':
    signal.signal(signal.SIGINT, shutdown_signal_handler)
    renderer = Renderer()

    image_path = '/mnt/local_data/providentia/test_recordings/images/s40_n_cam_far/stamp/1598434182.074549135.png'
    frame = load_frame_from_disk(image_path)

    size = frame.size()
    original_frame = cv.imread(image_path)
    frame = cv.cvtColor(original_frame, cv.COLOR_BGR2GRAY)

    scale = 1
    delta = 0
    ddepth = cv.CV_16S
    order = 1
    ksize = 3

    original_frame = Frame(original_frame)
    threshold_frame = Frame(frame)
    frame = Frame(cv.Canny(frame, 50, 200, None, 3))

    scharr_filter = CudaScharrFilter(ddepth=ddepth, order=order)
    sobel_filter = CudaSobelFilter(ddepth=ddepth, order=order, ksize=ksize, scale=scale, delta=delta)

    while True:
        if frame is not None:
            frame = original_frame.resize(size).cpu()
            results, contours = detect_road_markings(frame)
            results = [Frame(result) for result in results]
        else:
            frame = frame.resize(size)
            scharr = scharr_filter.apply(frame)
            sobel = sobel_filter.apply(frame)

            results = [
                scharr,
                # sobel,
            ]

            results = [cv.threshold(result.cpu(grayscale=True), 100, 255, 0)[1] for result in results]
            contours = [cv.findContours(result, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0] for result in results]
            filtered_contours = contours
            filtered_contours = filter_by_rect(contours)
            filtered_contours = filter_by_area(filtered_contours, 50)
            filtered_contours = filter_by_minAreaRect(filtered_contours)
            #
            # centroids = get_centroids(filtered_contours)
            #
            # filtered_contours = filter_by_centroids(filtered_contours, centroids, threshold_frame.cpu(grayscale=True),
            #                                         200)

            results = [Frame(cv.drawContours(result[0].cpu(), result[1], -1, (0, 255, 0), 2)) for result in
                       zip([threshold_frame] * len(filtered_contours), filtered_contours)]

            results = [result.resize(size) for result in results]
            # results = [Frame(result) for result in results]

        positions = [(0, results[0].size()[1] * i) for i in range(len(results))]

        if not renderer.render(results, positions, 'red'):
            break
