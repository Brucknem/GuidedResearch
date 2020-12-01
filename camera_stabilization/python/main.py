import cv2 as cv
import numpy as np

from cuda_utils import invert, multiply, to_3_channel_rgb, multiply_scalar
from image_segmentation import *
from filesystem_image_provider import VideoCapture
from rendering import Renderer, add_text, resize_frame
from algorithms import *

if __name__ == '__main__':
    np.random.seed(0)
    cap = VideoCapture('/mnt/nextcloud/tum/Master/5. Semester/Guided Research/videos/s40_n_cam_far/stamp')
    cuda_stream = cv.cuda_Stream()

    renderer = Renderer(cuda_stream)

    close_kernel = np.ones((50, 50), np.uint8)
    open_kernel = np.ones((5, 15), np.uint8)
    kernel = np.ones((25, 25), np.uint8)

    filters = [
        cv.cuda.createMorphologyFilter(cv.MORPH_OPEN, 0, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)), iterations=2),
        # cv.cuda.createMorphologyFilter(cv.MORPH_ERODE, 0, cv.getStructuringElement(cv.MORPH_RECT, (5, 5)), iterations=1),
        # cv.cuda.createMorphologyFilter(cv.MORPH_OPEN, 0, cv.getStructuringElement(cv.MORPH_ELLIPSE
        #                                                                           , (5, 5)), iterations=1),
        cv.cuda.createMorphologyFilter(cv.MORPH_DILATE, 0, cv.getStructuringElement(cv.MORPH_RECT, (5, 5)), iterations=9),
        ]

    algorithms = [
        # MOGAlgorithm(history=1000),
        MOG2Algorithm(history=1000, varThreshold=0, detectShadows=False)
    ]

    scale_factor = 2

    results = {}
    background = None

    i = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for algorithm in algorithms:
            result = algorithm.apply(frame)

            for filter in filters:
                filter.apply(result, result)
            result = add_text(result, algorithm.name, 'orange')
            results[algorithm.name] = result

        frame = resize_frame(frame, 1920 / scale_factor, 1200 / scale_factor)

        masks = []
        processed_images = []
        for name, result in results.items():
            result = resize_frame(result, 1920 / scale_factor, 1200 / scale_factor)
            processed_images.append(multiply(multiply_scalar(invert(result), 1. / 255.), frame))

            masks.append(result)

        masks = [to_cpu_frame(mask) for mask in masks]
        masks = np.concatenate(masks, axis=1)
        masks = to_3_channel_rgb(masks)

        processed_images = [to_cpu_frame(image) for image in processed_images]
        result = np.concatenate(processed_images, axis=1)

        result = np.concatenate([result, masks])

        renderer.render(result)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        i = (i + 1) % 50
