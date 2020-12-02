import cv2 as cv
import numpy as np

from cuda_utils import invert, multiply, to_3_channel_rgb, multiply_scalar
from teknomo_fernandez import *
from filesystem_image_provider import ImageBasedVideoCapture
from rendering import Renderer, add_text, resize_frame, add_circle
from background_segmentation import *
import pandas as pd


if __name__ == '__main__':
    np.random.seed(0)
    cap = ImageBasedVideoCapture('/mnt/nextcloud/tum/Master/5. Semester/Guided Research/videos/s40_n_cam_far/stamp', loop=False)
    cuda_stream = cv.cuda_Stream()
    renderer = Renderer(cuda_stream)

    teknomo_fernandez = TeknomoFernandez(levels=6, threads=24, history=100)
    scale_factor = 2

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        teknomo_fernandez.add_frame(frame)

        if i % 50 == 0:
            teknomo_fernandez.calculate()
        i += 1
        result = teknomo_fernandez.get_background()

        size = frame.size()
        size = np.array(size) / scale_factor
        frame = resize_frame(frame, *list(size))
        result = resize_frame(result, *list(size))

        result = np.concatenate([to_cpu_frame(frame), to_cpu_frame(result)], axis=1)

        if not renderer.render(result):
            break
