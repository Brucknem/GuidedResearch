import cv2 as cv
import numpy as np

from dataframe_writer import DataframeWriter
from teknomo_fernandez import *
from filesystem_image_provider import ImageBasedVideoCapture
from optical_flow import DenseOpticalFlow
from rendering import Renderer
from background_segmentation import *
import pandas as pd


if __name__ == '__main__':
    np.random.seed(0)
    cap = ImageBasedVideoCapture('/mnt/nextcloud/tum/Master/5. Semester/Guided Research/videos/s40_n_cam_far/stamp', loop=False)
    cuda_stream = cv.cuda_Stream()

    renderer = Renderer()
    optical_flow = DenseOpticalFlow()

    alpha = 0.2
    previous_frame = None

    optical_flow_rois = [
        np.array([100, 100]),
        np.array([1450, 110]),
        np.array([1800, 750])
    ]
    optical_flow_roi_colors = ['red', 'white', 'yellow']
    dataframeWriter = DataframeWriter('optical_flows')

    scale_factor = 2

    i = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = Frame(frame)
        size = frame.size()

        if previous_frame is None:
            previous_frame = frame.clone()

        frame.resize(size / scale_factor)

        # result = optical_flow.apply_cpu(frame)
        result = optical_flow.apply_gpu(frame)
        result.resize(size)
        frame.resize(size)

        result.blend(previous_frame, 0.2)
        result.add_text('{}/{}'.format(i, cap.num_frames), position=(10, 80))
        for roi, color in zip(optical_flow_rois, optical_flow_roi_colors):
            result.add_circle(roi, color)
            value = optical_flow.get_flow_value(roi[1] / scale_factor, roi[0] / scale_factor)

            dataframeWriter.append(roi, **value)

        dataframeWriter.append('mean', **optical_flow.get_flow_means())

        if not renderer.render(result):
            break

        previous_frame = frame
        i += 1
