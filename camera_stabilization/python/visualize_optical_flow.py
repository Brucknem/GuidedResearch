import cv2 as cv
import numpy as np

from cuda_utils import invert, multiply, to_3_channel_rgb, multiply_scalar
from teknomo_fernandez import *
from filesystem_image_provider import ImageBasedVideoCapture
from optical_flow import DenseOpticalFlow
from rendering import Renderer, add_text, resize_frame, add_circle
from background_segmentation import *
import pandas as pd


if __name__ == '__main__':
    np.random.seed(0)
    cap = ImageBasedVideoCapture('/mnt/nextcloud/tum/Master/5. Semester/Guided Research/videos/s40_n_cam_far/stamp', loop=False)
    cuda_stream = cv.cuda_Stream()

    renderer = Renderer(cuda_stream)
    optical_flow = DenseOpticalFlow()

    alpha = 0.2
    previous_frame = None

    optical_flow_rois = [
        (100, 100),
        (1450, 110),
        (1800, 750)
    ]

    mean_key = ('mean', 0)
    optical_flow_roi_colors = ['red', 'white', 'yellow']
    columns = ['Milliseconds', 'Timestamp', *DenseOpticalFlow.flow_columns]
    optical_flow_roi_dataframes = {roi: pd.DataFrame(columns=columns) for roi in [*optical_flow_rois, mean_key]}

    i = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if previous_frame is None:
            previous_frame = frame

        result = None
        # result = calculate_algorithms(frame, algorithms, filters)
        # result = optical_flow.apply_cpu(frame)
        result = optical_flow.apply_gpu(frame)

        if result is None:
            result = frame

        result = cv.cuda.addWeighted(to_gpu_frame(previous_frame), alpha, to_gpu_frame(result), 1 - alpha, 0)
        result = add_text(result, '{}/{}'.format(i, cap.num_frames))
        for roi, color in zip(optical_flow_rois, optical_flow_roi_colors):
            result = add_circle(result, roi, color)
            value = optical_flow.get_flow_value(roi[1], roi[0]).values()
            value = pd.DataFrame([[*([i * 1000] * 2), *value]], columns=columns)
            optical_flow_roi_dataframes[roi] = optical_flow_roi_dataframes[roi].append(
                value
            )

        value = optical_flow.get_flow_means().values()
        value = pd.DataFrame([[*([i * 1000] * 2), *value]], columns=columns)
        optical_flow_roi_dataframes[mean_key] = optical_flow_roi_dataframes[mean_key].append(
            value
        )
        for roi, df in optical_flow_roi_dataframes.items():
            filename = 'optical_flows/{}_{}.csv'.format(*roi)
            df.to_csv(filename, index=False)

        if not renderer.render(result):
            break

        previous_frame = frame
        i += 1
