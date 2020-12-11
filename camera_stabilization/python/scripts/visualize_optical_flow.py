import os

from src.dataframe_writer import DataframeWriter
from src.teknomo_fernandez import *
from src.image_providers import ImageBasedVideoCapture
from src.image_writer import ImageWriter
from src.optical_flow import DenseOpticalFlow
from src.rendering import Renderer
from src.background_segmentation import *

if __name__ == '__main__':
    np.random.seed(0)

    bridge = 's50_s'
    near_or_far = 'far'
    # base_path = '/mnt/local_data/providentia/test_recordings/images/{}_cam_{}'.format(bridge, near_or_far)
    base_path = '/mnt/local_data/providentia/on_site/2020_12_08/images/{}_cam_{}'.format(bridge, near_or_far)

    # cap = ImageBasedVideoCapture(os.path.join(base_path, 'stamp'), loop=False, max_loaded_frames=0, frame_rate=25)
    cap = ImageBasedVideoCapture(os.path.join(base_path, 'image_raw'), file_ending='.bmp', loop=False, max_loaded_frames=0, frame_rate=25)
    image_writer = ImageWriter(os.path.join(base_path, 'optical_flow/images'))
    dataframeWriter = DataframeWriter(os.path.join(base_path, 'optical_flow/values'))

    print('Open: ' + os.path.join(base_path, 'optical_flow/'))

    cuda_stream = cv.cuda_Stream()

    renderer = Renderer()
    optical_flow = DenseOpticalFlow()

    alpha = 0.2
    previous_frame = None

    if 's40_n' in bridge:
        optical_flow_rois = [
            np.array([100, 100]),
            np.array([1450, 110]),
            np.array([1800, 750]),
            np.array([700, 800]),
        ]
    else:
        if 'far' in near_or_far:
            optical_flow_rois = [
                np.array([100, 100]),
                np.array([1150, 110]),
                np.array([1600, 750]),
                np.array([700, 800]),
            ]
        else:
            optical_flow_rois = [
                np.array([100, 100]),
                np.array([910, 80]),
                np.array([1600, 750]),
                np.array([600, 800]),
            ]

    optical_flow_roi_colors = ['red', 'white', 'yellow', 'green']

    scale_factor = 1

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

        result.blend(previous_frame, 0.3)
        for roi, color in zip(optical_flow_rois, optical_flow_roi_colors):
            result.add_circle(roi, color)
            value = optical_flow.get_flow_value(roi[1] / scale_factor, roi[0] / scale_factor)

            dataframeWriter.append(roi, **value)

        image_writer.write(result.clone())

        result.add_text('{}/{}'.format(i, cap.num_frames), position=(10, 80))

        if not renderer.render(result):
            break

        previous_frame = frame
        i += 1
