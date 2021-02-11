import os
import numpy as np

from src.feature_matching import contour_matching, calculate_homography
from src.image_providers import ImageBasedVideoCapture
from src.image_writer import ImageWriter
from src.landmark_detection import FilterType, detect_road_markings
from src.optical_flow import DenseOpticalFlow
from src.rendering import Renderer, layout
from src.background_segmentation import *

if __name__ == '__main__':
    bridge = 's40_n'
    near_or_far = 'far'
    base_path = '/mnt/local_data/providentia/test_recordings/images/{}_cam_{}'.format(bridge, near_or_far)
    # base_path = '/mnt/local_data/providentia/on_site/2020_12_08/images/{}_cam_{}'.format(bridge, near_or_far)

    cap = ImageBasedVideoCapture(os.path.join(base_path, 'stamp'), loop=True, max_loaded_frames=0, frame_rate=25)
    # cap = ImageBasedVideoCapture(os.path.join(base_path, 'image_raw'), file_ending='.bmp', loop=False, max_loaded_frames=0, frame_rate=25)
    image_writer = ImageWriter(os.path.join(base_path, 'stabilized/images'))

    renderer = Renderer()
    optical_flow = DenseOpticalFlow()

    alpha = 0.2
    previous_frame = None

    scale_factor = 2

    keyframe = None
    min_mag, min_ang = np.Inf, np.Inf

    i = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = Frame(frame)
        size = frame.size()

        if previous_frame is None:
            previous_frame = frame.clone()

        if keyframe is None:
            keyframe = frame.clone()

        H = calculate_homography(frame.cpu(), keyframe.cpu())
        final_frame = Frame(cv.warpPerspective(frame.cpu(), H, tuple([keyframe.size()[1], keyframe.size()[0]])))

        results = [frame.clone(), final_frame.clone()]
        positions = [(0, 0), (0, results[0].size()[1])]

        if not renderer.render(results, positions):
            break

        previous_frame = frame
        i += 1
