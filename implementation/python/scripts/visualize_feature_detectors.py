import os
import sys

import cv2 as cv
import numpy as np
from src.frame_utils import Frame
from src.image_providers import ImageBasedVideoCapture, SingleFrameVideoCapture
from src.rendering import Renderer, layout
from src.feature_detection import detect_features
import signal
from src.utils import shutdown_signal_handler
from src.image_writer import ImageWriter


if __name__ == '__main__':
    signal.signal(signal.SIGINT, shutdown_signal_handler)
    renderer = Renderer()

    base_path = '/mnt/local_data/providentia/test_recordings/images/s40_n_cam_far/'
    # image_path = os.path.join(base_path, 'stamp', '1598434182.074549135.png')
    image_path = os.path.join(base_path, 'stamp', '1598434218.634034981.png')

    # base_path = '/mnt/local_data/providentia/test_recordings/images/s40_n_cam_far/cutout/raw'
    # image_path = os.path.join(base_path, 'white_markings_with_shadow.png')
    # image_path = os.path.join(base_path, 'white_markings_with_shadow.png')
    image_writer = ImageWriter(os.path.join(base_path, 'landmarks'))

    # cap = ImageBasedVideoCapture(os.path.join(base_path, 'stamp'), loop=False, max_loaded_frames=0, frame_rate=25)
    cap = SingleFrameVideoCapture(image_path)

    scale = 1

    written = False
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = Frame(frame).clone()
        size = frame.size()

        results = detect_features(frame.cpu())
        results = [Frame(result[1]) for result in results.items()]
        positions = [np.array([0, results[0].size()[1] * i]) for i in
                      range(len(results))]

        results = [result.resize(size / scale) for result in results]
        positions = [position / scale for position in positions]

        results[0].add_text('{}/{}'.format(i % cap.get_num_frames(), cap.get_num_frames()), position=(10, 80))
        i += 1

        if not renderer.render(results, positions, 'red'):
            break

        written = cap.get_num_frames() == 1
