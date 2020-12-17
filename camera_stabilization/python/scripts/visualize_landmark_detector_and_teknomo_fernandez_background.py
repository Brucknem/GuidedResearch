import os
import sys

import cv2 as cv
import numpy as np
from src.frame_utils import Frame
from src.image_providers import ImageBasedVideoCapture, SingleFrameVideoCapture
from src.landmark_detection import detect_road_markings, FilterType
from src.rendering import Renderer, layout
import signal

from src.teknomo_fernandez import TeknomoFernandez
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

    cap = ImageBasedVideoCapture(os.path.join(base_path, 'stamp'), loop=False, max_loaded_frames=0, frame_rate=25)
    # cap = SingleFrameVideoCapture(image_path)
    teknomo_fernandez = TeknomoFernandez(levels=6, history=200, verbose=False)

    scale = 2

    written = False
    i = 0
    while True:
        written = True
        ret, frame = cap.read()
        if not ret:
            break

        frame = Frame(frame).clone()
        size = frame.size()

        teknomo_fernandez.append(frame)
        background = teknomo_fernandez.get_background()

        results, contours = detect_road_markings(frame.cpu(), filter_type=FilterType.LAPLACIAN)
        results = [Frame(result[1]) for result in results.items()]
        results = [results[0], results[-1]]
        positions = [np.array([0, results[0].size()[1] * i]) for i in range(len(results))]

        background_results, contours = detect_road_markings(background.cpu(), filter_type=FilterType.LAPLACIAN)
        background_results = [Frame(result[1]) for result in background_results.items()]
        results += [background_results[0], background_results[-1]]
        positions += [np.array([results[0].size()[0], results[0].size()[1] * i]) for i in range(len(results))]

        if not written:
            layouted = layout(results, positions)
            image_writer.write(layouted, name='all.png')

        results = [result.resize(size / scale) for result in results]
        positions = [position / scale for position in positions]

        results[0].add_text('{}/{}'.format(i % cap.get_num_frames(), cap.get_num_frames()), position=(10, 80))
        i += 1

        if not renderer.render(results, positions, 'red'):
            break

        written = cap.get_num_frames() == 1
