import os
import sys

import cv2 as cv
import numpy as np
from src.frame_utils import Frame
from src.image_providers import ImageBasedVideoCapture, SingleFrameVideoCapture
from src.landmark_detection import detect_road_markings, FilterType
from src.rendering import Renderer, layout
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

    filter_types = list(FilterType)
    # filter_types = [FilterType.NONE]

    written = False
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = Frame(frame).clone()
        size = frame.size()

        results = []
        positions = []
        for filter_type in filter_types:
            new_results, contours = detect_road_markings(frame.cpu(), filter_type)
            results += [Frame(np.array(result[1])).add_text(result[0], color='cyan', thickness=2) for result in
                        new_results.items()]
            positions += [np.array([results[0].size()[0] * filter_type.value, results[0].size()[1] * i]) for i in
                          range(len(new_results))]

            if not written:
                filter_type_image_writer = ImageWriter(
                    os.path.join(image_writer.base_path, filter_type.name))
                i = 0
                for result in new_results.items():
                    filter_type_image_writer.write(Frame(result[1]), name='{:04d}_{}.png'.format(i, result[0]))
                    i += 1

        # results = [results[0], results[-1]]

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
