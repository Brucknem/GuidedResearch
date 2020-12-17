import os

from src.rendering import Renderer
from src.teknomo_fernandez import *
from src.image_providers import ImageBasedVideoCapture
from src.image_writer import ImageWriter

from src.frame_utils import Frame
import cv2 as cv
from src.feature_matching import flann

if __name__ == '__main__':
    np.random.seed(0)
    base_path = '/mnt/nextcloud/tum/Master/5. Semester/Guided Research/videos/s40_n_cam_far/'
    cap = ImageBasedVideoCapture(os.path.join(base_path, 'stamp'), loop=True, max_loaded_frames=0, frame_rate=25)
    background_writer = ImageWriter(os.path.join(base_path, 'backgrounds'))
    difference_writer = ImageWriter(os.path.join(base_path, 'difference'))

    cuda_stream = cv.cuda_Stream()
    renderer = Renderer()

    only_background = True
    teknomo_fernandez = TeknomoFernandez(levels=6, history=200, verbose=False)
    scale_factor = 2

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = Frame(frame)

        teknomo_fernandez.append(frame)
        background = teknomo_fernandez.get_background()
        size = frame.size()

        if only_background:
            for i in range(0):
                background = cv.cuda.bilateralFilter(background.gpu(), 9, 75, 75)
            # result.resize(size / 2)

            # background_writer.write(result)
            # difference = teknomo_fernandez.testing()
            # difference.resize(size / 2)
            # difference_writer.write(difference)

            flann_result = Frame(flann(frame.cpu(), background.cpu()))
            flann_size = np.array([size[0], size[1] * 2])

            results = [
                frame.resize(size / scale_factor),
                background.resize(size / scale_factor),
                flann_result.resize(flann_size / scale_factor)
            ]
            positions = [
                (0, 0),
                (0, size[1]),
                (size[0], 0)
            ]
        else:
            foreground, foreground_bitmask, background, background_bitmask = teknomo_fernandez.calculate_teknomo_fernandez_segmentation()

        # results = [result.resize(size / scale_factor) for result in results]
        positions = [np.array(position) / scale_factor for position in positions]
        if not renderer.render(results, positions):
            break
