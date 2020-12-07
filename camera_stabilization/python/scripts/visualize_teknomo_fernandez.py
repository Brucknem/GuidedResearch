import os

from src.rendering import Renderer
from src.teknomo_fernandez import *
from src.filesystem_image_provider import ImageBasedVideoCapture, ImageWriter
from src.frame_utils import Frame
import cv2 as cv

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
    scale_factor = 1.2
    test_position = tuple([200, 540])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = Frame(frame)

        teknomo_fernandez.append(frame)
        result = teknomo_fernandez.get_background()
        size = frame.size() / scale_factor

        frame.resize(size)

        if only_background:
            for i in range(0):
                result = cv.cuda.bilateralFilter(result.gpu(), 9, 75, 75)
            result.resize(size / 2)

            # background_writer.write(result)
            difference = teknomo_fernandez.testing()
            difference.resize(size / 2)
            # difference_writer.write(difference)

            result = [
                frame,
                result,
                difference
            ]
            positions = [
                (0, 0),
                (0, size[1]),
                ((size / 2)[0], size[1])
            ]
        else:
            foreground, foreground_bitmask, background, background_bitmask = teknomo_fernandez.calculate_teknomo_fernandez_segmentation()

        if not renderer.render(result, positions):
            break
