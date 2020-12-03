import os

from teknomo_fernandez import *
from filesystem_image_provider import ImageBasedVideoCapture, ImageWriter
from rendering import Renderer, resize_frame, add_circle, add_text
from background_segmentation import *
from cuda_utils import to_3_channel_rgb
import cv2 as cv


if __name__ == '__main__':
    np.random.seed(0)
    base_path = '/mnt/nextcloud/tum/Master/5. Semester/Guided Research/videos/s40_n_cam_far/'
    cap = ImageBasedVideoCapture(os.path.join(base_path, 'stamp'), loop=True, max_loaded_frames=0, frame_rate=25)
    background_writer = ImageWriter(os.path.join(base_path, 'backgrounds'))
    difference_writer = ImageWriter(os.path.join(base_path, 'difference'))

    cuda_stream = cv.cuda_Stream()
    renderer = Renderer(cuda_stream)

    only_background = True
    teknomo_fernandez = TeknomoFernandez(levels=6, history=200, verbose=False)
    scale_factor = 2
    test_position = tuple([200, 540])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # timer.add_timestamp('fps')
        # duration = timer.get_durations()['fps'].total_seconds()
        # add_text(frame, '{0:.2f} ms ({1:.2f} fps)'.format(duration, 1. / duration))
        #
        # if not renderer.render(frame):
        #     break
        # continue

        teknomo_fernandez.append(frame)
        result = teknomo_fernandez.get_background()
        size = list(np.array(to_gpu_frame(frame).size()) / scale_factor)
        frame = resize_frame(frame, *size)

        if only_background:
            for i in range(0):
                result = cv.cuda.bilateralFilter(to_gpu_frame(result), 9, 75, 75)
            result = resize_frame(result, *size)

            # background_writer.write(result)

            frame = to_cpu_frame(frame)
            result = to_cpu_frame(result)
            difference = to_cpu_frame(teknomo_fernandez.testing())

            difference = resize_frame(difference, *size)
            # difference_writer.write(difference)

            images = {'frame': frame, 'result': result, 'difference': difference}

            result = np.concatenate(list(images.values()), axis=1)
            # result = np.concatenate([frame, result], axis=1)
        else:
            foreground, foreground_bitmask, background, background_bitmask = teknomo_fernandez.calculate_teknomo_fernandez_segmentation()

            result = resize_frame(result, *size)
            foreground = resize_frame(foreground, *size)
            foreground_bitmask = resize_frame(foreground_bitmask, *size)
            background = resize_frame(background, *size)
            background_bitmask = resize_frame(background_bitmask, *size)

            # result = np.concatenate([to_cpu_frame(frame), to_cpu_frame(result)], axis=0)
            # foreground = np.concatenate([to_cpu_frame(to_3_channel_rgb(foreground)), to_cpu_frame(to_3_channel_rgb(foreground_bitmask))], axis=0)
            # background = np.concatenate([to_cpu_frame(to_3_channel_rgb(background)), to_cpu_frame(to_3_channel_rgb(background_bitmask))], axis=0)

            # result = np.concatenate([result, foreground, ], axis=1)
            result = np.concatenate([to_cpu_frame(frame), to_3_channel_rgb(to_cpu_frame(foreground)), ], axis=1)

        if not renderer.render(result):
            break