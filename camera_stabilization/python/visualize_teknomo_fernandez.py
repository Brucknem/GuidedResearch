from teknomo_fernandez import *
from filesystem_image_provider import ImageBasedVideoCapture
from rendering import Renderer, resize_frame
from background_segmentation import *
from cuda_utils import to_3_channel_rgb

if __name__ == '__main__':
    np.random.seed(0)
    cap = ImageBasedVideoCapture('/mnt/nextcloud/tum/Master/5. Semester/Guided Research/videos/s40_n_cam_far/stamp',
                                 loop=False)
    cuda_stream = cv.cuda_Stream()
    renderer = Renderer(cuda_stream)

    only_background = True
    teknomo_fernandez = TeknomoFernandez(levels=6, history=200, verbose=False)
    scale_factor = 2

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        teknomo_fernandez.append(frame)
        result = teknomo_fernandez.get_background()
        size = list(np.array(frame.size()) / scale_factor)
        frame = resize_frame(frame, *size)

        if only_background:
            result = resize_frame(result, *size)
            result = np.concatenate([to_cpu_frame(frame), to_cpu_frame(result)], axis=1)
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
