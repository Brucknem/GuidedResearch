import cv2 as cv
from matplotlib import colors

from cuda_utils import to_cpu_frame, is_gpu_frame, to_gpu_frame


def color_to_255_rgb(color: list):
    upper_bound = max(color)
    if upper_bound <= 1:
        color = [int(c * 255) for c in color]
    return color


def add_text(frame, text, color='white'):
    if type(color) is str:
        color = colors.to_rgb(color)
    color = color_to_255_rgb(color)

    is_gpu = is_gpu_frame(frame)
    result = to_cpu_frame(frame)
    cv.putText(result, text, (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv.LINE_AA)

    if is_gpu:
        result = to_gpu_frame(result)
    return result


def add_circle(frame, center: tuple, color='white', **kwargs):
    if type(color) is str:
        color = colors.to_rgb(color)
    color = color_to_255_rgb(color)
    color.reverse()

    is_gpu = is_gpu_frame(frame)
    result = to_cpu_frame(frame)
    cv.circle(result, center, 10, color, **kwargs)

    if is_gpu:
        result = to_gpu_frame(result)
    return result


def resize_frame(frame: object, columns: float, rows: float):
    is_gpu = is_gpu_frame(frame)
    cpu_frame = to_cpu_frame(frame)
    result = cv.resize(cpu_frame, (int(columns), int(rows)))
    if is_gpu:
        result = to_gpu_frame(result)
    return result


def scale_frame(frame: object, scale_factor: float):
    shape = to_cpu_frame(frame).shape() / scale_factor
    return resize_frame(frame, shape[0], shape[1])


class Renderer:
    def __init__(self, cuda_stream: object = None):
        if cuda_stream is not None:
            self.cuda_stream = cuda_stream
        else:
            self.cuda_stream = cv.cuda_Stream()

        self.window_name = 'Camera Visualization'
        # cv.namedWindow(self.window_name, flags=cv.WINDOW_GUI_EXPANDED | cv.WINDOW_AUTOSIZE)

    def render(self, frame: object):
        render_frame = to_cpu_frame(frame)
        cv.imshow(self.window_name, render_frame)
        return not (cv.waitKey(1) & 0xFF == ord('q'))

