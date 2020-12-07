import cv2 as cv
import numpy as np
from matplotlib import colors


def color_to_255_bgr(color: list):
    upper_bound = max(color)
    if upper_bound <= 1:
        color = [int(c * 255) for c in color]
    color.reverse()
    return color


def is_gpu_frame(frame: object):
    """
    Checks if the given frame is a cuda GPU frame
    :param frame: the frame to check
    :return: True if the frame is a GPU frame, false if the frame is a CPU frame, None else
    """
    if str(type(frame)) == '<class \'cv2.cuda_GpuMat\'>':
        return True
    elif type(frame) == np.ndarray:
        return False
    return None


def to_3_channel_rgb(frame: np.ndarray):
    """
    Converts the given frame to 3 channel rgb.

    :return: The frame if it already was 3 channel rgb, the converted frame else
    """
    shape = frame.shape

    if len(shape) == 3 and shape[2] == 3:
        return frame

    result = frame
    if len(shape) == 2:
        result = np.expand_dims(result, axis=2)

    shape = result.shape
    if shape[2] == 1:
        result = np.concatenate([result] * 3, axis=2)
    return np.array(result, dtype=np.uint8)


def invert(frame: object) -> object:
    """
    Inverts the given frame.
    """
    if is_gpu_frame(frame):
        return cv.cuda.bitwise_not(frame)

    return ~frame


def multiply(frame1: object, frame2: object):
    """
    Multiplies two frames.

    :return: A CPU frame if both frames are on CPU, a GPU frame else
    """
    _frame1 = to_3_channel_rgb(frame1)
    _frame2 = to_3_channel_rgb(frame2)
    if not is_gpu_frame(_frame1) and not is_gpu_frame(_frame2):
        return _frame1 * _frame2

    return cv.cuda.multiply(to_gpu_frame(_frame1), to_gpu_frame(_frame2))


def to_gpu_frame(frame: cv.cuda_GpuMat or np.ndarray) -> cv.cuda_GpuMat:
    """
    Converts the given frame to a GPU frame. Keeps the frame if it already is a GPU frame.
    :param frame: the frame to convert
    :return: the frame as a GPU frame
    """
    if is_gpu_frame(frame):
        return frame.clone()

    gpu_frame = cv.cuda_GpuMat()
    gpu_frame.upload(to_cpu_frame(frame))
    return gpu_frame


def to_cpu_frame(frame: cv.cuda_GpuMat or np.ndarray) -> np.ndarray:
    """
    Converts the given frame to a CPU frame. Keeps the frame if it already is a CPU frame.
    :param frame: the frame to convert
    :return: the frame as a CPU frame
    """

    if is_gpu_frame(frame):
        frame = frame.download()
    return frame


def resize_frame(frame: object, columns: float, rows: float):
    is_gpu = is_gpu_frame(frame)
    cpu_frame = to_cpu_frame(frame)
    result = cv.resize(cpu_frame, (int(columns), int(rows)))
    if is_gpu:
        result = to_gpu_frame(result)
    return result


def scale_frame(frame: object, scale_factor: float):
    shape = to_cpu_frame(frame).shape
    return resize_frame(frame, shape[0] / scale_factor, shape[1] / scale_factor)


class Frame:
    @staticmethod
    def empty():
        return Frame.empty_like((1, 1, 3))

    @staticmethod
    def empty_like(shape: np.ndarray or tuple or list):
        return Frame(np.zeros_like(np.ndarray(shape), dtype=np.uint8))

    def __init__(self, frame: cv.cuda_GpuMat or np.ndarray):
        self._cpu: np.ndarray = None
        self._gpu: cv.cuda_GpuMat = None
        self.set(frame)

    def __str__(self):
        return 'Frame ({})'.format(self.shape())

    def __mul__(self, other: float or int or object):
        """
        Multiplies a frame with a scalar.

        :return: The multiplied GPU/CPU frame
        """
        other_type = type(other)
        if other_type is float or other_type is int:
            self.set(cv.convertScaleAbs(self.cpu(), alpha=other))
        else:
            self.set(cv.cuda.multiply(self.gpu(), other.gpu()))
        return self

    def __invert__(self):
        return Frame(cv.cuda.bitwise_not(self.gpu()))

    def has_gpu(self):
        return self._gpu is not None

    def has_cpu(self):
        return self._cpu is not None

    def gpu(self, grayscale: bool = False):
        if not self.has_gpu():
            self.set(to_gpu_frame(self._cpu))

        if grayscale:
            value = cv.cuda_GpuMat()
            value.upload(cv.cvtColor(self.cpu(), cv.COLOR_RGB2GRAY))
        else:
            value = self._gpu
        return value

    def cpu(self, grayscale: bool = False):
        if not self.has_cpu():
            self.set(to_cpu_frame(self._gpu))

        if grayscale:
            value = cv.cvtColor(self.cpu(), cv.COLOR_RGB2GRAY)
        else:
            value = self._cpu
        return value

    def set(self, frame: cv.cuda_GpuMat or np.ndarray):
        is_gpu = is_gpu_frame(frame)
        if is_gpu is None:
            raise ValueError('No valid frame given. [{}]'.format(type(frame)))
        if is_gpu:
            self._gpu = frame.clone()
        else:
            self._cpu = to_3_channel_rgb(to_cpu_frame(frame))
        self.sync(is_gpu)
        return self

    def sync(self, is_gpu: bool):
        if is_gpu and self._cpu is not None:
            self._cpu = to_3_channel_rgb(to_cpu_frame(self._gpu))

        if not is_gpu and self._gpu is not None:
            self._gpu = to_gpu_frame(self._cpu)
        return self

    def subtract(self, other: any, absolute=True):
        result = np.array(self.cpu(), dtype=float) - np.array(other.cpu(), dtype=float)
        if absolute:
            result = np.abs(result)
        else:
            result = np.clip(result, 0, 255)
        return self.set(result)

    def size(self):
        if self.has_cpu():
            size = self._cpu.shape[:2]
        else:
            size = self._gpu.size()
        return np.array(list(size), dtype=int)

    def shape(self):
        return np.array([*self.size(), 3], dtype=int)

    def resize(self, size_or_rows: int or float or tuple or np.ndarray, columns: int or float = -1):
        type_size_or_rows = type(size_or_rows)
        if type_size_or_rows is tuple or type_size_or_rows is np.ndarray:
            columns = size_or_rows[1]
            size_or_rows = size_or_rows[0]

        return self.set(cv.resize(self.cpu(), (int(columns), int(size_or_rows))))

    def add_text(self, text, color: str or tuple or np.ndarray = 'white', position: tuple = (10, 40)):
        if type(color) is str:
            color = colors.to_rgb(color)
        color = color_to_255_bgr(color)
        return self.set(cv.putText(self.cpu(), text, position, cv.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv.LINE_AA))

    def add_circle(self, center: tuple or np.ndarray, color='white', **kwargs):
        if type(color) is str:
            color = colors.to_rgb(color)
        color = color_to_255_bgr(color)
        return self.set(cv.circle(self.cpu(), tuple(center), 10, color, **kwargs))

    def merge(self, other: any or None, position: int = 1):
        other.resize(self.size())
        if position == 0:
            images = [other.cpu(), self.cpu()]
        else:
            images = [self.cpu(), other.cpu()]
        return np.concatenate(images, axis=position % 2)

    @staticmethod
    def merge_frames(self: any, other: any, axis=1):
        if self is None and other is None:
            return Frame.empty()
        if self is None:
            return other
        if other is None:
            return self
        self.merge(other, axis % 2)

    def clone(self):
        return Frame(np.array(self.cpu()))

    def blend(self, other: any, own_alpha: float, other_alpha: float = -1.):
        alpha = other_alpha
        if alpha == -1:
            alpha = 1 - own_alpha
        return self.set(
            cv.cuda.addWeighted(self.gpu(), own_alpha, other.gpu(), 1 - alpha, 0)
        )
