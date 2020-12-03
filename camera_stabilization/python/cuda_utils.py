import cv2 as cv
import numpy as np


def multiply_scalar(frame: object, scalar: float):
    """
    Multiplies a frame with a scalar.

    :return: The multiplied GPU/CPU frame
    """
    is_gpu = is_gpu_frame(frame)
    result = to_cpu_frame(frame)
    result = cv.convertScaleAbs(result, alpha=scalar)

    if is_gpu:
        result = to_gpu_frame(result)
    return result


def to_3_channel_rgb(frame: object):
    """
    Converts the given frame to 3 channel rgb.

    :return: The frame if it already was 3 channel rgb, the converted frame else
    """
    is_gpu = is_gpu_frame(frame)

    if is_gpu:
        shape = list(frame.size())
        shape.append(frame.channels())
    else:
        shape = frame.shape

    if len(shape) == 3 and shape[2] == 3:
        return frame

    result = to_cpu_frame(frame)
    shape = result.shape
    if len(shape) == 2:
        result = np.expand_dims(result, axis=2)

    shape = result.shape
    if shape[2] == 1:
        result = np.concatenate([result] * 3, axis=2)
    return result


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


def to_gpu_frame(frame: object):
    """
    Converts the given frame to a GPU frame. Keeps the frame if it already is a GPU frame.
    :param frame: the frame to convert
    :return: the frame as a GPU frame
    """
    if is_gpu_frame(frame):
        return frame

    gpu_frame = cv.cuda_GpuMat()
    gpu_frame.upload(frame)
    return gpu_frame


def to_cpu_frame(frame: object):
    """
    Converts the given frame to a CPU frame. Keeps the frame if it already is a CPU frame.
    :param frame: the frame to convert
    :return: the frame as a CPU frame
    """
    if not is_gpu_frame(frame):
        return frame
    return frame.download()


def subtract(frame0, frame1, absolute=True):
    _frame0 = np.array(to_cpu_frame(frame0), dtype=float)
    _frame1 = np.array(to_cpu_frame(frame1), dtype=float)
    if absolute:
        result = np.abs(_frame0 - frame1)
    result = np.array(result, dtype=np.uint8)
    return result
