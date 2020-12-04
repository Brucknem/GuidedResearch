from abc import ABC, abstractmethod

import cv2 as cv
import numpy as np

from frame_utils import to_gpu_frame, to_cpu_frame, Frame

SHAPE = 'shape'
KSIZE = 'ksize'
ITERATIONS = 'iterations'


class Filter(ABC):
    @abstractmethod
    def apply(self, frame):
        pass


class WhereFilter(Filter):
    def __init__(self, threshold: int):
        self.threshold = threshold

    def apply(self, frame: Frame) -> Frame:
        return frame.threshold(self.threshold)


class CudaFilter(ABC):
    cuda_stream = cv.cuda_Stream()


class MorphologyFilter(Filter, ABC):
    pass


class CudaMorphologyFilter(MorphologyFilter, CudaFilter):

    def __init__(self, op: int, **kwargs):
        shape = kwargs.get('shape', cv.MORPH_RECT)
        kernel_size = kwargs.get('ksize', (5, 5))
        iterations = kwargs.get('iterations', 1)
        src_type = kwargs.get('srcType', cv.CV_8UC1)

        self.structuring_element = cv.getStructuringElement(shape, kernel_size)
        self.filter = cv.cuda.createMorphologyFilter(op, src_type, self.structuring_element, iterations=iterations)

    def apply(self, frame: Frame) -> Frame:
        result = cv.cuda_GpuMat()
        result = self.filter.apply(frame.gpu(grayscale=True), result, CudaFilter.cuda_stream)
        return Frame(result)


class CudaOpenFilter(CudaMorphologyFilter):
    def __init__(self, **kwargs):
        CudaMorphologyFilter.__init__(self, cv.MORPH_OPEN, **kwargs)


class CudaCloseFilter(CudaMorphologyFilter):
    def __init__(self, **kwargs):
        CudaMorphologyFilter.__init__(self, cv.MORPH_CLOSE, **kwargs)


class CudaDilateFilter(CudaMorphologyFilter):
    def __init__(self, **kwargs):
        CudaMorphologyFilter.__init__(self, cv.MORPH_DILATE, **kwargs)


class CudaErodeFilter(CudaMorphologyFilter):
    def __init__(self, **kwargs):
        CudaMorphologyFilter.__init__(self, cv.MORPH_ERODE, **kwargs)


class CudaBilateralFilter(CudaFilter):
    def __init__(self, **kwargs):
        self.diameter = kwargs.get('diameter', 9)
        self.sigma_color = kwargs.get('sigma_color', 75)
        self.sigma_space = kwargs.get('sigma_space', 75)

    def apply(self, frame: Frame):
        return cv.cuda.bilateralFilter(frame.gpu(), self.diameter, self.sigma_color, self.sigma_space,
                                       stream=CudaFilter.cuda_stream)
