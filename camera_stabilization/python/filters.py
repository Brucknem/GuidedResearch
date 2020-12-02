from abc import ABC, abstractmethod

import cv2 as cv
import numpy as np

from cuda_utils import to_gpu_frame, to_cpu_frame

SHAPE = 'shape'
KSIZE = 'ksize'
ITERATIONS = 'iterations'


class Filter(ABC):
    @abstractmethod
    def apply(self, frame):
        pass


class GrayScale(Filter):
    def apply(self, frame):
        try:
            return cv.cvtColor(to_cpu_frame(frame), cv.COLOR_BGR2GRAY)
        except:
            return to_cpu_frame(frame)


class WhereFilter(Filter):
    def __init__(self, threshold: float, min: float, max: float):
        self.grayscale = GrayScale()
        self.threshold = threshold
        self.min = min
        self.max = max

    def apply(self, frame):
        _frame = self.grayscale.apply(frame)
        whered = np.where(_frame < self.threshold, self.min, self.max)
        return np.array(whered, dtype=np.uint8)



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
        self.grayscale = GrayScale()

    def apply(self, frame):
        _frame = to_gpu_frame(self.grayscale.apply(frame))
        result = _frame.clone()
        self.filter.apply(_frame, result, CudaFilter.cuda_stream)
        return result


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

    def apply(self, frame):
        return cv.cuda.bilateralFilter(to_gpu_frame(frame), self.diameter, self.sigma_color, self.sigma_space,
                                       stream=CudaFilter.cuda_stream)
