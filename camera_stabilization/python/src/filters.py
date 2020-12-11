from abc import ABC, abstractmethod

import cv2 as cv
import numpy as np

from src.frame_utils import Frame, to_cpu_frame, to_gpu_frame

SHAPE = 'shape'
KSIZE = 'ksize'
ITERATIONS = 'iterations'


class Filter(ABC):
    @abstractmethod
    def apply(self, frame):
        pass


class CudaFilter(ABC):
    cuda_stream = cv.cuda_Stream()


class CudaGradientFilter(CudaFilter, ABC):
    def __init__(self, **kwargs):
        self.keys = ['srcType', 'dstType', 'scale', *self.get_keys()]
        self.params = CudaGradientFilter.get_parameters(**kwargs)
        self.filter_x, self.filter_y = self.get_filters()

    def apply(self, frame: Frame):
        gpu_frame = frame.gpu()
        gpu_frame = cv.cuda.cvtColor(gpu_frame, cv.COLOR_BGR2GRAY)
        gpu_frame = to_gpu_frame(np.float32(to_cpu_frame(gpu_frame)))

        grad_x = self.filter_x.apply(gpu_frame, None, CudaFilter.cuda_stream)
        grad_y = self.filter_y.apply(gpu_frame, None, CudaFilter.cuda_stream)

        return self.blend(grad_x, grad_y)

    def get_keys(self):
        return []

    @abstractmethod
    def get_filters(self, **kwargs):
        pass

    @staticmethod
    def blend(grad_x, grad_y) -> Frame:
        abs_grad_x = cv.convertScaleAbs(to_cpu_frame(grad_x))
        abs_grad_y = cv.convertScaleAbs(to_cpu_frame(grad_y))

        return Frame(cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0))

    @staticmethod
    def get_parameters(**kwargs):
        params = kwargs
        params['srcType'] = cv.CV_32FC1
        params['dstType'] = cv.CV_32FC1
        params['dx'] = params.get('order', 1)
        params['dy'] = params.get('order', 1)
        params['ksize'] = params.get('ksize', 3)
        params['scale'] = params.get('scale', 1)
        params['delta'] = params.get('delta', 0)
        return params


class CudaScharrFilter(CudaGradientFilter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_filters(self):
        params = {key: self.params[key] for key in self.keys}
        return cv.cuda.createScharrFilter(dx=self.params['dx'], dy=0, **params),\
               cv.cuda.createScharrFilter(dx=0, dy=self.params['dy'], **params)


class CudaSobelFilter(CudaGradientFilter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_keys(self):
        return ['ksize']

    def get_filters(self):
        params = {key: self.params[key] for key in self.keys}
        return cv.cuda.createSobelFilter(dx=self.params['dx'], dy=0, **params),\
               cv.cuda.createSobelFilter(dx=0, dy=self.params['dy'], **params)


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
