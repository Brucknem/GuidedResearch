import cv2 as cv
from cuda_utils import to_gpu_frame, to_cpu_frame


class Algorithm:
    """
    Abstract base class for all algorithms.

    Attributes:
        cuda_stream:    Cuda stream object to perform async GPU computation
    """
    cuda_stream = cv.cuda_Stream()

    def __init__(self, name: str, algorithm: object, is_gpu_algorithm: bool = False, learning_rate: float = 0.1):
        """
        constructor

        :param name: The name of the algorithm
        :param algorithm: The algorithm
        :param learning_rate: The learning rate for GPU algorithms
        """
        self.name = name
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.is_gpu_algorithm = is_gpu_algorithm

    def apply(self, frame: object):
        """
        Applies the algorithm to the given frame.

        :param frame: The frame to apply the algorithm to
        :return The resulting frame
        """
        if self.is_gpu_algorithm:
            result = self.apply_gpu(frame)
        else:
            result = self.apply_cpu()

        return result

    def apply_gpu(self, frame: object):
        """
        Applies the algorithm to the frame using the GPU.

        :param frame: The frame to process
        :return: The resulting frame
        """
        gpu_frame = to_gpu_frame(frame)
        return self.algorithm.apply(gpu_frame, self.learning_rate, Algorithm.cuda_stream)

    def apply_cpu(self, frame: object):
        """
        Applies the algorithm to the frame using the CPU.

        :param frame: The frame to process
        :return: The resulting frame
        """
        cpu_frame = to_cpu_frame(frame)
        return self.algorithm.apply(cpu_frame)


class MOGAlgorithm(Algorithm):
    """

    """

    def __init__(self, learning_rate: float = 0.1, **kwargs):
        Algorithm.__init__(self, 'MOG', cv.cuda.createBackgroundSubtractorMOG(**kwargs), True, learning_rate)


class MOG2Algorithm(Algorithm):
    """
    Zoran Zivkovic.
    Improved adaptive gaussian mixture model for background subtraction.
    In Pattern Recognition, 2004. ICPR 2004.
    Proceedings of the 17th International Conference on, volume 2, pages 28–31. IEEE, 2004.
    """

    def __init__(self, learning_rate: float = 0.1, **kwargs):
        Algorithm.__init__(self, 'MOG2', cv.cuda.createBackgroundSubtractorMOG2(**kwargs), True, learning_rate)
