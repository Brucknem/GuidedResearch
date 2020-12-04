import numpy as np
import cv2 as cv
from threading import Thread

from frame_utils import to_3_channel_rgb, Frame
from timable import ITimable
from utils import FixedSizeList
from filters import *
from matplotlib import pyplot as plt


class TeknomoFernandez(ITimable, FixedSizeList):
    """
    Calculates the background of a set of images based on the Teknomo-Fernandez algorithm.

    https://en.wikipedia.org/wiki/Teknomo%E2%80%93Fernandez_algorithm
    """

    def __init__(self, levels: int = 6, history: int = 200, random_history: bool = True,
                 verbose: bool = False, use_gpu: bool = True):
        """
        constructor

        :param levels: The number of levels during the algorithm
        :param history: The number of images in the history buffer to calculate the background from
        :param random_history: Flag if the elements in the history are removed randomly if the buffer is filled
        :param verbose: Print the timings of the steps of the algorithm
        """
        ITimable.__init__(self, name='Teknomo-Fernandez')
        FixedSizeList.__init__(self, max_num_elements=history, remove_random=random_history)
        self.levels = levels
        self.backgrounds_buffer = []
        self.background_calculation_buffer = []
        self.backgrounds = []
        self.verbose = verbose
        self.recalculation_allowed = True
        self.use_gpu = use_gpu

    def append(self, value: Frame) -> None:
        super().append(value.clone())

        self.calculate_async()

    def calculate_background_image_on_history(self, indices: np.ndarray or list):
        """
        Calculates the background image based on the Teknomo-Fernandez algorithm over the image history.

        :return: img3 * (img1 ^ img2) + img1 * img2
        """
        if self.use_gpu:
            model = cv.cuda.bitwise_or(
                cv.cuda.bitwise_and(self[indices[2]].gpu(),
                                    (cv.cuda.bitwise_xor(self[indices[0]].gpu(), self[indices[1]].gpu()))),
                cv.cuda.bitwise_and(self[indices[0]].gpu(), self[indices[1]].gpu()))
        else:
            model = self[indices[2]].cpu() * (self[indices[0]].cpu() ^ self[indices[1]].cpu()) + self[
                indices[0]].cpu() * self[indices[1]].cpu()
        return Frame(model)

    def calculate_background_image_on_background_buffer(self, indices: np.ndarray or list):
        """
        Calculates the background image based on the Teknomo-Fernandez algorithm over the image history.

        :return: img3 * (img1 ^ img2) + img1 * img2
        """
        if self.use_gpu:
            model = cv.cuda.bitwise_or(cv.cuda.bitwise_and(self.background_calculation_buffer[indices[2]].gpu(), (
                cv.cuda.bitwise_xor(self.background_calculation_buffer[indices[0]].gpu(),
                                    self.background_calculation_buffer[indices[1]].gpu()))),
                                       cv.cuda.bitwise_and(self.background_calculation_buffer[indices[0]].gpu(),
                                                           self.background_calculation_buffer[indices[1]].gpu()))
        else:
            model = self.background_calculation_buffer[indices[2]].cpu() * (
                    self.background_calculation_buffer[indices[0]].cpu() ^ self.background_calculation_buffer[
                indices[1]].cpu()) + \
                    self.background_calculation_buffer[indices[0]].cpu() * self.background_calculation_buffer[
                        indices[1]].cpu()
        return Frame(model)

    def get_background(self) -> Frame:
        """
        Gets the latest calculated background image.

        :return: The latest background image if enough images are in the history,
                    The latest added frame if not,
                    An empty frame if no frames are added yet
        """
        if len(self.backgrounds) >= 1:
            result = self.backgrounds[-1]
        elif self.__len__() == 0:
            result = Frame(np.zeros((1, 1, 3)))
        else:
            result = self[-1]
        return result.clone()

    def reset_buffers(self):
        """
        Resets the background calculation buffers.
        """
        self.backgrounds_buffer = []
        self.background_calculation_buffer = []

    def calculation(self):
        """
        Performs calculation of the Teknomo-Fernandez algorithm on the current set of images.
        """
        if len(self) < 3:
            return

        self.recalculation_allowed = False
        self.reset_buffers()

        try:
            self.clear_timestamps()
            indices = [np.random.randint(low=0, high=(len(self) - 1), size=3) for _ in range(3 ** (self.levels - 1))]
            self.background_calculation_buffer = [self.calculate_background_image_on_history(index) for index in indices]
            self.backgrounds_buffer.append(self.background_calculation_buffer[-1])
            # self.add_timestamp('level 1')

            for i in range(2, self.levels + 1):
                indices = [[3 * j, 3 * j + 1, 3 * j + 2] for j in range(3 ** (self.levels - i))]
                results = [self.calculate_background_image_on_background_buffer(index) for index in indices]
                for x in range(len(results)):
                    self.background_calculation_buffer[x] = results[x]
                self.backgrounds_buffer.append(self.background_calculation_buffer[0])
                # self.add_timestamp('level {}'.format(i))

            if self.verbose:
                print(self.to_str())
            # self.clear_timestamps()
            self.backgrounds = self.backgrounds_buffer
        except Exception as e:
            print(e)
        self.recalculation_allowed = True

    def calculate_async(self):
        """
        Tries to start the calculation of the Teknomo-Fernandez algorithm on the current set of images.
        If no calculation is currently running it starts a new process, else does nothing.
        """
        if not self.recalculation_allowed:
            return
        Thread(target=self.calculation).start()

    def testing(self) -> Frame:
        frame = self[-1].clone()
        background = self.get_background()
        difference = frame.subtract(background, absolute=True)
        self.add_timestamp('difference')

        threshold = 80

        # hist = cv.calcHist([difference], [0], None, [256], [0, 256])
        # hist_mean = np.mean(hist)
        # threshold = 255
        # for index, h in enumerate(hist):
        #     if h[0] > hist_mean * 2:
        #         if index < threshold:
        #             threshold = index

        rel, difference = cv.threshold(difference.cpu(), threshold, 255, cv.THRESH_BINARY)
        self.add_timestamp('threshold')
        foreground_bitmask = Frame(difference)
        print(self.to_str(reset=True))
        return foreground_bitmask

    def calculate_teknomo_fernandez_segmentation(self, kernel=np.ones((5, 5), np.uint8), diameter=9,
                                                 sigma_color=75, sigma_space=75, dilate_iterations=(3, 3),
                                                 bitmask_threshold=0.5):
        scale_factor = 1
        background = scale_frame(self.get_background(), scale_factor)
        current_frame = to_cpu_frame(scale_frame(self[-1], scale_factor))

        foreground_bitmask = current_frame - to_cpu_frame(background)
        foreground_bitmask = CudaOpenFilter().apply(foreground_bitmask)
        foreground_bitmask = CudaBilateralFilter().apply(foreground_bitmask)
        foreground_bitmask = CudaDilateFilter().apply(foreground_bitmask)
        foreground_bitmask = WhereFilter(100, 0, 255).apply(foreground_bitmask)
        foreground_bitmask = CudaOpenFilter().apply(foreground_bitmask)
        foreground_bitmask = to_3_channel_rgb(foreground_bitmask)

        # current_frame = cv.cvtColor(current_frame, cv.COLOR_RGB2GRAY)
        # background = cv.cvtColor(background, cv.COLOR_RGB2GRAY)
        #
        # foreground_bitmask = current_frame - background
        # foreground_bitmask = np.where(foreground_bitmask < 50, 0, 255)
        # foreground = current_frame * (foreground_bitmask / 255.)
        # background_bitmask = ~foreground_bitmask
        # # background = current_frame * (background_bitmask / 255.)
        # foreground = current_frame * foreground_bitmask

        scale = to_gpu_frame(self[-1]).size()
        foreground_bitmask = to_cpu_frame(resize_frame(foreground_bitmask, scale[0], scale[1]))

        foreground = (to_cpu_frame(self[-1]) * foreground_bitmask / 255.0) * 255
        background_bitmask = (~np.array(foreground_bitmask, dtype=np.bool)).astype(np.float32)
        background = to_cpu_frame(self[-1]) * background_bitmask / 255.0

        return [np.array(to_cpu_frame(x), dtype=np.uint8) for x in
                [foreground, foreground_bitmask, background, background_bitmask]]
