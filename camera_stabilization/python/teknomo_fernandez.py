import concurrent
import os
from datetime import datetime

import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
import random

from timable import ITimable
from utils import FixedSizeList


def calculate_background_image(img1, img2, img3):
    """
    Calculates the background image based on the Teknomo-Fernandez algorithm.

    :return: img3 * (img1 ^ img2) + img1 * img2
    """
    model = cv2.cuda.bitwise_or(cv2.cuda.bitwise_and(img3, (cv2.cuda.bitwise_xor(img1, img2))), cv2.cuda.bitwise_and(img1, img2))
    return model


class TeknomoFernandez(ITimable, FixedSizeList):
    """
    Calculates the background of a set of images based on the Teknomo-Fernandez algorithm.

    https://en.wikipedia.org/wiki/Teknomo%E2%80%93Fernandez_algorithm
    """
    def __init__(self, levels: int = 6, threads: int = 24, history: int = 200, random_history: bool = True):
        """
        constructor

        :param levels: The number of levels during the algorithm
        :param threads: The number of parallel threads used during calculation
        :param history: The number of images in the history buffer to calculate the background from
        :param random_history: Flag if the elements in the history are removed randomly if the buffer is filled
        """
        ITimable.__init__(self, name='Teknomo-Fernandez')
        FixedSizeList.__init__(self, max_num_elements=history, remove_random=random_history)
        self.levels = levels
        self.backgrounds_buffer = []
        self.background_calculation_buffer = []
        self.executor = ThreadPoolExecutor(threads)
        self.backgrounds = []

    def get_background(self):
        """
        Gets the latest calculated background image.

        :return: The latest background image if enough images are in the history,
                    The latest added frame if not,
                    None if no frames are added yet
        """
        if len(self.backgrounds) >= 1:
            return self.backgrounds[-1]
        if self.__len__() == 0:
            return None
        return self[-1]

    def reset_buffers(self):
        """
        Resets the background calculation buffers.
        """
        self.backgrounds_buffer = []
        self.background_calculation_buffer = []

    def calculate(self):
        """
        Performs calculation of the Teknomo-Fernandez algorithm on the current set of images.
        """
        if len(self) < 3:
            return

        self.add_timestamp('start')

        self.reset_buffers()

        futures = []
        for i in range(3 ** (self.levels - 1)):
            futures.append(self.executor.submit(self.calculate_random_background))

        self.background_calculation_buffer = [future.result() for future in concurrent.futures.as_completed(futures)]
        self.backgrounds_buffer.append(self.background_calculation_buffer[-1])

        self.add_timestamp('initialization')

        for i in range(2, self.levels + 1):
            futures = []
            for j in range(3 ** (self.levels - i)):
                futures.append(self.executor.submit(calculate_background_image, self.background_calculation_buffer[3 * j],
                                                    self.background_calculation_buffer[3 * j + 1],
                                                    self.background_calculation_buffer[3 * j + 2]))
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
            for x in range(len(results)):
                self.background_calculation_buffer[x] = results[x]
            self.backgrounds_buffer.append(self.background_calculation_buffer[0])
            self.add_timestamp('level {}'.format(i))

        self.add_timestamp('finish')
        # print(self.durations_to_str(reset=True))
        self.backgrounds = self.backgrounds_buffer

    def calculate_random_background(self):
        """
        Calculates a background image of three random images in the buffer.

        :return: The calculated background^
        """
        indices = np.random.randint(low=0, high=(len(self) - 1), size=3)
        return calculate_background_image(self[indices[0]], self[indices[1]], self[indices[2]])

    def calculate_teknomo_fernandez_segmentation(self, kernel=np.ones((5, 5), np.uint8), diameter=9,
                                                 sigma_color=75, sigma_space=75, dilate_iterations=(3, 3),
                                                 bitmask_threshold=0.5):
        current_frame = self[-1]
        self.calculate()
        background = self.get_background()

        foreground_bitmask = cv2.dilate(
            cv2.bilateralFilter(
                cv2.morphologyEx(current_frame - background, cv2.MORPH_OPEN, kernel)
                , diameter, sigma_color, sigma_space)
            , kernel, iterations=dilate_iterations[0])

        foreground_bitmask = np.sum(foreground_bitmask / 3, axis=2, dtype=np.float32) / 255.
        foreground_bitmask = np.where(foreground_bitmask < bitmask_threshold, 0.0, 1.0)

        foreground_bitmask = cv2.dilate(foreground_bitmask, kernel, iterations=dilate_iterations[1])[..., None]
        foreground_bitmask = foreground_bitmask

        foreground = current_frame * foreground_bitmask / 255.0
        background_bitmask = (~np.array(foreground_bitmask, dtype=np.bool)).astype(np.float32)
        background = current_frame * background_bitmask / 255.0

        return foreground, foreground_bitmask, background, background_bitmask
