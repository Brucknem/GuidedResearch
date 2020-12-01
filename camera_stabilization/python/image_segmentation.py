import concurrent
import os
from datetime import datetime

import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
import random


def calculate_background_image(img1, img2, img3):
    model = cv2.cuda.bitwise_or(cv2.cuda.bitwise_and(img3, (cv2.cuda.bitwise_xor(img1, img2))), cv2.cuda.bitwise_and(img1, img2))
    return model


class TeknomoFernandez:
    def __init__(self, images: list, levels: int = 6, threads: int = 24):
        self.images = images
        self.levels = levels
        self.backgrounds = []
        self.background_buffer = []
        self.executor = ThreadPoolExecutor(threads)

    def reset(self):
        self.backgrounds = []
        self.background_buffer = []

    def calculate_teknomo_fernandez(self):
        start_time = datetime.now()
        if len(self.images) < 3:
            raise ValueError('The Teknomo-Fernandez algorithm needs at least 3 images to process.')

        self.reset()

        futures = []
        for i in range(3 ** (self.levels - 1)):
            futures.append(self.executor.submit(self.initialize_background_buffer))

        self.background_buffer = [future.result() for future in concurrent.futures.as_completed(futures)]
        self.backgrounds.append(self.background_buffer[-1])

        end_time = datetime.now()
        print("Duration: ", end_time - start_time)
        start_time = datetime.now()

        for i in range(2, self.levels + 1):
            futures = []
            for j in range(3 ** (self.levels - i)):
                futures.append(self.executor.submit(calculate_background_image, self.background_buffer[3 * j],
                                                    self.background_buffer[3 * j + 1],
                                                    self.background_buffer[3 * j + 2]))
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
            for x in range(len(results)):
                self.background_buffer[x] = results[x]
            self.backgrounds.append(self.background_buffer[0])
            end_time = datetime.now()
            print("Duration: ", end_time - start_time)
            start_time = datetime.now()

        end_time = datetime.now()
        print("Duration: ", end_time - start_time)

        return self.backgrounds

    def initialize_background_buffer(self):
        indices = np.random.randint(low=0, high=(len(self.images) - 1), size=3)
        return calculate_background_image(self.images[indices[0]], self.images[indices[1]], self.images[indices[2]])

    def calculate_teknomo_fernandez_segmentation(self, kernel=np.ones((5, 5), np.uint8), diameter=9,
                                                 sigma_color=75, sigma_space=75, dilate_iterations=(3, 3),
                                                 bitmask_threshold=0.5):
        current_frame = self.images[-1]
        backgrounds = self.calculate_teknomo_fernandez()
        background = backgrounds[-1]

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
