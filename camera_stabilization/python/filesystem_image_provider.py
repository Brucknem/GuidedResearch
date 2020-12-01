import cv2 as cv

from time import sleep
from os import listdir
from os.path import isfile, join
import numpy as np

def read_files_in_path(path: str, file_ending: str = '.png'):
    """
    Read all files with the given ending in the given path

    :param path:
    :param file_ending:
    :return:
    """
    if isfile(path):
        return []

    files = []
    for f in listdir(path):
        file_path = join(path, f)
        if isfile(file_path):
            files.append(file_path)
    files.sort()
    return files


class VideoCapture:
    """

    """

    def __init__(self, path: str, file_ending: str = '.png', frame_rate: int = 25, max_loaded_frames: int = 400):
        """
        constructor

        :param path:
        :param file_ending:
        """
        self.path = path
        self.file_names = read_files_in_path(path, file_ending)
        self.frames = {}
        self.num_frames = len(self.file_names)
        self.frame_index = 0
        self.frame_rate = frame_rate
        self.sleep_duration = 1.0 / frame_rate
        self.max_loaded_frames = max_loaded_frames

    def read(self):
        while len(self.frames) >= self.max_loaded_frames:
            del self.frames[list(self.frames.keys())[np.random.randint(0, len(self.frames))]]

        current_frame = self.file_names[self.frame_index]
        if current_frame not in self.frames:
            self.frames[current_frame] = cv.cuda_GpuMat()
            self.frames[current_frame].upload(cv.imread(current_frame))

        current_frame = self.frames[current_frame]

        self.frame_index = (self.frame_index + 1) % self.num_frames
        return True, current_frame

    def get_frames(self):
        return self.frames

    def __str__(self):
        """
        to string
        """
        return "[Filesystem VideoCapture]@{} ({}:{})".format(self.path, self.frame_index, self.num_frames)
