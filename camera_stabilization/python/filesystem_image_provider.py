import cv2 as cv

from time import sleep
from os import listdir
from os.path import isfile, join
import numpy as np
from utils import FixedSizeOrderedDict


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
        if not f.endswith(file_ending):
            continue
        file_path = join(path, f)
        if isfile(file_path):
            files.append(file_path)
    files.sort()
    return files


class ImageBasedVideoCapture:
    """
    OpenCV video capture mock to stream image files like a video stream from disk
    """

    def __init__(self, path: str, file_ending: str = '.png', frame_rate: int = 25, max_loaded_frames: int = 10,
                 loop: bool = True):
        """
        constructor

        :param path: The path to the directory containing the image files
        :param file_ending: The file ending of the images to load
        :param frame_rate: The framerate of the video stream
        :param max_loaded_frames: The maximum number of frames that can be in the frame buffer
        :param loop: Loop when the last image is read
        """
        self.path = path
        self.file_names = read_files_in_path(path, file_ending)
        self.frames = FixedSizeOrderedDict(max_num_elements=max_loaded_frames, remove_random=True)
        self.num_frames = len(self.file_names)
        self.frame_index = 0
        # self.frame_rate = frame_rate
        # self.sleep_duration = 1.0 / frame_rate
        self.max_loaded_frames = max_loaded_frames
        self.loop = loop

    def read(self):
        """
        Reads the next frame from the list of images

        :return: True and image if frame loaded successfully, False and None else
        """
        if self.loop:
            self.frame_index %= self.num_frames
        elif self.frame_index >= self.num_frames:
            return False, None

        current_frame = self.file_names[self.frame_index]
        if current_frame not in self.frames:
            self.frames[current_frame] = cv.cuda_GpuMat()
            self.frames[current_frame].upload(cv.imread(current_frame))

        current_frame = self.frames[current_frame]
        self.frame_index = self.frame_index + 1
        return True, current_frame

    def get_frames(self):
        """
        Getter for the frames
        """
        return self.frames

    def __str__(self):
        """
        to string
        """
        return "[Filesystem VideoCapture]@{} ({}:{})".format(self.path, self.frame_index, self.num_frames)
