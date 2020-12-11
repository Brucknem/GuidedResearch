from threading import Thread
from time import sleep

import cv2 as cv
from os import listdir
from os.path import isfile, join

from src.frame_utils import Frame
from src.timable import ITimable
from src.utils import FixedSizeSortedDict


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


def load_frame_from_disk(file_name: str) -> Frame:
    return Frame(cv.imread(file_name))


class ImageBasedVideoCapture(FixedSizeSortedDict, ITimable):
    """
    OpenCV video capture mock to stream image files like a video stream from disk
    """

    def __init__(self, path: str, file_ending: str = '.png', frame_rate: int = 25, max_loaded_frames: int = 100,
                 loop: bool = True):
        """
        constructor

        :param path: The path to the directory containing the image files
        :param file_ending: The file ending of the images to load
        :param frame_rate: The framerate of the video stream
        :param max_loaded_frames: The maximum number of frames that can be in the frame buffer
        :param loop: Loop when the last image is read
        """
        FixedSizeSortedDict.__init__(self, max_num_elements=max_loaded_frames, remove_random=True)
        ITimable.__init__(self, 'ImageBasedVideoCapture')
        self.path = path
        self.file_names = read_files_in_path(path, file_ending)
        self.num_frames = len(self.file_names)
        self.frame_index = 0
        self.frame_rate = frame_rate
        self.frame_time = 1.0 / frame_rate
        self.loop = loop
        self.latest_loaded_frame = 0
        self.is_thread_running = False
        self.add_timestamp('given')

        # self.load_frames()
        self.load_thread = self.new_load_thread()

    def new_load_thread(self):
        thread = Thread(target=self.load_frames)
        thread.start()
        return thread

    def load_frames(self):
        if self.is_thread_running:
            return

        start_index = self.frame_index
        elements_to_load = self._max_num_elements
        if elements_to_load <= 0:
            elements_to_load = self.num_frames

        self.is_thread_running = True
        for i in range(elements_to_load):
            if not self.is_thread_running:
                break
            index = i + start_index
            if self.loop:
                index %= self.num_frames
            else:
                if index >= self.num_frames:
                    break
            file_name = self.file_names[index]
            if file_name in super().keys():
                continue
            self[file_name] = None
            self[file_name] = cv.imread(file_name)
            self.latest_loaded_frame = index
            # print(index)
        self.is_thread_running = False

    def __getitem__(self, item):
        if self.frame_index > self.latest_loaded_frame - self.latest_loaded_frame / 10:
            self.is_thread_running = False
            self.load_thread = self.new_load_thread()

        current_frame = self.file_names[self.frame_index]
        if current_frame not in self or super().__getitem__(current_frame) is None:
            self[current_frame] = cv.imread(current_frame)
            # self[current_frame] = cv.cuda_GpuMat()
            # self[current_frame].upload(cv.imread(current_frame))
        return super().__getitem__(current_frame)

    def read(self):
        """
        Reads the next frame from the list of images

        :return: True and image if frame loaded successfully, False and None else
        """
        if self.loop:
            self.frame_index %= self.num_frames
        elif self.frame_index >= self.num_frames:
            return False, None

        current_frame = self[self.file_names[self.frame_index]]
        self.add_timestamp()

        duration = self.get_latest_duration()
        sleeping = self.frame_time - duration
        # print(sleeping)
        sleep(max(0., sleeping))

        self.frame_index = self.frame_index + 1
        self.add_timestamp()
        return True, current_frame

    def get_frame_name(self):
        """
        Getter for the latest read frame
        """
        if self.frame_index <= 0:
            return None
        return self.file_names[self.frame_index - 1]

    def get_frames(self):
        """
        Getter for the frames
        """
        return super().items()

    def __str__(self):
        """
        to string
        """
        return "[Filesystem VideoCapture]@{} ({}:{})".format(self.path, self.frame_index, self.num_frames)
