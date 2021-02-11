import os
from threading import Thread
import cv2 as cv
from src.frame_utils import Frame
from pathlib import Path
import shutil


class ImageWriter:
    def __init__(self, base_path, suffix: str = 'png', prefix: str = 'img_', remove_if_exists: bool = True):
        self.base_path = os.path.expanduser(base_path)
        if remove_if_exists:
            shutil.rmtree(self.base_path, ignore_errors=True)
        Path(self.base_path).mkdir(parents=True, exist_ok=True)
        self.sequence_num: int = 0
        self.prefix = prefix
        self.suffix = suffix

    def write(self, frame: Frame, name: str = None):
        filename = self.base_path
        if name is not None:
            filename = os.path.join(filename, name)
        else:
            filename = os.path.join(filename,
                                    '{}{}.{}'.format(self.prefix, str(self.sequence_num).zfill(5), self.suffix))
            self.sequence_num += 1
        Thread(target=cv.imwrite, args=(filename, frame.cpu())).start()
