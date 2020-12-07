import cv2 as cv

from src.frame_utils import Frame
from src.timable import ITimable
import numpy as np


def layout(frames: list, positions: list) -> Frame:
    """
    Creates a frame that has the given frames at the given positions.

    :param frames: The frames to merge
    :param positions: The pixel position at which to place the top left pixel of the frames
    :return: The final merged frame
    """
    num_frames, num_positions = len(frames), len(positions)
    if num_frames == 0:
        return Frame.empty()
    if num_frames > num_positions:
        raise ValueError(
            'More frames than positions given. Frames: {} - Positions: {}'.format(num_frames, num_positions))

    positions = positions[:num_frames]
    num_rows = 0
    num_columns = 0

    for frame, position in zip(frames, positions):
        size = frame.size()
        end_pixel = position[0] + size[0]
        if end_pixel > num_rows:
            num_rows = end_pixel
        end_pixel = position[1] + size[1]
        if end_pixel > num_columns:
            num_columns = end_pixel

    result = np.ones_like(np.ndarray([int(num_rows), int(num_columns), 3]), dtype=np.uint8) * 127

    for frame, position in zip(frames, positions):
        size = frame.size()
        start_row = int(position[0])
        start_column = int(position[1])
        result[start_row:start_row + size[0], start_column: start_column + size[1], :] = frame.cpu()

    return Frame(result)


class Renderer(ITimable):
    """
    A renderer for frames.
    """
    def __init__(self, window_name: str = 'Camera Visualization'):
        """
        constructor

        :param window_name: The name of the OpenCV window.
        """
        super().__init__('Renderer')
        self.window_name = window_name

    def render(self, frames: list or Frame, positions: list = None) -> bool:
        """
        Renders the given frame/frames. Layouts the frames to the given positions if specified.
        The frames are sized

        :param frames:
        :param positions:
        :return:
        """
        self.add_timestamp()
        duration = self.get_latest_duration()
        if type(frames) is list:
            render_frame = layout(frames, positions)
        else:
            render_frame = frames
        render_frame.add_text('{0:.2f} ms ({1:.2f} fps)'.format(duration, 1. / duration))
        cv.imshow(self.window_name, render_frame.cpu())
        return not (cv.waitKey(1) & 0xFF == ord('q'))
