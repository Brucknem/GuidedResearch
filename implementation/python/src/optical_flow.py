import cv2 as cv
import numpy as np
from src.frame_utils import to_cpu_frame, Frame


class DenseOpticalFlow:
    """
    Dense Optical flow based on the farneback algorithm
    """
    cuda_stream = cv.cuda_Stream()
    flow_columns = ['mag', 'ang']

    def __init__(self):
        """
        Constructor
        """
        self.previous_frame = None
        self.hsv = None
        self.optical_flow = cv.cuda.FarnebackOpticalFlow_create()
        self._gpu_flow = cv.cuda_GpuMat()
        self.flow = None
        self.mag = None
        self.ang = None

    def initialize(self, frame: Frame):
        """
        Initialises the previous frame and buffers

        :param frame: The frame used for initialization
        """
        self.previous_frame = frame.clone()
        self.hsv = np.zeros_like(frame.cpu())
        self.hsv[..., 1] = 255

    def flow_to_bgr(self) -> Frame:
        """
        Converts the dense optical flow image to a BGR image

        :return: The BGR image
        """
        mag, ang = cv.cartToPolar(self.flow[..., 0], self.flow[..., 1])
        self.mag = mag
        self.ang = ang
        mag = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        ang = ang * 180 / np.pi / 2
        self.hsv[..., 0] = ang
        self.hsv[..., 2] = mag
        return Frame(cv.cvtColor(self.hsv, cv.COLOR_HSV2BGR))

    def get_movement_mask(self) -> Frame:
        if self.mag is None:
            return Frame.empty()
        ret, threshold = cv.threshold(self.mag, 3, 255, cv.THRESH_BINARY)
        return Frame(threshold)

    def apply_cpu(self, frame: Frame) -> Frame:
        """
        Applies the CPU version of the dense optical flow algorithm

        :param frame: The frame to calculate the optical flow.
        :return: The dense optical flow as BGR image
        """
        if self.previous_frame is None:
            self.initialize(frame)
            return Frame.empty_like(frame.shape())

        self.flow = cv.calcOpticalFlowFarneback(self.previous_frame.cpu(grayscale=True), frame.cpu(grayscale=True), None, 0.5, 3, 15, 3, 5, 1.2, 0)
        self.previous_frame = frame
        return self.flow_to_bgr()

    def apply_gpu(self, frame: Frame, previous_frame: Frame = None) -> Frame:
        """
        Applies the GPU version of the dense optical flow algorithm

        :param previous_frame:
        :param frame: The frame to calculate the optical flow.
        :return: The dense optical flow as BGR image
        """
        if previous_frame is not None:
            self.initialize(previous_frame)

        if self.previous_frame is None:
            self.initialize(frame)
            return Frame.empty_like(frame.shape())

        self.flow = self.optical_flow.calc(self.previous_frame.gpu(grayscale=True), frame.gpu(grayscale=True),
                                           self._gpu_flow,
                                           stream=DenseOpticalFlow.cuda_stream)
        self.flow = to_cpu_frame(self.flow)
        self.previous_frame = frame.clone()
        return self.flow_to_bgr()

    def get_flow_value(self, row: int or float, column: int or float) -> dict:
        """
        Getter for a the magnitude and angle at a specific pixel location in the optical flow image

        :param row: The row of the pixel location
        :param column: The column of the pixel location
        :return: The magnitude and angle at the pixel location
        """
        values = (np.NaN, np.NaN)
        row = int(row)
        column = int(column)
        if self.flow is not None:
            values = [self.mag[row, column], self.ang[row, column]]
        return dict(zip(DenseOpticalFlow.flow_columns, values))

    def get_flow_means(self):
        """
        Getter for the mean of the magnitude and angle of the dense optical flow

        :return: The mean magnitude and angle of the dense optical flow
        """
        values = (np.NaN, np.NaN)
        if self.flow is not None:
            values = [np.nanmean(x) for x in [self.mag, self.ang]]
        return dict(zip(DenseOpticalFlow.flow_columns, values))

    def get_flow_medians(self):
        """
        Getter for the mean of the magnitude and angle of the dense optical flow

        :return: The mean magnitude and angle of the dense optical flow
        """
        values = (np.NaN, np.NaN)
        if self.flow is not None:
            values = [np.nanmedian(x) for x in [self.mag, self.ang]]
        return dict(zip(DenseOpticalFlow.flow_columns, values))
