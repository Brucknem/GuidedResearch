import cv2 as cv
import numpy as np
from src.frame_utils import to_cpu_frame, Frame


class DenseOpticalFlow:
    """
    Dense Optical flow based on the farneback algorithm
    """
    cuda_stream = cv.cuda_Stream()
    flow_columns = ['Magnitude', 'Angle']

    def __init__(self):
        """
        Constructor
        """
        self.previous_frame = None
        self.hsv = None
        self.optical_flow = cv.cuda.FarnebackOpticalFlow_create()
        self._gpu_flow = cv.cuda_GpuMat()
        self.flow = None

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
        self.hsv[..., 0] = ang * 180 / np.pi / 2
        self.hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        return Frame(cv.cvtColor(self.hsv, cv.COLOR_HSV2BGR))

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

    def apply_gpu(self, frame: Frame) -> Frame:
        """
        Applies the GPU version of the dense optical flow algorithm

        :param frame: The frame to calculate the optical flow.
        :return: The dense optical flow as BGR image
        """
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
            values = self.flow[row, column]
        return dict(zip(DenseOpticalFlow.flow_columns, values))

    def get_flow_means(self):
        """
        Getter for the mean of the magnitude and angle of the dense optical flow

        :return: The mean magnitude and angle of the dense optical flow
        """
        values = (np.NaN, np.NaN)
        if self.flow is not None:
            values = [np.nanmean(self.flow[:, :, x]) for x in range(self.flow.shape[2])]
        return dict(zip(DenseOpticalFlow.flow_columns, values))
