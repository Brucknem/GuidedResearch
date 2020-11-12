#!/usr/bin/env python
from __future__ import print_function
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import cv2
import rospy


class OpenCVVisualization:
    """
    An OpenCV visualizer for the providentia camera streams
    """

    def __init__(self, topic, x, y, width, height):
        """
        constructor

        Keyword arguments:
            x -- pos x
            y -- pos y
            width -- window width
            height -- window height
        """
        self.ros_name = 'visualization'
        self.topic = topic
        self.pos = (int(x), int(y))
        self.size = (int(width), int(height))

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.window_created = False

    def callback(self, data):
        if not self.window_created:
            cv2.namedWindow(self.topic)
            cv2.moveWindow(self.topic, *self.pos)
            self.window_created = True

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            cv_image = self.process(cv_image, self.size)
        except CvBridgeError as e:
            print(e)

    def process(self, cv_image, size):
        cv_image = cv2.resize(cv_image, size)
        cv2.imshow(self.topic, cv_image)
        cv2.waitKey(3)
        return cv_image
