#!/usr/bin/env python
from __future__ import print_function
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import cv2
import rospy


class RosStreamToOpenCV:
    """
    An OpenCV visualizer for the providentia camera streams
    """

    def __init__(self, *topics):
        """
        constructor

        Keyword arguments:
            x -- pos x
            y -- pos y
            width -- window width
            height -- window height
        """
        self.ros_name = 'visualization'
        self.topics = topics

        self.bridge = CvBridge()
        self.image_subs = {}
        self.images = {}

        for topic in topics:
            self.image_subs[topic] = rospy.Subscriber(topic, Image, self.callback, (topic, ))
            self.images[topic] = None

    def callback(self, data, args):
        topic = args[0]
        try:
            self.images[topic] = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
