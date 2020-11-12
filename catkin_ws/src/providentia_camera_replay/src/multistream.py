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

    def __init__(self, topics, width=860, height=600):
        """
        constructor

        Keyword arguments:
            x -- pos x
            y -- pos y
            width -- window width
            height -- window height
        """
        self.ros_name = "multistream"
        self.bridge = CvBridge()

        # self.image_pub = rospy.Publisher("converted_image", Image, queue_size=10)
        self.image_subs = []
        for index, topic in enumerate(topics):
            self.image_subs.append(rospy.Subscriber(topic, Image, self.callback,
                                                    (topic, index, len(topics), int(width), int(height))))
        self.latest_images = {}
        self.window_name = 'window'
        # cv2.namedWindow(self.window_name)

    def callback(self, data, args):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            # cv_image = self.process(cv_image, args[3], args[4])
        except CvBridgeError as e:
            print(e)

        cv2.imshow(args[0], cv_image)
        cv2.waitKey(3)

        # try:
        #     # self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        # except CvBridgeError as e:
        #     print(e)

    def process(self, cv_image, width, height):
        cv_image = cv2.resize(cv_image, (width, height))
        return cv_image
