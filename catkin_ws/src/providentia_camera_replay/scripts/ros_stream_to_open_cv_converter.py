#!/usr/bin/env python
from __future__ import print_function

from datetime import datetime

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import rospy


class RosStreamToOpenCVConverter:
    """
    Class to convert ROS image_raw streams to opencv images
    """

    def __init__(self, *topics):
        """
        constructor

        Args:
            *topics: list The names of the topics to subscribe
        """
        self.ros_name = 'ros_stream_to_open_cv_converter'
        self.topics = topics

        self.bridge = CvBridge()
        self.image_subs = {}
        self.images = {}
        self.last_update_time = {}

        for topic in topics:
            self.image_subs[topic] = rospy.Subscriber(topic, Image, self.callback, (topic, ))
            self.images[topic] = None
            self.last_update_time[topic] = datetime.now()

    def callback(self, data, args):
        """
        Called when a subscriber receives data.

        Args:
            data: object ROS image message in sensor_msgs/Image format
            args: (str, ) The topic that received data
        """
        topic = args[0]
        try:
            self.images[topic] = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.last_update_time[topic] = datetime.now()
        except CvBridgeError as e:
            print(e)

    def get_images(self):
        """
        Gets the latest received images by topic

        Returns:
            {str: cv2.image}
        """
        now = datetime.now()
        for topic in self.topics:
            if (now - self.last_update_time[topic]).seconds > 3:
                self.images[topic] = None

        return dict(filter(lambda elem: elem[1] is not None, self.images.items()))