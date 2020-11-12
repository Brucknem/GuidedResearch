#!/usr/bin/env python
from __future__ import print_function

import numpy as np

import rospy
import sys
import cv2

import roslib

from ros_stream_to_open_cv_converter import RosStreamToOpenCVConverter
from image_processing import resize_images, layout_images, LayoutEntry, add_topic_mark

roslib.load_manifest('providentia_camera_replay')


def shutdown_signal_hook():
    """
    Signal hook for the shutdown signal
    """
    cv2.destroyAllWindows()


def main(args):
    """
    main

    Args:
        args: list
    """
    args = rospy.myargv(args)
    size = (int(args[1]), int(args[2]))

    layout_entries = []
    for i in range(3, len(args), 3):
        layout_entries.append(LayoutEntry(*args[i:i + 3]))

    rospy.on_shutdown(shutdown_signal_hook)
    ros_stream_to_open_cv_converter = RosStreamToOpenCVConverter(
        *list(map(lambda entry: entry.topic, layout_entries)))
    rospy.init_node(ros_stream_to_open_cv_converter.ros_name, anonymous=True, disable_signals=True)
    rate = rospy.Rate(30)

    window_name = 'Providentia++ Camera Stream Visualization'
    cv2.namedWindow(window_name)
    try:
        while not rospy.is_shutdown():
            images = ros_stream_to_open_cv_converter.get_images()
            images = resize_images(images, size)
            images = add_topic_mark(images)
            render_image = layout_images(images, layout_entries)
            cv2.imshow(window_name, render_image)
            cv2.waitKey(3)
            rate.sleep()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main(sys.argv)
