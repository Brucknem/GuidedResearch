#!/usr/bin/env python
from __future__ import print_function

import numpy as np

import rospy
import sys
import cv2

import roslib

from opencv_convert import RosStreamToOpenCV
from threading import Thread

roslib.load_manifest('providentia_camera_replay')


def main(args):
    global rospy_spin_thread
    size = (int(args[1]), int(args[2]))
    topics = args[3:]
    cv_visualization = RosStreamToOpenCV(*topics)
    rospy.init_node(cv_visualization.ros_name, anonymous=True)

    window_name = 'window'
    cv2.namedWindow(window_name)

    try:
        rospy_spin_thread = Thread(target=rospy.spin)
        rospy_spin_thread.start()
        while True:
            images = []
            for _, image in cv_visualization.images.items():
                if image is not None:
                    images.append(cv2.resize(image, size))

            if images:
                image = np.concatenate(images, axis=1)
                cv2.imshow(window_name, image)
                cv2.waitKey(3)
    except KeyboardInterrupt:
        print("Shutting down")
        rospy.signal_shutdown("Shutting down")
        rospy_spin_thread.join()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
