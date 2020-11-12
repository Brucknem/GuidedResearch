#!/usr/bin/env python
from __future__ import print_function
import rospy
import sys
import cv2

import roslib

from opencv_visualization import OpenCVVisualization

roslib.load_manifest('providentia_camera_replay')

def main(args):
    cv_visualization = OpenCVVisualization(*args[1:6])
    rospy.init_node(cv_visualization.ros_name, anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt | Exception:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
