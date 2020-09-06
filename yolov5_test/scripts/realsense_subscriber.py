#!/usr/bin/python
import rospy
from sensor_msgs.msg import Image
import ros_numpy

import numpy as np
from PIL import Image as PILImage
import sys

#from cv_bridge import CvBridge, CvBridgeError



IMAGE_TOPIC = "/camera/color/image_raw"
DEPTH_TOPIC = "/camera/depth/image_rect_raw"


def get_image(show=False):
    print("CALLING GET_REALSENSE_IMAGE")
    rospy.init_node("realsense_subscriber")
    rgb = rospy.wait_for_message(IMAGE_TOPIC, Image)
    depth = rospy.wait_for_message(DEPTH_TOPIC, Image)
    print("GET_AN_IMAGE")
    

    # Convert sensor_msgs.Image readings into readable format
    rgb_arr = ros_numpy.numpify(rgb)
    depth_arr = ros_numpy.numpify(depth)
    #bridge = CvBridge()
    #rgb = bridge.imgmsg_to_cv2(rgb, rgb.encoding)
    #depth = bridge.imgmsg_to_cv2(depth, depth.encoding)


    #image = rgb
    #image[:, :, 2] = depth
    if (show):
        im = PILImage.fromarray(rgb_arr, 'RGB')
        im.show()

    #return image
    return rgb_arr, depth_arr


if __name__ == '__main__':
    image = get_image(show=True)
