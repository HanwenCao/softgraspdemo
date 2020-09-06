#!/usr/bin/python
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayLayout
from std_msgs.msg import MultiArrayDimension
import ros_numpy
import numpy as np
from PIL import Image as PILImage
import sys


pub = rospy.Publisher('/yolo', Float32MultiArray, queue_size=10)
rospy.init_node('node_name')
r = rospy.Rate(10) # 10hz
while not rospy.is_shutdown():
    arg = Float32MultiArray()
    #layout = arg.layout
    #layout = MultiArrayLayout()
    dim = MultiArrayDimension()

    dim.label = "X_cam,Y_cam,Z_cam"
    dim.size = 1
    dim.stride = 1
    arg.layout.dim.append(dim)

    arg.data = [100]

    pub.publish(arg)
    r.sleep()
