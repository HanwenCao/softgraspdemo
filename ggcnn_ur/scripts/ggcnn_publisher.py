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
import os
import socket
import select
import numpy as np
import struct
import pickle


pub = rospy.Publisher('/yolo', Float32MultiArray, queue_size=10)
rospy.init_node('node_name')
r = rospy.Rate(10) # 10hz
BUFSIZE = 1024
ip_port = ('127.0.0.1', 9999)
UDPSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # udp
UDPSock.bind(ip_port)
'''
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
'''
while not rospy.is_shutdown():
    data, client_addr = UDPSock.recvfrom(BUFSIZE)
    L = pickle.loads(data)
    print repr(L)
    #publish -> ros topic
    rospy.loginfo("happily publishing yolo results.. ")
    arg = Float32MultiArray()
    dim = MultiArrayDimension()
    dim.label = "x1,y1,x2,y2,X,Y,Z,average depth"
    dim.size = 8
    dim.stride = 8
    arg.layout.dim.append(dim)
    arg.data = L

    pub.publish(arg)
    r.sleep()
UDPSock.close()
