import sys, os
import rospy

import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from std_msgs.msg import Float32MultiArray
from ggcnn_ur.msg import Grasp

import math
from math import pi
import numpy
import numpy as np
import time


import socket
import pickle
import select
import copy
import copy as copy_module


from utils.slerp import slerp
from utils.transformations import euler_from_quaternion
from utils.python_serial_driver import PythonSerialDriver
#psd = PythonSerialDriver()
#psd.loopTestRTFRetry(False, False)


def list_to_quaternion(l):
    q = geometry_msgs.msg.Quaternion()
    q.x = l[0]
    q.y = l[1]
    q.z = l[2]
    q.w = l[3]
    return q

def quaternion_to_list(q):
    l = [0,0,0,0]
    l[0] = q.x
    l[1] = q.y
    l[2] = q.z
    l[3] = q.w
    return l

class MoveGroupInteface(object):
    def __init__(self):
        super(MoveGroupInteface, self).__init__()
        ######################### setup ############################
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('ur_move_test_node', anonymous=True)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()  # Not used in this tutorial
        group_name = "manipulator"  # group_name can be find in ur5_moveit_config/config/ur5.srdf
        self.move_group_commander = moveit_commander.MoveGroupCommander(group_name)
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',moveit_msgs.msg.DisplayTrajectory,queue_size=20)

        ################ Getting Basic Information ######################
        self.planning_frame = self.move_group_commander.get_planning_frame()
        print "============ Planning frame: %s" % self.planning_frame
        self.eef_link = self.move_group_commander.get_end_effector_link()
        print "============ End effector link: %s" % self.eef_link
        self.group_names = self.robot.get_group_names()
        print "============ Available Planning Groups:", self.robot.get_group_names()
        print "============ Printing robot state:"
        print self.robot.get_current_state()  # get
        self.move_group_commander.set_pose_reference_frame('base_link')
        # self.move_group_commander.set_goal_position_tolerance(0.01)
        # self.move_group_commander.set_goal_orientation_tolerance(0.01)
        self.move_group_commander.set_goal_position_tolerance(0.001)
        self.move_group_commander.set_goal_orientation_tolerance(0.001)
        # self.move_group_commander.set_max_acceleration_scaling_factor(0.1)
        # self.move_group_commander.set_max_velocity_scaling_factor(0.3)
        self.move_group_commander.set_max_acceleration_scaling_factor(1)
        self.move_group_commander.set_max_velocity_scaling_factor(1)
        #self.move_group_commander.set_named_target('home')

        print "============ Go home ============ "
        home = [1.9197, -1.5707, 1.5707, -1.5707, -1.5707, 0]
        self.move_group_commander.go(home, wait=True)  #go to home
        time.sleep(0.5)
        pose_init = self.move_group_commander.get_current_pose()
        print "current euler: ", euler_from_quaternion(quaternion_to_list( pose_init.pose.orientation ))

        ################ fruit pnp paras ######################
        self.cam_res = []
        self.last_cam_res = []        
        self.txyz = []
        self.pxyz = []        
        self.is_gripper_open = False
        self.grasp_new = Grasp() # to listen to the latest Grasp() msg

    def plan_cartesian_path(self, txyz):
        waypoints = []
        wpose = self.move_group_commander.get_current_pose().pose #!!
        print "Current pose: ", wpose
        print "Goal pose: ", txyz

        cnt =  100
        # interpolate quaternion
        v0 = [wpose.orientation.x,wpose.orientation.y,wpose.orientation.z,wpose.orientation.w] # quaternion start
        v1 = [txyz[3],txyz[4],txyz[5],txyz[6]]
        quat_itp = slerp(v0, v1, np.arange(0,1,1.0/cnt))
        # interpolate position
        dx = (txyz[0] - wpose.position.x) / cnt
        dy = (txyz[1] - wpose.position.y) / cnt
        dz = (txyz[2] - wpose.position.z) / cnt
        # dox = (txyz[3] - wpose.orientation.x) / cnt
        # doy = (txyz[4] - wpose.orientation.y) / cnt
        # doz = (txyz[5] - wpose.orientation.z) / cnt
        # dow = (txyz[6] - wpose.orientation.w) / cnt
        for i in range(cnt):
            wpose.position.x += dx
            wpose.position.y += dy
            wpose.position.z += dz
            # wpose.orientation.x += dox
            # wpose.orientation.y += doy
            # wpose.orientation.z += doz
            # wpose.orientation.w += dow
            wpose.orientation.x = quat_itp[i][0]
            wpose.orientation.y = quat_itp[i][1]
            wpose.orientation.z = quat_itp[i][2]
            wpose.orientation.w = quat_itp[i][3]
            waypoints.append(copy.deepcopy(wpose))
            # print('wpose:',wpose.orientation.x,wpose.orientation.y,wpose.orientation.z,wpose.orientation.w)
            # print('quat_itp:',quat_itp[i])

        # We want the Cartesian path to be interpolated at a resolution of 1 cm
        # which is why we will specify 0.01 as the eef_step in Cartesian
        # translation.  We will disable the jump threshold by setting it to 0.0,
        # ignoring the check for infeasible jumps in joint space, which is sufficient
        # for this tutorial.
        (plan, fraction) = self.move_group_commander.compute_cartesian_path(waypoints,   # waypoints to follow
            0.01,      # eef_step
            0.0)         # jump_threshold[-0.2919149470652611, 0.3813172330051884, 0.27368078496715426]
        
        if fraction < 1:
            print ("WARNNING: only " + str(fraction) + " of planned trajectory can be executed")
        # Note: We are just planning, not asking move_group to actually move the robot yet:
        # print "=========== Planning completed, Cartesian path is saved============="
        # print('plan plan', plan)
        return plan, fraction

    def execute_plan(self, plan):
        ## Use execute if you would like the robot to follow
        ## the plan that has already been computed:
        self.move_group_commander.execute(plan, wait=True)

    # other functions
    def RotX(self, theta):
        ans = numpy.array([[ 1,                 0,                 0],
                           [ 0, +numpy.cos(theta), -numpy.sin(theta)],
                           [ 0, +numpy.sin(theta), +numpy.cos(theta)]])
        return ans

    def RotY(self, theta):
        ans = numpy.array([[ +numpy.cos(theta), 0, +numpy.sin(theta)],
                           [                 0, 1,                 0],
                           [ -numpy.sin(theta), 0, +numpy.cos(theta)]])
        return ans

    def RotZ(self, theta):
        ans = numpy.array([[ +numpy.cos(theta), -numpy.sin(theta), 0],
                           [ +numpy.sin(theta), +numpy.cos(theta), 0],
                           [                 0,                 0, 1]])
        return ans

    def RPY2Mat(self, RPY): # [x, y, z] = [roll, pitch, yaw]
        return numpy.dot(self.RotZ(RPY[2]), numpy.dot(self.RotY(RPY[1]), self.RotX(RPY[0])))

    def position_from_camera_to_robot(self, x, y, z):
        R0 = numpy.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        #R1 = self.RPY2Mat(numpy.array([1, 10, -50]) * pi / 180.0)
        #R1 = self.RPY2Mat(numpy.array([6, 18, -50]) * pi / 180.0)        #robot to the rightmost
        #R1 = self.RPY2Mat(numpy.array([-2, 5, -53]) * pi / 180.0)        #brute force 0
        #R1 = self.RPY2Mat(numpy.array([-2, 7, -48]) * pi / 180.0)        #brute force 1
        R1 = self.RPY2Mat(numpy.array([-4, 8, -45]) * pi / 180.0)         #brute force 2
        cRb = numpy.dot(R1, R0)       
        #cPb = numpy.array([-0.3775, 0.554, 1.016])
        #cPb = numpy.array([-0.3475, 0.604, 1.016])                       #robot on left 7cm of rightmost posn
        #cPb = numpy.array([-0.3775+0.09, 0.554-0.04, 1.016+0.427])       #bf1
        #cPb = numpy.array([-0.3775+0.065, 0.554-0.06, 1.016+0.397])      #bf2@center
        cPb = numpy.array([-0.3775+0.055, 0.554-0.06, 1.016+0.357])       #bf2@xc=210
        cTb = numpy.eye(4)
        cTb[0:3, 0:3] = cRb
        cTb[0:3, 3] = cPb
        bTc = numpy.linalg.pinv(cTb)
        print('bTc', bTc)
        cPo = numpy.array([x, y, z, 1])
        bPo = numpy.dot(bTc, cPo)        
        return [-bPo[0], -bPo[1], bPo[2]]

    # def recieve(self):
    #     self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #     self.ip_port = ('127.0.0.1', 9999)
    #     self.s.bind(self.ip_port)
    #     self.s.setblocking(0)
    #     readable = select.select([self.s], [], [], 1.5)[0]
    #     if readable:
    #         data, client_addr = self.s.recvfrom(1024)            
    #         return [1, data]
    #     else:
    #         return [0]


    # def get_data(self):        
    #     data = self.recieve()
    #     if data[0]:
    #         self.cam_res = pickle.loads(data[1])
    #     else:            
    #         self.cam_res = [0]     
    #     self.s.close() 

    def recieve(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # can be put in init so that only created once??
        ip_port = ('127.0.0.1', 9999)
        self.s.bind(ip_port)
        self.s.setblocking(0)
        readable = select.select([self.s], [], [], 1.5)[0]

        grasp = Grasp()

        if readable:
            data, client_addr = self.s.recvfrom(1024)         
            grasp = pickle.loads(data)          
            return grasp
        else:            
            return grasp


    def get_data(self):
        grasp = self.recieve() # if no update, keep 0
        if grasp.quality > 0.0:
            self.grasp_new = copy_module.deepcopy(grasp) # if no update, keep the old data
            # update cam_res
            # cam_res = [orientation.x, orientation.y, orientation.z, orientation.w, position.x, position.y, position.z, object class(always the last one)]
            self.cam_res = [self.grasp_new.pose.orientation.x, self.grasp_new.pose.orientation.y, self.grasp_new.pose.orientation.z, self.grasp_new.pose.orientation.w, 
                            self.grasp_new.pose.position.x, self.grasp_new.pose.position.y, self.grasp_new.pose.position.z, 
                            3]
            print "test euler: ", euler_from_quaternion(quaternion_to_list( self.grasp_new.pose.orientation )) #/??why not same as what I send??
        else:
            self.cam_res = [0]

        self.s.close()
        return 1







print "----------------------------------------------------------"
print "Welcome to the MoveIt MoveGroup Python Interface Tutorial"
print "----------------------------------------------------------"
print "============ Press `Enter` to plan and display a Cartesian path ..."
raw_input()
tutorial = MoveGroupInteface()
rate = rospy.Rate(10)

################ fruit pnp config ######################
index = 0
#start posn UR joint_angle_0-5: 123,-90,90,-90,-90,0
#z_ready = 0.4

gripper_id = 1  # e.g. 0 -- 4fingers  (input arg)
grippers = ['4fingers','3fingers','2fingers']
offset_grippers = [0.025, 0, 0]  # offset of z_pick for each gripper
offset_gripper = offset_grippers[gripper_id]

z_ready = 0.4316+0.1 # above object (prepick)
orientation_default = [-0.5792, 0.4057, 0.5791, 0.4055] # pose.orientation of home, x,y,z,w
pick_idle = [0.10956455409982657, 0.4849471563129701, z_ready, 
            orientation_default[0],orientation_default[1],orientation_default[2],orientation_default[3]]  # mid stop between home and prepick
frontmost_xyz = [0.5647463207342078, 0.36478811826345703, 0.25877492322608786]
min_pcik_z = frontmost_xyz[2] - 0.002
#place location [0:3]
#class  ['moon cake', 'mango', 'durian', 'pineapple', 'apple', 'pear', 'orange', 'lemon']
# above conveyor
place0_xyz = [-0.763494309114598, 0.15840462134125174, 0.2988613606722884]
place1_xyz = [-0.46046322789986643, 0.6286065120173737, 0.3703955058493536]
place2_xyz = [-0.46046322789986643, 0.6286065120173737, 0.3703955058493536]
place3_xyz = [-0.26950425167001957, 0.41980037607846077, 0.4316088626751762]
place4_xyz = [-0.46046322789986643, 0.6286065120173737, 0.3703955058493536]
place5_xyz = [-0.3804991705753626, 0.109044414501911, 0.36213963613239697]
place6_xyz = [-0.5327506226449323, 0.2683311635139261, 0.3696519087605047]
place7_xyz = [-0.6014936133196379, -0.013024309127075877, 0.3300894039208666]
tutorial.pxyz = [place0_xyz, place1_xyz, place2_xyz, place3_xyz, place4_xyz, place5_xyz, place6_xyz, place7_xyz]

x_pick = frontmost_xyz[0]
y_pick = frontmost_xyz[1]
z_pick = frontmost_xyz[2]
x_place = tutorial.pxyz[3][0]
y_place = tutorial.pxyz[3][1]
z_place = tutorial.pxyz[3][2]

tutorial.txyz = [pick_idle, 
                [x_pick, y_pick, z_ready, orientation_default[0],orientation_default[1],orientation_default[2],orientation_default[3]], 
                [x_pick, y_pick, z_pick, orientation_default[0],orientation_default[1],orientation_default[2],orientation_default[3]], 
                [x_pick, y_pick, z_ready, orientation_default[0],orientation_default[1],orientation_default[2],orientation_default[3]], 
                pick_idle, 
                [x_place, y_place, z_ready, orientation_default[0],orientation_default[1],orientation_default[2],orientation_default[3]], 
                [x_place, y_place, z_place, orientation_default[0],orientation_default[1],orientation_default[2],orientation_default[3]], 
                [x_place, y_place, z_ready, orientation_default[0],orientation_default[1],orientation_default[2],orientation_default[3]]]

#psd.moveTo(psd.FLG_ZERO, 0, None, False)
#psd.moveTo(psd.FLG_NEG, 50, None, False)


while not rospy.is_shutdown():        
    ################ robot part ######################
    if index == 1:        
        tutorial.get_data()
        print('cam_res:', tutorial.cam_res)

        if len(tutorial.cam_res) > 1:
            if tutorial.cam_res[4] == 0.0 and tutorial.cam_res[5] == 0.0:
                print ("============ Conveyor Empty, Pending ...")            
                print(tutorial.cam_res[4:7])  
                rospy.sleep(1)
                continue
            else:
                if len(tutorial.last_cam_res) > 1:
                    tmp_x = tutorial.last_cam_res[4]
                    tmp_y = tutorial.last_cam_res[5]
                    tutorial.last_cam_res = tutorial.cam_res                    
                    if abs(tutorial.cam_res[4] - tmp_x) > 30 or abs(tutorial.cam_res[5] - tmp_y) > 30:
                        print ("=!=!=!=!=!=!=  Pre Pick, Invalid Target, Try New Img ...")                    
                        rospy.sleep(1)
                        continue    
                else:
                    print ("============  Pre Pick, Last Cam Res No Data, Retry ...")                
                    tutorial.last_cam_res = tutorial.cam_res
                    rospy.sleep(0.5)
                    continue   
  
                #assuming conveyor always running
                #acc to vision algo, we always recv the xyz close to snr
                #AND snr has pre-configed within ur working range
                #thus no need to use xyz filter anymore
                t = tutorial.position_from_camera_to_robot(tutorial.cam_res[4]/1000, tutorial.cam_res[5]/1000, tutorial.cam_res[6]/1000)
                print ("============  Pre Pick, Found Target: ", tutorial.txyz[2])

                x_pick = t[0]
                y_pick = t[1]
                if t[2] < min_pcik_z + offset_gripper:
                    z_pick = min_pcik_z + offset_gripper
                    print ("=!=!=!=!=!=!=  Pre Pick, Pick Z Too Small, Reset to Minimum ...")                    
                else:
                    z_pick = t[2] + offset_gripper              
                                      
                # tutorial.txyz = [pick_idle, [x_pick, y_pick, z_ready], [x_pick, y_pick, z_pick], [x_pick, y_pick, z_ready], pick_idle, [x_place, y_place, z_ready], [x_place, y_place, z_place], [x_place, y_place, z_ready]]
                tutorial.txyz = [pick_idle, 
                [x_pick, y_pick, z_ready, tutorial.cam_res[0],tutorial.cam_res[1],tutorial.cam_res[2],tutorial.cam_res[3]], 
                [x_pick, y_pick, z_pick, tutorial.cam_res[0],tutorial.cam_res[1],tutorial.cam_res[2],tutorial.cam_res[3]], 
                [x_pick, y_pick, z_ready, tutorial.cam_res[0],tutorial.cam_res[1],tutorial.cam_res[2],tutorial.cam_res[3]], 
                pick_idle, 
                [x_place, y_place, z_ready, orientation_default[0],orientation_default[1],orientation_default[2],orientation_default[3]], 
                [x_place, y_place, z_place, orientation_default[0],orientation_default[1],orientation_default[2],orientation_default[3]], 
                [x_place, y_place, z_ready, orientation_default[0],orientation_default[1],orientation_default[2],orientation_default[3]]]
                
        else:
            print ("============  Pre Pick, Pending for data ...")                
            rospy.sleep(1)
            continue                            

    elif index == 5:           
        #class  ['moon cake', 'mango', 'durian', 'pineapple', 'apple', 'pear', 'orange', 'lemon']
        if tutorial.cam_res[-1] == 0:
            print (">>>>>>>>>> Go to Mooncake Basket <<<<<<<<<< ")
            x_place = tutorial.pxyz[0][0]
            y_place = tutorial.pxyz[0][1]
            z_place = tutorial.pxyz[0][2] + offset_gripper                
        elif tutorial.cam_res[-1] == 3:
            print (">>>>>>>>>> Go to Pineapple Basket <<<<<<<<<< ")
            x_place = tutorial.pxyz[3][0]
            y_place = tutorial.pxyz[3][1]
            z_place = tutorial.pxyz[3][2] + offset_gripper
        elif tutorial.cam_res[-1] == 4:
            print (">>>>>>>>>> Go to Apple Basket <<<<<<<<<< ")
            x_place = tutorial.pxyz[4][0]
            y_place = tutorial.pxyz[4][1]
            z_place = tutorial.pxyz[4][2] + offset_gripper
        elif tutorial.cam_res[-1] == 5:
            print (">>>>>>>>>> Go to Pear Basket <<<<<<<<<< ")
            x_place = tutorial.pxyz[5][0]
            y_place = tutorial.pxyz[5][1]
            z_place = tutorial.pxyz[5][2] + offset_gripper
        elif tutorial.cam_res[-1] == 6:
            print (">>>>>>>>>> Go to Orange Basket <<<<<<<<<< ")
            x_place = tutorial.pxyz[6][0]
            y_place = tutorial.pxyz[6][1]
            z_place = tutorial.pxyz[6][2] + offset_gripper
        elif tutorial.cam_res[-1] == 7:
            print (">>>>>>>>>> Go to Lemon Basket <<<<<<<<<< ")
            x_place = tutorial.pxyz[7][0]
            y_place = tutorial.pxyz[7][1]
            z_place = tutorial.pxyz[7][2] + offset_gripper
        else:
            print ("=!=!=!=!=!=!=  Target Not Used in This Demo, Update Nothing ...")                    
            
        # tutorial.txyz = [pick_idle, [x_pick, y_pick, z_ready], [x_pick, y_pick, z_pick], [x_pick, y_pick, z_ready], pick_idle, [x_place, y_place, z_ready], [x_place, y_place, z_place], [x_place, y_place, z_ready]]
        tutorial.txyz = [pick_idle, 
                [x_pick, y_pick, z_ready, tutorial.cam_res[0],tutorial.cam_res[1],tutorial.cam_res[2],tutorial.cam_res[3]], 
                [x_pick, y_pick, z_pick, tutorial.cam_res[0],tutorial.cam_res[1],tutorial.cam_res[2],tutorial.cam_res[3]], 
                [x_pick, y_pick, z_ready, tutorial.cam_res[0],tutorial.cam_res[1],tutorial.cam_res[2],tutorial.cam_res[3]], 
                pick_idle, 
                [x_place, y_place, z_ready, orientation_default[0],orientation_default[1],orientation_default[2],orientation_default[3]], 
                [x_place, y_place, z_place, orientation_default[0],orientation_default[1],orientation_default[2],orientation_default[3]], 
                [x_place, y_place, z_ready, orientation_default[0],orientation_default[1],orientation_default[2],orientation_default[3]]]

    print('current index:',index)
    cartesian_plan, fraction = tutorial.plan_cartesian_path(tutorial.txyz[index])
    tutorial.execute_plan(cartesian_plan)

    ################ gripper part ######################
    if index == 2:
    	print('')
        #psd.moveTo(psd.FLG_ZERO, 0)
        #psd.moveTo(psd.FLG_POS, 70, None, False) 
        
    elif index == 6:
        if tutorial.is_gripper_open == False:
            #psd.moveTo(psd.FLG_ZERO, 0)
            #psd.moveTo(psd.FLG_NEG, 50, None, False)         
            tutorial.is_gripper_open = True
        tutorial.get_data()
        if len(tutorial.cam_res) > 1:           
            if tutorial.cam_res[4] == 0.0 and tutorial.cam_res[5] == 0.0:
                print ("============ Post Place, Conveyor Empty ...")            
                print(tutorial.cam_res[4:7])  
                rospy.sleep(1)
                continue
            else:
                tutorial.last_cam_res = tutorial.cam_res                
 
        else:
            print ("============  Post Place, Pending for data ...")                
            rospy.sleep(1)
            continue        
    
    index += 1
    #if index > 7:
    if index >= len(tutorial.txyz):
        index = 0
        tutorial.is_gripper_open = False
    
    rate.sleep() 