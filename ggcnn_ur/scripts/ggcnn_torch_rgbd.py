from os import path
import time
import warnings
import math

import socket
import pickle
import select

import sys
import geometry_msgs.msg
import cv2
import numpy as np
import scipy.ndimage as ndimage

import torch
import realsense_subscriber as realsense
from utils.dataset_processing import evaluation_k as evaluation#
# from ggcnn.srv import GraspPrediction, GraspPredictionResponse
from ggcnn_ur.msg import Grasp
#from dougsm_helpers.timeit import TimeIt
# from tf import transformations as tft




MODEL_FILE = './output/models/210223_1204_tryold/epoch_01_iou_0.98'
#'./output/bottle_second_210203_1444_training_bottle/epoch_03_iou_1.00' 


here = path.dirname(path.abspath(__file__))
sys.path.append(here)
print(path.join(path.dirname(__file__), MODEL_FILE))
model = torch.load(path.join(path.dirname(__file__), MODEL_FILE))
print(model)
device = torch.device("cuda:0")
print(device)

_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}
_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())
_NEXT_AXIS = [1, 2, 0, 1]


def quaternion_from_euler(ai, aj, ak, axes='sxyz'):
    """Return quaternion from Euler angles and axis sequence.
    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple
    >>> q = quaternion_from_euler(1, 2, 3, 'ryxz')
    >>> numpy.allclose(q, [0.310622, -0.718287, 0.444435, 0.435953])
    True
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    quaternion = np.empty((4, ), dtype=np.float64)
    if repetition:
        quaternion[i] = cj*(cs + sc)
        quaternion[j] = sj*(cc + ss)
        quaternion[k] = sj*(cs - sc)
        quaternion[3] = cj*(cc - ss)
    else:
        quaternion[i] = cj*sc - sj*cs
        quaternion[j] = cj*ss + sj*cc
        quaternion[k] = cj*cs - sj*sc
        quaternion[3] = cj*cc + sj*ss
    if parity:
        quaternion[j] *= -1

    return quaternion


def list_to_quaternion(l):
    q = geometry_msgs.msg.Quaternion()
    q.x = l[0]
    q.y = l[1]
    q.z = l[2]
    q.w = l[3]
    return q


def process_depth_image(depth, RGB, crop_size, out_size=300, return_mask=False, crop_y_offset=0):
    imh, imw = depth.shape
    imh, imw, imd = RGB.shape

    # with TimeIt('1'):
    # Crop.
    depth_crop = depth[(imh - crop_size) // 2 - crop_y_offset:(imh - crop_size) // 2 + crop_size - crop_y_offset,
                           (imw - crop_size) // 2:(imw - crop_size) // 2 + crop_size]
    RGB_crop = RGB[(imh - crop_size) // 2 - crop_y_offset:(imh - crop_size) // 2 + crop_size - crop_y_offset,
                           (imw - crop_size) // 2:(imw - crop_size) // 2 + crop_size,
                           (imd - crop_size) // 2:(imd - crop_size) // 2 + crop_size]


    # inpaint box background
    # fack_bg_rgb = [174,149,136] # 
    # RGB_crop[0:125,:] = fack_bg_rgb
    # RGB_crop[380:480,:] = fack_bg_rgb
    # RGB_crop[:,0:170] = fack_bg_rgb
    # RGB_crop[:,510:640] = fack_bg_rgb

    # RGB_crop.normalise()
    RGB_norm = RGB_crop.astype(np.float32)/255.0
    RGB_norm -= RGB_norm.mean()
    RGB_norm = RGB_norm.transpose((2, 0, 1))

    # print('RGB shape=',RGB_crop.shape)
    # cv2.imshow('1',RGB_crop)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print('depth range=',np.amax(depth_crop), np.amin(depth_crop))
    # fake_depth = np.average(depth_crop[60:70,260:280])
    # # print('average depth impaint=',fake_depth)
    # depth_crop[0:42,:] = fake_depth

    # fake_depth = 1095 #887.0 for box # a fixed value to make all depth imgs consistent
    # depth_arr_crop[0:125,:] = fake_depth
    # depth_arr_crop[380:480,:] = fake_depth
    # depth_arr_crop[:,0:170] = fake_depth
    # depth_arr_crop[:,510:640] = fake_depth

    # depth_nan_mask = np.isnan(depth_crop).astype(np.uint8)

    # Inpaint
    # OpenCV inpainting does weird things at the border.
    # with TimeIt('2'):
    depth_crop = cv2.copyMakeBorder(depth_crop, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    # deal with negative depth value
    # depth_crop[np.where(depth_crop<=0)] = np.nan
    # depth_crop[np.where(depth_crop>1.3)] = np.nan
    depth_nan_mask = np.isnan(depth_crop).astype(np.uint8)

    # with TimeIt('3'):
    depth_crop[depth_nan_mask==1] = 0

    # with TimeIt('4'):
    # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
    depth_scale = np.abs(depth_crop).max()
    depth_crop = depth_crop.astype(np.float32) / depth_scale  # Has to be float32, 64 not supported.

    # with TimeIt('Inpainting'):
    depth_crop = cv2.inpaint(depth_crop, depth_nan_mask, 1, cv2.INPAINT_NS)

    # inpaint for negative depth    

    # Back to original size and value range.
    depth_crop = depth_crop[1:-1, 1:-1]
    depth_crop = depth_crop * depth_scale

    # with TimeIt('5'):
    # Resize
    depth_crop = cv2.resize(depth_crop, (out_size, out_size), cv2.INTER_AREA)

    if return_mask:
    # with TimeIt('6'):
        depth_nan_mask = depth_nan_mask[1:-1, 1:-1]
        depth_nan_mask = cv2.resize(depth_nan_mask, (out_size, out_size), cv2.INTER_NEAREST)
        return depth_crop, depth_nan_mask, RGB_norm, RGB_crop
    else:
        return depth_crop


def predict(depth_arr, RGB, process_depth=True, crop_size=300, out_size=300, depth_nan_mask=None, crop_y_offset=0, filters=(2.0, 1.0, 1.0)):
    if process_depth:
        depth_crop, depth_nan_mask, RGB_norm, RGB_crop = process_depth_image(depth_arr, RGB, crop_size, out_size=out_size, return_mask=True, crop_y_offset=crop_y_offset)
        

    # Inference
    depth = np.clip((depth_crop - depth_crop.mean()), -1, 1)
    depthT = torch.from_numpy(np.expand_dims(np.concatenate((np.expand_dims(depth, 0),RGB_norm),0), 0).astype(np.float32)).to(device)
    # depthT = torch.from_numpy(depth.reshape(1, 1, out_size, out_size).astype(np.float32)).to(device)
    with torch.no_grad():
        pred_out = model(depthT) #expected input[1, 1, 300, 300] to have 4 channels

    points_out = pred_out[0].cpu().numpy().squeeze()
    points_out[depth_nan_mask] = 0

    # Calculate the angle map.
    cos_out = pred_out[1].cpu().numpy().squeeze()
    sin_out = pred_out[2].cpu().numpy().squeeze()
    ang_out = np.arctan2(sin_out, cos_out) / 2.0

    # Calculate the width map.
    width_out = pred_out[3].cpu().numpy().squeeze() * 150.0  # Scaled 0-150:0-1

    # Calculate the k map.
    # k_out = pred_out[4].cpu().numpy().squeeze()
    k_out = points_out

    # Filter the outputs.
    if filters[0]:
        points_out = ndimage.filters.gaussian_filter(points_out, filters[0])  # 3.0
    if filters[1]:
        ang_out = ndimage.filters.gaussian_filter(ang_out, filters[1])
    if filters[2]:
        width_out = ndimage.filters.gaussian_filter(width_out, filters[2])
    # no filter applied to k_out

    points_out = np.clip(points_out, 0.0, 1.0-1e-3)

    # SM
    # temp = 0.15
    # ep = np.exp(points_out / temp)
    # points_out = ep / ep.sum()
    # points_out = (points_out - points_out.min())/(points_out.max() - points_out.min())
    # return depth.squeeze(), RGB_crop, points_out, ang_out, width_out, k_out


    #post-process
    g = post_processing(depth_arr, depth_crop, RGB_crop, points_out, ang_out, width_out, k_out, crop_size=300, crop_y_offset=0)
    print('g',g)

    # Vis results
    if False:
        evaluation.plot_output(RGB_crop, depth_crop, points_out, ang_out, no_grasps=1, grasp_width_img=width_out, k_img=k_out)
        # evaluation.plot_output(RGB_crop, depth_crop, points_out, ang_out, no_grasps=1, grasp_width_img=width_out)

    return g








def post_processing(depth, depth_crop, RGB, points, angle, width_img, k_img, crop_size=300, crop_y_offset=0):

    # cam_info_topic = '/camera/aligned_depth_to_color/camera_info'
    K = [609.674560546875, 0.0, 323.9862365722656, 0.0, 608.5648193359375, 227.5126495361328, 0.0, 0.0, 1.0]
    cam_K = np.array(K).reshape((3, 3))
    # print('cam_K ',cam_K)

    # angle -= np.arcsin(camera_rot[0, 1])  # Correct for the rotation of the camera
    angle = (angle + np.pi/2) % np.pi - np.pi/2  # Wrap [-np.pi/2, np.pi/2]
    
    # Convert to 3D positions.
    imh, imw = depth.shape
    # print('imh, imw ',imh, imw)
    x = ((np.vstack((np.linspace((imw - crop_size) // 2, (imw - crop_size) // 2 + crop_size, depth_crop.shape[1], np.float), )*depth_crop.shape[0]) - cam_K[0, 2])/cam_K[0, 0] * depth_crop).flatten()
    y = ((np.vstack((np.linspace((imh - crop_size) // 2 - crop_y_offset, (imh - crop_size) // 2 + crop_size - crop_y_offset, depth_crop.shape[0], np.float), )*depth_crop.shape[1]).T - cam_K[1,2])/cam_K[1, 1] * depth_crop).flatten()
    # pos = np.dot(camera_rot, np.stack((x, y, depth_crop.flatten()))).T + np.array([[cam_p.x, cam_p.y, cam_p.z]])

    cam_fov = 65.5
    width_m = width_img / 300.0 * 2.0 * depth_crop * np.tan(cam_fov * crop_size/depth.shape[0] / 2.0 / 180.0 * np.pi) #something wrong here

    # find the best grasp in quality map(original method)
    best_g = np.argmax(points) #find where the first max is in the quality map
    best_g_unr = np.unravel_index(best_g, points.shape) #best grasp pixel coord 
    print('best at(original method):',best_g_unr)

    # find the best grasp in quality map(my method)
    # maxwhere = np.where(points == np.amax(points))
    # listOfCordinates = list(zip(maxwhere[0], maxwhere[1])) #find where are all max are in the quality map #
    # #find the geometry center of all max if there are multiple max
    # best_g_unr = (int(sum([p[0] for p in listOfCordinates]) / len(listOfCordinates)), int(sum([p[1] for p in listOfCordinates]) / len(listOfCordinates)))
    # print('best at(center method):',best_g_unr)


    # from 2D-pixel-frame to 3D-camera-frame
    depth_best = depth_crop[best_g_unr]*1000.0 #m->mm
    print('depth_best:', depth_best)
    xc = best_g_unr[1] + (imw-crop_size)/2.0  # from (300*300) to (640*480) in pixel
    yc = best_g_unr[0] + (imh-crop_size)/2.0
    xyz_grasp = np.dot(np.linalg.inv(cam_K), depth_best * np.array([xc,yc,1]).transpose())  # XYZ (mm) in camera frame
    print('xyz_grasp:',xyz_grasp)
    # xyz_grasp = xyz_grasp.transpose()



    # return a Grasp
    # ret = GraspPredictionResponse()
    # print('ret',ret)
    # ret.success = True
    # g = ret.best_grasp
    # g = geometry_msgs.msg.Pose()
    g = Grasp()
    # g.pose.position.x = pos[best_g, 0]
    # g.pose.position.y = pos[best_g, 1]
    # g.pose.position.z = pos[best_g, 2]
    # g.pose.orientation = tfh.list_to_quaternion(tft.quaternion_from_euler(np.pi, 0, ((angle[best_g_unr]%np.pi) - np.pi/2)))

    g.pose.position.x = xyz_grasp[0]
    g.pose.position.y = xyz_grasp[1]
    g.pose.position.z = xyz_grasp[2]
    print(' orientation: ', ((angle[best_g_unr]%np.pi) - np.pi/2) /np.pi*180.0 )
    # g.pose.orientation = list_to_quaternion(quaternion_from_euler(np.pi, 0, ((angle[best_g_unr]%np.pi) - np.pi/2))) # in camera frame
    g.pose.orientation = list_to_quaternion(quaternion_from_euler(0, np.pi/2, ((angle[best_g_unr]%np.pi) - np.pi/2))) # in rough base frame
    g.width = width_m[best_g_unr] # original claculation method: too small value
    g.width = width_img[best_g_unr] #direct method: maybe pixel unit
    g.quality = points[best_g_unr]
    # g.k = k_img[best_g_unr]
    g.angle = ((angle[best_g_unr]%np.pi) - np.pi/2) /np.pi*180.0
    return g


def ggcnn_predict():

    img_arr, depth_arr = realsense.get_image(show=False) #array RGB
    depth_arr = depth_arr/1000.0  # mm->m (important!)
    # print(np.shape(depth_arr), np.shape(img_arr), np.amax(depth_arr), np.amin(depth_arr), depth_arr)

    # time.sleep(0.1)
    with torch.no_grad():
        time_s = time.time()
        g = predict(depth_arr, img_arr, process_depth=True, crop_size=300, out_size=300, depth_nan_mask=None, crop_y_offset=0, filters=(2.0, 1.0, 1.0)) 
        time_e = time.time()          
        print('runtime =', time_e-time_s)

    return g





if __name__ == '__main__':
    while True:
        # time_s = time.time()
        grasp = ggcnn_predict()
        # time_e = time.time()
        # print('runtime =', time_e-time_s) #roughly 15Hz from taking img to outputing a grasp

        # UDP send out of python3 env
        client = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)            
        ip_port = ('127.0.0.1', 9999)
        client.sendto(pickle.dumps(grasp,protocol=2),ip_port) #send   
        time.sleep(0.2)

    client.close() 




    
