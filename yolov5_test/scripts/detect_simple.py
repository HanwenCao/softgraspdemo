#! /usr/bin/python3.8
import realsense_subscriber as realsense
import argparse
import time
import numpy as np

import sys
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages')

import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized



def detect(opt,img_arr,depth_arr):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size

    img_shape = depth_arr.shape  
    x1,y1,x2,y2,depth_avg = 0,0,0,0,0
    xyz_obj = np.array([0,0,0])
    K = np.array([[609.674560546875, 0.0, 323.9862365722656], [0.0, 608.5648193359375, 227.5126495361328], [0.0, 0.0, 1.0]])  # intrinsic 
    

    # convert
    img_pad = letterbox(img_arr[:, :, ::-1], new_shape=imgsz)[0] # first to BGR and padding
    img_pad = img_pad[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img_pad = np.ascontiguousarray(img_pad)  # 将一个内存不连续存储的数组转换为内存连续存储的数组


    # Initialize
    set_logging()
    device = select_device(opt.device)
    #if os.path.exists(out):
        #shutil.rmtree(out)  # delete output folder
    #os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA


    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16


    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


    # Run inference 
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    img = img_pad
    im0s = img_arr[:, :, ::-1]  # BGR
    path= './inference/images/snap.jpg'
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    # Inference
    pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)


    # Process detections
    how_far_to_center = []
    xyz_objs = []
    cls_names = []

    for i, det in enumerate(pred):  # detections per image

        p, s, im0 = path, '', im0s
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            
            #for c in det[:, -1].unique():
                #n = (det[:, -1] == c).sum()  # detections per class
                #s += '%g %ss, ' % (n, names[int(c)])  # add to string
            
            for *xyxy, conf, cls in reversed(det):
                # average depth
                x1,y1,x2,y2 = xyxy[0].cpu().numpy(),xyxy[1].cpu().numpy(),xyxy[2].cpu().numpy(),xyxy[3].cpu().numpy()
                '''x: horizontal axis; y: vertical axis; origin: upper left corner'''
                xc,yc = (x1+x2)/2, (y1+y2)/2
                dist_to_phoelectric = abs(xc-(img_shape[1]/2-210))  #center on topleft corner (with offset)
                how_far_to_center.append(dist_to_phoelectric)

                # estimate the depth of the object
                # a smaller patch (use this one or bbox itself)
                # x1c, x2c, y1c, y2c = int((x1+xc)/2), int((x2+xc)/2), int((y1+yc)/2), int((y2+yc)/2)
                depth_patch = depth_arr[int(y1):int(y2), int(x1):int(x2)]  # depth of bbox
                depth_patch = depth_patch.flatten()
                depth_patch = depth_patch[depth_patch!=0]  # depth inpainting could be better
                depth_patch = depth_patch[depth_patch>750]
                if depth_patch!=[]:
                    depth_patch = depth_patch[ (depth_patch>depth_patch.mean()-2*depth_patch.std()) & (depth_patch<depth_patch.mean()+2*depth_patch.std())]
                    if depth_patch!=[]:
                        depth_avg = depth_patch.mean() # depth option one
                        print('xc=',xc,'yc=',yc, 'depth=',depth_avg)
                        depth_min = depth_patch.min()
                        print('min depth:', depth_min)  # depth option two

                # from 2D to 3D
                xyz_obj = np.dot(np.linalg.inv(K), depth_avg * np.array([xc,yc,1]).transpose())  # XYZ (mm) in camera frame
                xyz_obj = xyz_obj.transpose()
                xyz_objs.append(xyz_obj)

                # label
                cls_names.append(int(cls))  # object classes
				# class name -> ['moon cake', 'mango', 'durian', 'pineapple']
                label = '%s %.2f' % (names[int(cls)], conf)
                im0 = np.array(im0) # this fix the error https://github.com/opencv/opencv/issues/18120
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)  # error

        # Visualization
        view_img=True
        if view_img:
            cv2.imshow(p, im0)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration

      
    # output one detection
    cls_obj = 0
    if len(how_far_to_center) != 0:        
        output_index = np.argmin(how_far_to_center)
        xyz_obj = xyz_objs[output_index] #XYZ in camera frame
        cls_obj = cls_names[output_index] #class  ['moon cake', 'mango', 'durian', 'pineapple', 'apple', 'pear', 'orange', 'lemon']

    return cls_obj, xyz_obj[0], xyz_obj[1], xyz_obj[2]  # class, X, Y, Z(mm)


def yolov5():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='last.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()

    while True:
        time.sleep(0.1)
        with torch.no_grad():
            img_arr, depth_arr = realsense.get_image(show=False) #input: array RGB, array Depth
            yolo_results = detect(opt, img_arr, depth_arr)
            print('yolo_results =', yolo_results) 
    return yolo_results



if __name__ == '__main__':
    
    yolo_results = yolov5()



      
                              
            
          
        
  


