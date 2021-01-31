'''
The dataset should in the following format：
--file_path
  --*.jpg (RGD image)
  --**.jpg
  ...
  
'''
import os
import glob
import numpy as np
import cv2

from .grasp_data import GraspDatasetBase
from utils.dataset_processing import grasp, image


class MyDataset2(GraspDatasetBase):
    """
    Dataset wrapper for the Customized dataset.
    """
    def __init__(self, file_path, start=0.0, end=1.0, ds_rotate=0, **kwargs):
        """
        :param file_path:  Dataset directory.
        :param start: If splitting the dataset, start at this fraction [0,1]
        :param end: If splitting the dataset, finish at this fraction
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(MyDataset2, self).__init__(**kwargs)

        # graspf = glob.glob(os.path.join(file_path, '*', '*_GRB.png')) #if there is an extra folder
        graspf = glob.glob(os.path.join(file_path, '*.jpg'))  #if images are directly under the path
        graspf.sort()
        l = len(graspf)
        print('at: ',file_path,'found *.jpg ', l)

        if l == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        if ds_rotate:
            graspf = graspf[int(l*ds_rotate):] + graspf[:int(l*ds_rotate)]

        # *_perfect_depth.tiff: a float32 tiff image with the perfect depth image.
        depthf = graspf
        # *_RGB.png: a png image of the scene rendered with Blender.
        # rgbf = [f.replace('perfect_depth.tiff', 'RGB.png') for f in depthf]
        rgbf = graspf

        self.grasp_files = graspf[int(l*start):int(l*end)]
        self.depth_files = depthf[int(l*start):int(l*end)]
        self.rgb_files = rgbf[int(l*start):int(l*end)]

        
    def process_depth_image(self, depth, crop_size=300, out_size=300, return_mask=True, crop_y_offset=0):
        imh, imw = depth.shape
        # Crop.
        depth_crop = depth[(imh - crop_size) // 2 - crop_y_offset:(imh - crop_size) // 2 + crop_size - crop_y_offset,
                               (imw - crop_size) // 2:(imw - crop_size) // 2 + crop_size]
        print('crop:',(imh - crop_size) // 2 - crop_y_offset,(imh - crop_size) // 2 + crop_size - crop_y_offset,(imw - crop_size) // 2,(imw - crop_size) // 2 + crop_size)
        # Inpaint
        # OpenCV inpainting does weird things at the border.
        depth_crop = cv2.copyMakeBorder(depth_crop, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        depth_nan_mask = np.isnan(depth_crop).astype(np.uint8)
        kernel = np.ones((3, 3),np.uint8)
        depth_nan_mask = cv2.dilate(depth_nan_mask, kernel, iterations=1)
        depth_crop[depth_nan_mask==1] = 0
        # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
        depth_scale = np.abs(depth_crop).max()
        depth_crop = depth_crop.astype(np.float32) / depth_scale  # Has to be float32, 64 not supported.
        depth_crop = cv2.inpaint(depth_crop, depth_nan_mask, 1, cv2.INPAINT_NS)
        # Back to original size and value range.
        depth_crop = depth_crop[1:-1, 1:-1]
        depth_crop = depth_crop * depth_scale
        # Resize
        depth_crop = cv2.resize(depth_crop, (out_size, out_size), cv2.INTER_AREA)
        if return_mask:
            depth_nan_mask = depth_nan_mask[1:-1, 1:-1]
            depth_nan_mask = cv2.resize(depth_nan_mask, (out_size, out_size), cv2.INTER_NEAREST)
            return depth_crop, depth_nan_mask
        else:
            return depth_crop
          
    def get_gtbb(self, idx, rot=0, zoom=1.0):
        gtbbs = grasp.GraspRectangles.load_from_jacquard_file(self.grasp_files[idx], scale=self.output_size / 1024.0)
        c = self.output_size//2
        gtbbs.rotate(rot, (c, c))
        gtbbs.zoom(zoom, (c, c))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
#         depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
#         print(self.depth_files[idx])
#         depth_img.rotate(rot)
#         depth_img.normalise()
#         depth_img.zoom(zoom)
#         depth_img.resize((self.output_size, self.output_size))
        rgd_img = image.Image.from_file(self.depth_files[idx])
        # print(self.depth_files[idx])
        rgd_img.rotate(rot)
        rgd_img.zoom(zoom)
        rgd_img.normalise()
        rgd_img.img = rgd_img.img.transpose((2, 0, 1)) #DGR-RGD
        # print(rgd_img.img.shape)
        depth = rgd_img.img[2,:,:] # take depth channel
        # print(depth.shape)
        depth_processed, depth_nan_mask = self.process_depth_image(depth)
        depth_processed = np.clip((depth_processed - depth_processed.mean()), -1, 1)
        # depth_reshaped = depth_processed.reshape((1, 300, 300)) # not sure?
        return depth_processed

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_img = image.Image.from_file(self.rgb_files[idx])
        rgb_img.rotate(rot)
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img

    def get_jname(self, idx):
        # idx_nameOfObject (e.g. 4_ffe702c059d0fe5e6617a7fd9720002b)
        # idx = {0,1,2,3,4} , each idx represents a different shotting angle of the same object
        return '_'.join(self.grasp_files[idx].split(os.sep)[-1].split('_')[:-1])
