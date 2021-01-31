'''
The dataset should in the following format：
--file_path
  --*
    --*r.png
    --*d.tiff
    --**r.png
    --**d.tiff
  --*
    --*r.png
    --*d.tiff
    --**r.png
    --**d.tiff
  ...
  
'''
import os
import glob

from .grasp_data import GraspDatasetBase
from utils.dataset_processing import grasp, image


class MyDataset3(GraspDatasetBase):
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
        super(MyDataset3, self).__init__(**kwargs)

        # graspf = glob.glob(os.path.join(file_path, '*', '*_GRB.png')) #if there is an extra folder
        graspf = glob.glob(os.path.join(file_path, '*', '*r.png'))  #if images are directly under the path
        graspf.sort()
        l = len(graspf)
        print('at: ',file_path,'found *.png ', l)

        if l == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        if ds_rotate:
            graspf = graspf[int(l*ds_rotate):] + graspf[:int(l*ds_rotate)]

        # *_perfect_depth.tiff: a float32 tiff image with the perfect depth image.
        depthf = [f.replace('r.png', 'd.tiff') for f in graspf]
        # *_RGB.png: a png image of the scene rendered with Blender.
        # rgbf = [f.replace('perfect_depth.tiff', 'RGB.png') for f in depthf]
        rgbf = graspf

        self.grasp_files = graspf[int(l*start):int(l*end)]
        self.depth_files = depthf[int(l*start):int(l*end)]
        self.rgb_files = rgbf[int(l*start):int(l*end)]

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        gtbbs = grasp.GraspRectangles.load_from_jacquard_file(self.grasp_files[idx], scale=self.output_size / 1024.0)
        c = self.output_size//2
        gtbbs.rotate(rot, (c, c))
        gtbbs.zoom(zoom, (c, c))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
        print(self.depth_files[idx])
        depth_img.rotate(rot)
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img

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
