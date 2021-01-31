import numpy as np

import torch
import torch.utils.data

import random
# import matplotlib.pyplot as plt


class GraspDatasetBase(torch.utils.data.Dataset):
    """
    An abstract dataset for training GG-CNNs in a common format.
    """
    def __init__(self, output_size=300, include_depth=True, include_rgb=False, random_rotate=False,
                 random_zoom=False, input_only=False):
        """
        :param output_size: Image output size in pixels (square)
        :param include_depth: Whether depth image is included
        :param include_rgb: Whether RGB image is included
        :param random_rotate: Whether random rotations are applied
        :param random_zoom: Whether random zooms are applied
        :param input_only: Whether to return only the network input (no labels)
        """
        self.output_size = output_size
        self.random_rotate = random_rotate
        self.random_zoom = random_zoom
        self.input_only = input_only
        self.include_depth = include_depth
        self.include_rgb = include_rgb

        self.grasp_files = []

        if include_depth is False and include_rgb is False:
            raise ValueError('At least one of Depth or RGB must be specified.')
        # print('input_only = ',input_only)

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_depth(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_rgb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def __getitem__(self, idx):
        if self.random_rotate:
            rotations = [0, np.pi/2, 2*np.pi/2, 3*np.pi/2]
            rot = random.choice(rotations)
        else:
            rot = 0.0

        if self.random_zoom:
            zoom_factor = np.random.uniform(0.5, 1.0)
        else:
            zoom_factor = 1.0

        # Load the depth image
        if self.include_depth:
            depth_img = self.get_depth(idx, rot, zoom_factor)
            # fig = plt.figure(figsize=(5, 5))
            # fig.suptitle(idx, fontsize=10)
            # ax = fig.add_subplot(1, 1, 1)
            # ax.imshow(depth_img, cmap='gray')
            # plt.show()

        # Load the RGB image
        if self.include_rgb:
            rgb_img = self.get_rgb(idx, rot, zoom_factor)

        # Load the grasps
        if not self.input_only:
            bbs = self.get_gtbb(idx, rot, zoom_factor)
            pos_img, ang_img, width_img = bbs.draw((self.output_size, self.output_size))

            # fig = plt.figure(figsize=(15, 6))
            # fig.suptitle(idx, fontsize=10)
            # ax = fig.add_subplot(1, 5, 1)
            # ax.imshow(pos_img, cmap='gray')
            # ax = fig.add_subplot(1, 5, 2)
            # ax.imshow(ang_img, cmap='gray')
            # ax = fig.add_subplot(1, 5, 3)
            # ax.imshow(width_img, cmap='gray')
            # ax = fig.add_subplot(1, 5, 4)
            # ax.imshow(depth_img, cmap='gray')
            # ax = fig.add_subplot(1, 5, 5)
            # ax.imshow(rgb_img.transpose((1,2,0))*255)
            

            width_img = np.clip(width_img, 0.0, 150.0)/150.0
            pos = self.numpy_to_torch(pos_img)
            cos = self.numpy_to_torch(np.cos(2*ang_img))
            sin = self.numpy_to_torch(np.sin(2*ang_img))
            width = self.numpy_to_torch(width_img)

        if self.include_depth and self.include_rgb:
            x = self.numpy_to_torch(
                np.concatenate(
                    (np.expand_dims(depth_img, 0),
                     rgb_img),
                    0
                )
            )
        elif self.include_depth:
            x = self.numpy_to_torch(depth_img)
        elif self.include_rgb:
            x = self.numpy_to_torch(rgb_img)

        # print(depth_img.shape,self.include_depth,self.include_rgb,'shape of x:',x.shape,'shape of pos:',pos.shape,'idx：',idx,'zoom：',zoom_factor,'rot：',rot)
        # plt.show()

        if not self.input_only:
            return x, (pos, cos, sin, width), idx, rot, zoom_factor
        else:  # no labels
            return x, idx, rot, zoom_factor 

    def __len__(self):
        return len(self.grasp_files)
