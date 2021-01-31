import matplotlib.pyplot as plt
import imageio
from imageio import imsave
import tifffile as tif
import cv2
import numpy as np

def inpaint(img, missing_value=0):
    """
    Inpaint missing values in depth image.
    :param missing_value: Value to fill in teh depth image.
    """
    # cv2 inpainting doesn't handle the border properly
    # https://stackoverflow.com/questions/25974033/inpainting-depth-map-still-a-black-image-border
    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    mask = (img == missing_value).astype(np.uint8)

    # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
    scale = np.abs(img).max()
    img = img.astype(np.float32) / scale  # Has to be float32, 64 not supported.
    img = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)

    # Back to original size and value range.
    img = img[1:-1, 1:-1]
    img = img * scale

    return img

##Note
# "tif.imread" is the same as "imageio.imread"


img1 = imageio.imread("./inference/images/lemon1_depth.tiff") #from camera
img1_inpain = inpaint(img1)
print('max=',np.max(img1_inpain),'min=',np.min(img1_inpain),'first=',img1_inpain[0,0],img1_inpain.dtype,img1_inpain.shape)
plt.imshow(img1_inpain)
plt.savefig("./inference/images/lemon1_depth.png")

# img0 = imageio.imread("0.tiff")#from Jacquard dataset(pip install imagecodecs-lite)
# print('max=',np.max(img0),'min=',np.min(img0),'first=',img0[0,0],img0.dtype)
# plt.imshow(img0)
# plt.savefig("depth_fig0.png")


#test
# shape = (5,5)
# img = np.zeros(shape)
# img[0,0] = 5.1
# print('max=',np.max(img),'min=',np.min(img),'first=',img[0,0])
# imsave('test.tiff', img.astype(np.float32))

# img_read = imageio.imread("test.tiff")
# plt.imshow(img)
# plt.show()
# plt.savefig("depth_fig_test.png")

