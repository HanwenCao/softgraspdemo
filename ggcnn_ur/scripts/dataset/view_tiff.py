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


img1 = imageio.imread("./testall0216/card/pcdcard7d.tiff") #from camera
print('max=',np.max(img1),'min=',np.min(img1),'first=',img1[0,0],img1.dtype,img1.shape)
plt.imshow(img1)
plt.show()
# plt.savefig("./inference/images/lemon1_depth.png")




#test
# shape = (5,5)
# img = np.zeros(shape)
# img[0,0] = 5.1
# print('max=',np.max(img),'min=',np.min(img),'first=',img[0,0])
# imsave('test.tiff', img.astype(np.float32))



