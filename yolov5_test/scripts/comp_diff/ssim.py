# from skimage.measure import compare_ssim as ssim
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2
import realsense_subscriber as realsense

def compare_ssim(imageA, imageB):
  s = ssim(imageA, imageB)
  return s

if __name__ == '__main__':
  # Load 2 background images
  original1 = cv2.imread('./comp_diff/images2/0.jpg')
  #original1 = cv2.imread('l0.jpg')
  original1_g = cv2.cvtColor(original1, cv2.COLOR_BGR2GRAY)

  #call realsense
  while True:
    img_arr = realsense.get_image(show=False)
    im_cur = PILImage.fromarray(img_arr,'RGB')
    im_cur_g = cv2.cvtColor(im_cur, cv2.COLOR_BGR2GRAY)
    #im_cur = cv2.imread('l2.jpg')
    #im_cur_g = cv2.cvtColor(im_cur, cv2.COLOR_BGR2GRAY)
    cur_thresh = compare_ssim(original1_g, im_cur_g)

    if cur_thresh < 0.45:
      print('Scene changing.Do YOLO.')
      break

  #object appear,do yolo
  yolo_results = detect()





