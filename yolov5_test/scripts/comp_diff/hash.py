#! /usr/bin/python3.8

#from skimage.measure import structural_similarity as ssim
from skimage.metrics import structural_similarity as ssim
#https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
from PIL import Image
import imagehash

def ahash(imageA, imageB):
	hashA = imagehash.average_hash(imageA)
	hashB = imagehash.average_hash(imageB)
	return hashA - hashB
def phash(imageA, imageB):
	hashA = imagehash.phash(imageA)
	hashB = imagehash.phash(imageB)
	return hashA - hashB
def dhash(imageA, imageB):
	hashA = imagehash.dhash(imageA)
	hashB = imagehash.dhash(imageB)
	return hashA - hashB
def whash(imageA, imageB):
	hashA = imagehash.whash(imageA)
	hashB = imagehash.whash(imageB)
	return hashA - hashB
def whash_db4(imageA, imageB):
	hashA = imagehash.whash(imageA, mode='db4')
	hashB = imagehash.whash(imageB, mode='db4')
	return hashA - hashB
def colorhash(imageA, imageB):
	hashA = imagehash.colorhash(imageA)
	hashB = imagehash.colorhash(imageB)
	return hashA - hashB

def compare_images(imageA, imageB, title):

	t0 = time.time()
	hash_a = ahash(imageA, imageB)
	t1 = time.time()
	hash_p = phash(imageA, imageB)
	hash_d = dhash(imageA, imageB)
	hash_w = whash(imageA, imageB)
	hash_w4 = whash_db4(imageA, imageB)
	hash_c = colorhash(imageA, imageB)
	print('Average hashing diff=', hash_a,'-- (%.3fs)' % (t1 - t0))
	print('Perceptual hashing diff=',hash_p)
	print('Difference hashing diff=',hash_d)
	print('Wavelet hashing diff=',hash_w)
	print('Wavelet db4 hashing diff=',hash_w4)
	print('HSV color hashing diff=',hash_w)
	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("ahash: %.2f, phash: %.2f, dhash: %.2f, \nwhash: %.2f, w4hash: %.2f, chash: %.2f" % (hash_a, hash_p,hash_d,hash_w,hash_w4,hash_c))
	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA)
	plt.axis("off")
	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB)
	plt.axis("off")
	# show the images
	#plt.show()
	fig.savefig('diff.jpg')


# load the images 
image0 = Image.open('./images/0.jpg').resize((480,640))
image00 = Image.open('./images/00.jpg').resize((480,640))
image01 = Image.open('./images/01.jpg').resize((480,640))
image1 = Image.open('./images/1.jpg').resize((480,640))
image2 = Image.open('./images/2.jpg').resize((480,640))
print(image0.size)
# initialize the figure
fig = plt.figure("Images")
images = ("image0", image0), ("image00", image00), ("image1", image1)
# loop over the images
for (i, (name, image)) in enumerate(images):
	# show the image
	ax = fig.add_subplot(1, 3, i + 1)
	ax.set_title(name)
	plt.imshow(image, cmap = plt.cm.gray)
	plt.axis("off")
# show the figure
#plt.show()
fig.savefig('all.jpg')

# compare the images
compare_images(image0, image01, "image0 vs. image01")
#compare_images(original, contrast, "Original vs. Contrast")
#compare_images(original, shopped, "Original vs. Photoshopped")


