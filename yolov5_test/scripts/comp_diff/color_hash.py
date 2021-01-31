#! /usr/bin/python3.8
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
#import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
#import cv2
#sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
from PIL import Image as PILImage
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

def compare_hash(imageA, imageB, title=''):

	t0 = time.time()
	hash_a = ahash(imageA, imageB)
	#hash_p = phash(imageA, imageB)
	#hash_d = dhash(imageA, imageB)
	#hash_w = whash(imageA, imageB)
	#hash_w4 = whash_db4(imageA, imageB)
	hash_c = colorhash(imageA, imageB)
	t1 = time.time()
	#print('Average hashing diff=', hash_a,'-- (%.3fs)' % (t1 - t0))
	#print('Perceptual hashing diff=',hash_p)
	#print('Difference hashing diff=',hash_d)
	#print('Wavelet hashing diff=',hash_w)
	#print('Wavelet db4 hashing diff=',hash_w4)
	#print('HSV color hashing diff=',hash_c)
	return hash_c, hash_a


if __name__ == '__main__':
	# load the images 
	image0 = PILImage.open('./images/0.jpg').resize((480,640))
	image1 = PILImage.open('./images/1.jpg').resize((480,640))
	image2 = PILImage.open('./images/2.jpg').resize((480,640))

	# compare the images
	compare_hash(image0, image1, "image0 vs. image1")



