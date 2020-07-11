
import tensorflow as tf
import os
import time
import sys
from matplotlib import pyplot as plt
from IPython import display
import datetime
import glob
from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import pi, exp, sqrt



database_path="./Database/FrontView_Houses" # Update These paths before using
database_original_path="./Database/FrontView_Houses/original"
database_path_base=database_path+"/base"

SAVE_IN_TRAIN=500

def MKDIR(Dir):
	if not os.path.isdir(Dir):
		os.mkdir(Dir)


MKDIR(database_path+"/train")
MKDIR(database_path+"/base")
MKDIR(database_path+"/test")


arr_og = glob.glob(database_original_path+"/*.jpg")



for index,item in enumerate(arr_og):
	image = Image.open(item)
	tf.keras.preprocessing.image.save_img(database_path_base+'/image'+str(index)+'.jpg',image)
	


arr = sorted(glob.glob(database_path_base+"/*.jpg"))


def GetMinimal(image_path):
	img = cv2.imread(image_path)
	kernel=np.array([[1 ,4, 7, 4, 1],
				[4 ,16, 26, 16, 4],
				[7 ,26, 41, 26, 7],
				[4 ,16, 26, 16, 4],
				[1 ,4, 7, 4, 1]])/273
	
	s, k = 1, 31
	probs = [exp(-z*z/(2*s*s))/sqrt(2*pi*s*s) for z in range(-k,k+1)] 
	kernel = np.outer(probs, probs)	
	dst = cv2.filter2D(img,-1,kernel)

	image = dst

	
	pixel_values = image.reshape((-1, 3))

	pixel_values = np.float32(pixel_values)

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.9)
	k = 10
	compactness, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

	# convert back to 8 bit values
	centers = np.uint8(centers)

	# flatten the labels array
	labels = labels.flatten()

	# convert all pixels to the color of the centroids
	segmented_image = centers[labels]

	# reshape back to the original image dimension
	segmented_image = segmented_image.reshape(image.shape)
	return segmented_image

for index,item in enumerate(arr):
	print(item)
	segmented_image=GetMinimal(item)

	tf.keras.preprocessing.image.save_img(item.replace("jpg","png"),segmented_image)
	#plt.imshow(segmented_image)
	#plt.savefig(database_path_base+'/image'+str(index)+'.png')


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


for index,item in enumerate(arr):
	im1=Image.open(item)
	im2=Image.open(item.replace("jpg","png"))
	
	result_im=get_concat_h(im1, im2)
	out_fol="/train"
	print (index)
	if index >SAVE_IN_TRAIN:
		out_fol="/test"
	result_im.save(database_path+out_fol+"/image_"+str(index)+".png")
	





