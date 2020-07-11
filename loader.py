import numpy as np
from PIL import Image
import os,glob
import matplotlib.pyplot as plt
from keras import layers
from keras import models
from keras import optimizers
import tensorflow as tf
#get path of the available image.
def generate_path(dir_original,dir_segmented):

	#get paths using glob
	paths_original = glob.glob(dir_original + "/*")
	paths_segmented = glob.glob(dir_segmented + "/*")

	#Error indication
	if (len(paths_original)*len(paths_segmented)==0):
		raise FileNotFoundError("Sorry, we could not find your files.Check path please.")

	filenames = list(map(lambda path: path.split(os.sep)[-1].split(".")[0],paths_segmented))
	paths_original = list(map(lambda filename: dir_original +"/"+filename+".jpg",filenames))

	return paths_original,paths_segmented


def image_generator(file_paths, init_size=(64,64), normalization = True):
	"""
	file_paths (list[string]):File paths you want to load.
	init_size (tuple(int,int)):We will resize the pics.
	normalization (bool):If true, we transfer the pixel value to [0,1].
	"""
	images = np.zeros((len(file_paths),init_size[0],init_size[1],3))
	i = 0
	for file_path in file_paths:
		if file_path.endswith(".png") or file_path.endswith(".jpg"):
			image = Image.open(file_path)
			#image = Loader.crop_to_square(image)
			if(init_size is not None and init_size !=image.size):
				image = image.resize(init_size)
			if(image.mode == "RGBA"):
				image = image.convert("RGB")
			image = np.asarray(image,dtype = np.float32)
			if normalization:
				image = image/255.0
			images[i] = image
		i+=1
	return images

def segment_generator(file_paths, init_size =(64,64),max_class = 22):
	"""
	file_paths (list[string]):File paths you want to load.
	init_size (tuple(int,int)):We will resize the pics.
	"""
	images = np.zeros((len(file_paths),init_size[0],init_size[1],max_class))
	i = 0
	for file_path in file_paths:
		if file_path.endswith(".png") or file_path.endswith(".jpg"):
			image = Image.open(file_path)
			if(init_size is not None and init_size !=image.size):
				image = image.resize(init_size)
		image = np.asarray(image,dtype = np.uint8)
		image = np.where(image ==255, max_class-1,image)

		identity = np.identity(max_class,dtype = np.uint8)
		image = identity[image]
		images[i] = image
		i+=1

	return images

#データローダの動きを確認
def test():
	path_a , path_b = generate_path("./VOCdataset/original","./VOCdataset/segmented")
	print("path_generation is done")
	im = image_generator(path_a)
	sg = segment_generator(path_b)
	print(im.shape,sg.shape)
	sum_or = True
	for i in range(sg.shape[0]):
		sum_or = sum_or or np.all(sg.sum(axis=-1)[i]==1)
	print(sum_or)
	#sns.heatmap(sg.sum(axis=-1)[20])
	#plt.show()
