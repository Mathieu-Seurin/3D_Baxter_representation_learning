import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.neighbors import NearestNeighbors
import shutil
import random
import sys
import pandas as pd
from PIL import Image
import os, os.path
import subprocess

from Utils import ALL_STATE_FILE, LEARNED_REPRESENTATIONS_FILE, LAST_MODEL_FILE, GLOBAL_SCORE_LOG_FILE, IMG_TEST_SET, COMPLEX_TEST_SET, STATIC_BUTTON_SIMPLEST, COMPLEX_DATA, MOBILE_ROBOT, ROBOT_TEST_SET, SUPERVISED, DEFAULT_DATASET, COLORFUL75, COMPLEX_DATA_MOVIE_TEST_SET, COLORFUL75_MOVIE_TEST_SET, STATIC_BUTTON_SIMPLEST_MOVIE_TEST_SET, COLORFUL_MOVIE_TEST_SET, MOBILE_ROBOT_MOVIE_TEST_SET
from Utils import get_data_folder_from_model_name, file2dict, parse_repr_file, parse_true_state_file, get_test_set_for_data_folder, get_movie_test_set_for_data_folder

import unittest
test = unittest.TestCase('__init__')



"""
NOTE, if sklearn.neighbours import fails, remove  and install:
Either use conda (in which case all your installed packages would be in ~/miniconda/ or pip install --user don't mix the two and do not use -U, nor sudo.
 Removing either
rm -rf ~/.local/lib/python2.7/site-packages/sklearn or your ~/miniconda folder and reinstalling it cleanly should fix this.
sudo rm -rf scikit_learn-0.18.1.egg-info/
pip uninstall sklearn
and
1)  pip install --user scikit-learn
or 2) conda install -c anaconda scikit-learn=0.18.1
If needed, also do
pip install --user numpy
pip install --user scipy

NOTE: Q: if this error is obtained: _tkinter.TclError: no display name and no $DISPLAY environment variable
A: Instead of ssh account@machine, do: ssh -X

Example to run this program for a given trained model:
python generateNNImages.py 5 5 Log/modelY2017_D24_M06_H06M19S10_staticButtonSimplest_resnet_cont_MCD0_8_S0_4
IMPORTANT: In order to run it with a non random fixed test set of images,
call it with only one argument (the number of neigbours to generate for each
image in the test set and it will assess the test set of 50 images defined in Const.lua and Utils.py)

"""

print"\n\n >> Running generateNNImages.py...."
if len(sys.argv) <= 1:
    sys.exit("Give number of neighbors to produce, followed by number of input images (and model dir if you don't want to use the last model created)")

# Some parameters
nbr_neighbors= int(sys.argv[1])
nbr_images = -1
use_test_set = True  
with_title = True


if nbr_neighbors == -1: # TODO FIX AND ADD MODEL NAME TO SUPERVISED!
	generating_neigbours_for_movie = True
	nbr_neighbors = 2 # for GIF creation purposes

	#TEST_SET = TEST_SET[:2]  # TODO TEST
	if len(sys.argv) != 3 and len(sys.argv) != 4:
		sys.exit('calling this program with first argument being -1 means we will use TEST_MOVIE test sets and you need to provide aftewrards, in the command line, the path to the model you want to build the neigbours for. Exiting...')
	else:
		path_to_model = sys.argv[2]
		if len(sys.argv) == 2:
			data_folder = DEFAULT_DATASET
		elif len(sys.argv) == 4:
			data_folder = sys.argv[3]
		else:
			sys.exit('calling this program with first argument being -1 means we will use TEST_MOVIE test sets and you need to provide aftewrards, in the command line, the path to the model you want to build the neigbours for. For a given data_folder, add it as a 4th argument. Exiting...')
	TEST_SET = get_movie_test_set_for_data_folder(data_folder)
	#TEST_SET = TEST_SET[:2] # just for fast testing!
else:
	generating_neigbours_for_movie = False
	if len(sys.argv) >= 3:
		nbr_images=int(sys.argv[2])
	if len(sys.argv) == 4:
	    path_to_model = sys.argv[3]
	    print """====================================
	    WARNING: DATASET IS SET BY HAND HERE  (IN ALL PYTHON SCRIPTS, take into account when running pipeline scripts such as gridsearch): MOBILE ROBOT FOR NOW
	    ============================================="""
	    data_folder = get_data_folder_from_model_name(path_to_model)
	else:
	    lastModelFile = open(LAST_MODEL_FILE)
	    path_to_model = lastModelFile.readline()[:-1]
	    data_folder = get_data_folder_from_model_name(path_to_model)
	    
	TEST_SET = get_test_set_for_data_folder(data_folder)



print "Using data_folder set by hand in all python scripts for SUPERVISED scripts. HERE DATA_FOLDER: ", data_folder, " Using path_to_model: ", path_to_model
if len(sys.argv) == 2:
	# We use fixed test set for fair comparison reasons
	use_test_set = True
	nbr_images = len(TEST_SET) 
    
if not generating_neigbours_for_movie:
	#THE FOLLOWING ONLY WILL RUN IN USE_CUDA false way  #print('Calling lua subprocesses with ',data_folder)
	subprocess.call(['th','create_plotStates_file_for_all_seq.lua','-use_cuda','-use_continuous','-data_folder', data_folder])  
	# TODO: READ CMD LINE ARGS FROM FILE INSTEAD (and set accordingly here) TO NOT HAVING TO MODIFY INSTEAD train_predict_plotStates and the python files
	subprocess.call(['th','create_all_reward.lua','-use_cuda','-use_continuous','-data_folder',data_folder])
# else, the files should exist


#Parsing representation file
#===================
file_representation_string=path_to_model+"/"+LEARNED_REPRESENTATIONS_FILE
images, representations = parse_repr_file(file_representation_string)

#Parsing true state file
#===================
true_states = parse_true_state_file(data_folder) #No need to send parameters, the const ALL_STATE_FILE is used

# Compute nearest neighbors
nbrs = NearestNeighbors(n_neighbors=(nbr_neighbors+1), algorithm='ball_tree').fit(representations)
distances, indexes = nbrs.kneighbors(representations)

#Generate mosaics
if generating_neigbours_for_movie:
	path_to_neighbour = path_to_model + '/NearestNeighborsGIFSeq/'
else:
	path_to_neighbour = path_to_model + '/NearestNeighbors/'
last_model_name = path_to_model.split('/')[-1]

print "path_to_model: ",path_to_model
print "path_to_neighbours: ",path_to_neighbour
#shutil.rmtree('NearestNeighbors', 1)
if not os.path.exists(path_to_neighbour):
	os.mkdir(path_to_neighbour)

if use_test_set or nbr_images == -1:
	data = zip(images,indexes,distances,representations)
	if len(set(images).intersection(TEST_SET)) == 0:
		sys.exit('Error in generateNNImages.py: the TEST_SET for this dataset has not been properly defined in Utils.py. TEST_SET must contain a subset of the full set of images in DATA_FOLDER => which in this case is:',data_folder)
else:
	print ('Using a random test set of images for KNN MSE evaluation...')
	data = random.sample(zip(images,indexes,distances,representations),nbr_images)


# For each random selected images (or all images in nbr_images==-1), you take
# the k-nearest neighbour in the REPRESENTATION SPACE (the first argv parameter)

#As a quantitative measure, for the k nearest neighbour
#you compute the distance between the state of the original image and
#the images retrieved using knn on representation space

total_error = 0 # to assess the quality of repr
nb_tot_img = 0

if nbr_neighbors<=5:
	numline = 1  # number of rows to show in the image of neigbours to be saved, for visibility
elif nbr_neighbors<=10:
	numline = 2
else:
	numline = 3

# TODO: more efficient: for img_name in test_set.keys() revising data above: 
# HOWEVER this needs to compute also in create_all_rewards and create_plotStates for the test set, separately and  an extra file. Is it fair comparison to test images for nearest neigbours that are seen during training?
print 'nbr_neighbours: ', nbr_neighbors, ' nbr of images: ', len(data), 'use_test_set ',use_test_set, ' of size: ', len(TEST_SET)#, TEST_SET
for img_name,neigbour_indexes,dist,state in data:
	if use_test_set: #		print img_name   colorful75/record_073/recorded_cameras_head_camera_2_image_compressed/frame00022.jpg
		if not(img_name in TEST_SET): 
			continue
	base_name= os.path.splitext(os.path.basename(img_name))[0]
	seq_name= img_name.split("/")[1]
	print('Processing ' + seq_name + "/" + base_name + ' ...'+base_name)
	fig = plt.figure()
	fig.set_size_inches(60,35)
	a=fig.add_subplot(numline+1,5,3)
	a.axis('off')
	# img = mpimg.imread(img_name)
	img = Image.open(img_name)
	imgplot = plt.imshow(img)
	state_str='[' + ",".join(['{:.3f}'.format(float(x)) for x in state]) + "]"

	original_coord = true_states[img_name]

	if with_title:
		a.set_title(seq_name + "/" + base_name + ": \n" + state_str + '\n' + str(original_coord))

	for i in range(0,nbr_neighbors):
		a=fig.add_subplot(numline+1,5,6+i)
		img_name=images[neigbour_indexes[i+1]]
		# img = mpimg.imread(img_name)
		img = Image.open(img_name)
		imgplot = plt.imshow(img)

		base_name_n= os.path.splitext(os.path.basename(img_name))[0]
		seq_name_n= img_name.split("/")[1]

		dist_str = ' d=' + '{:.4f}'.format(dist[i+1])

		state_str='[' + ",".join(['{:.3f}'.format(float(x)) for x in representations[neigbour_indexes[i+1]]]) + "]"
		neighbour_coord = true_states[img_name]
		total_error += np.linalg.norm(neighbour_coord-original_coord)
		nb_tot_img += 1

		if with_title:
			a.set_title(seq_name_n + "/" + base_name_n + ": \n" + state_str + dist_str + '\n' + str(neighbour_coord))
		a.axis('off')


	plt.tight_layout()
	output_file = path_to_neighbour + seq_name + "_" + base_name

	plt.savefig(output_file, bbox_inches='tight')
	plt.close() # efficiency: avoids keeping all images into RAM

mean_error = total_error/nb_tot_img  #print "MEAN MSE ERROR : ", str(mean_error)[:5]

score_file = path_to_model+'/scoreNN.txt'
f = open(score_file,'w')
f.write(str(mean_error)[:5])
f.close()
print 'KNN_MSE score for given neighbors: ', mean_error

if not generating_neigbours_for_movie:
	# writing scores to global log for plotting and reporting
	header = ['Model', 'KNN_MSE']
	d = {'Model':[last_model_name], 'KNN_MSE': [mean_error]}
	global_scores_df = pd.DataFrame(data=d, columns = header) #global_scores_df.reset_index()

	if not os.path.isfile(GLOBAL_SCORE_LOG_FILE):
		global_scores_df.to_csv(GLOBAL_SCORE_LOG_FILE, header=True)
	else: # it exists so append without writing the header
		global_scores_df.to_csv(GLOBAL_SCORE_LOG_FILE, mode ='a', header=False)

	print 'Saved mean KNN MSE score entry from model \n++ ', last_model_name, ' ++\n to ', GLOBAL_SCORE_LOG_FILE, '. Last score is in: ',score_file ,': KNN_MSE: \n'
	print global_scores_df.tail(20)

# try:
# 	distances, indexes = nbrs.kneighbors(representations)

# 	#Generate mosaics
# 	path_to_neighbour = path_to_model + '/NearestNeighbors/'
# 	last_model_name = path_to_model.split('/')[-1]

# 	print "path_to_model: ",path_to_model
# 	print "path_to_neighbours: ",path_to_neighbour
# 	#shutil.rmtree('NearestNeighbors', 1)
# 	if not os.path.exists(path_to_neighbour):
# 		os.mkdir(path_to_neighbour)

# 	if use_test_set or nbr_images == -1:
# 		data = zip(images,indexes,distances,representations)
# 	else:
# 		print ('Using a random test set of images for KNN MSE evaluation...')
# 		data = random.sample(zip(images,indexes,distances,representations),nbr_images)


# 	# For each random selected images (or all images in nbr_images==-1), you take
# 	# the k-nearest neighbour in the REPRESENTATION SPACE (the first argv parameter)

# 	#As a quantitative measure, for the k nearest neighbour
# 	#you compute the distance between the state of the original image and
# 	#the images retrieved using knn on representation space

# 	total_error = 0 # to assess the quality of repr
# 	nb_tot_img = 0

# 	if nbr_neighbors<=5:
# 		numline = 1  # number of rows to show in the image of neigbours to be saved, for visibility
# 	elif nbr_neighbors<=10:
# 		numline = 2
# 	else:
# 		numline = 3

# 	# TODO: more efficient: for img_name in test_set.keys() revising data above
# 	print 'nbr_neighbours: ', nbr_neighbors, ' nbr of images: ', len(data), 'use_test_set ',use_test_set, ' of size: ', len(TEST_SET)
# 	for img_name,neigbour_indexes,dist,state in data:
# 		if use_test_set:
# 			if not(img_name in TEST_SET): 
# 				continue
# 		base_name= os.path.splitext(os.path.basename(img_name))[0]
# 		seq_name= img_name.split("/")[1]
# 		print('Processing ' + seq_name + "/" + base_name + ' ...'+base_name)
# 		fig = plt.figure()
# 		fig.set_size_inches(60,35)
# 		a=fig.add_subplot(numline+1,5,3)
# 		a.axis('off')
# 		# img = mpimg.imread(img_name)
# 		img = Image.open(img_name)
# 		imgplot = plt.imshow(img)
# 		state_str='[' + ",".join(['{:.3f}'.format(float(x)) for x in state]) + "]"

# 		original_coord = true_states[img_name]

# 		if with_title:
# 			a.set_title(seq_name + "/" + base_name + ": \n" + state_str + '\n' + str(original_coord))

# 		for i in range(0,nbr_neighbors):
# 				a=fig.add_subplot(numline+1,5,6+i)
# 				img_name=images[neigbour_indexes[i+1]]
# 				# img = mpimg.imread(img_name)
# 				img = Image.open(img_name)
# 				imgplot = plt.imshow(img)

# 				base_name_n= os.path.splitext(os.path.basename(img_name))[0]
# 				seq_name_n= img_name.split("/")[1]

# 				dist_str = ' d=' + '{:.4f}'.format(dist[i+1])

# 				state_str='[' + ",".join(['{:.3f}'.format(float(x)) for x in representations[neigbour_indexes[i+1]]]) + "]"
# 				neighbour_coord = true_states[img_name]
# 				total_error += np.linalg.norm(neighbour_coord-original_coord)
# 				nb_tot_img += 1

# 				if with_title:
# 					a.set_title(seq_name_n + "/" + base_name_n + ": \n" + state_str + dist_str + '\n' + str(neighbour_coord))
# 				a.axis('off')


# 		plt.tight_layout()
# 		output_file = path_to_neighbour + seq_name + "_" + base_name

# 		plt.savefig(output_file, bbox_inches='tight')
# 		plt.close() # efficiency: avoids keeping all images into RAM

# 	mean_error = total_error/nb_tot_img  #print "MEAN MSE ERROR : ", str(mean_error)[:5]

# 	score_file = path_to_model+'/scoreNN.txt'
# 	f = open(score_file,'w')
# 	f.write(str(mean_error)[:5])
# 	f.close()

# 	# writing scores to global log for plotting and reporting
# 	header = ['Model', 'KNN_MSE']
# 	d = {'Model':[last_model_name], 'KNN_MSE': [mean_error]}
# 	global_scores_df = pd.DataFrame(data=d, columns = header) #global_scores_df.reset_index()

# 	if not os.path.isfile(GLOBAL_SCORE_LOG_FILE):
# 		global_scores_df.to_csv(GLOBAL_SCORE_LOG_FILE, header=True)
# 	else: # it exists so append without writing the header
# 		global_scores_df.to_csv(GLOBAL_SCORE_LOG_FILE, mode ='a', header=False)

# 	print 'Saved mean KNN MSE score entry from model \n++ ', last_model_name, ' ++\n to ', GLOBAL_SCORE_LOG_FILE, '. Last score is in: ',score_file ,' Latest 20 most recent scores computed:\n'
# 	print global_scores_df.tail(20)

# except:
# 	# Potential error: anaconda2/lib/python2.7/site-packages/sklearn/utils/validation.py", line 58, in _assert_all_finite
#     #" or a value too large for %r." % X.dtype)    ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
#     print "Unexpected error: This happens when training losses result in being nan (e.g. when sigma value is too small, e.g. 0.01 (for MCD 0.9). 0.1 precision magnitude is fine.", sys.exc_info()[0]
#     raise
