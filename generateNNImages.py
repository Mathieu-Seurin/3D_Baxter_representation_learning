import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.neighbors import NearestNeighbors
import shutil
import random
import sys
import pandas as pd
import os, os.path
import subprocess
from Utils import ALL_STATE_FILE, LEARNED_REPRESENTATIONS_FILE, LAST_MODEL_FILE, GLOBAL_SCORE_LOG_FILE, get_data_folder_from_model_name
import unittest 
test = unittest.TestCase('__init__')




"""
NOTE, if sklearn.neighbours import fails, remove  and install:
Either use conda (in which case all your installed packages would be in ~/miniconda/ or pip install --user don't mix the two. Removing either 
rm -rf ~/.local/lib/python2.7/site-packages/sklearn or your ~/miniconda folder and reinstalling it cleanly should fix this.
sudo rm -rf scikit_learn-0.18.1.egg-info/
pip uninstall sklearn
and
1)  pip install -U scikit-learn
or 2) conda install -c anaconda scikit-learn=0.18.1
If needed, also do 
pip install -U numpy
pip install -U scipy

""" 

if len(sys.argv) <= 1:
    sys.exit("Give number of neighbors to produce, followed by number of input images (and model dir if you don't want to use the last model created)")

# Some parameters
nbr_neighbors= int(sys.argv[1])
nbr_images = -1


if len(sys.argv) >= 3:
    nbr_images=int(sys.argv[2])

if len(sys.argv) == 4:
    path_to_model = sys.argv[3]
else:
    lastModelFile = open(LAST_MODEL_FILE)
    path_to_model = lastModelFile.readline()[:-1]
data_folder = get_data_folder_from_model_name(path_to_model)

# THE FOLLOWING ONLY WILL RUN IN USE_CUDA false way
#subprocess.call(['th','create_plotStates_file_for_all_seq.lua'])  
#subprocess.call(['th','create_all_reward.lua'])
subprocess.call(['th','create_plotStates_file_for_all_seq.lua'])  
subprocess.call(['th','create_all_reward.lua'])

file_representation_string=path_to_model+"/"+LEARNED_REPRESENTATIONS_FILE


#Parsing representation file
#===================

#reading data
file_representation  = open(file_representation_string, "r")

#parsing
images=[]
representations=[]
for line in file_representation:
    if line[0]!='#':
        words = line.split()
        images.append(words[0])
        representations.append(words[1:])

#Parsing true state file
#===================
true_states = {}
file_state = open(ALL_STATE_FILE, "r")

for line in file_state:
    if line[0]!='#':
        words = line.split()
        true_states[words[0]] = np.array(map(float,words[1:]))

#Compute nearest neighbors
nbrs = NearestNeighbors(n_neighbors=(nbr_neighbors+1), algorithm='ball_tree').fit(representations)
distances, indexes = nbrs.kneighbors(representations)

#Generate mosaics
path_to_neighbour = path_to_model + '/NearestNeighbors/'
last_model_name = path_to_model.split('/')[-1]
print "path_to_model: ",path_to_model 
print "path_to_neighbour: ",path_to_neighbour
#shutil.rmtree('NearestNeighbors', 1)
if not os.path.exists(path_to_neighbour):
	os.mkdir(path_to_neighbour)

if nbr_images == -1:
	data= zip(images,indexes,distances,representations)
else:
	data= random.sample(zip(images,indexes,distances,representations),nbr_images)


# For each random selected images (or all images in nbr_images==-1), you take
# the k-nearest neighbour in the REPRESENTATION SPACE (the first argv parameter)

#As a quantitative measure, for the k nearest neighbour
#you compute the distance between the state of the original image and 
#the images retrieved using knn on representation space

total_error = 0 # to assess the quality of repr
nb_tot_img = 0

for img_name,id,dist,state in data:
	base_name= os.path.splitext(os.path.basename(img_name))[0]
	seq_name= img_name.split("/")[1]
	print('Processing ' + seq_name + "/" + base_name + ' ...')
	fig = plt.figure()
	fig.set_size_inches(6*(nbr_neighbors+1), 6)
	a=fig.add_subplot(1,nbr_neighbors+1,1)
	a.axis('off')
	img = mpimg.imread(img_name)
	imgplot = plt.imshow(img)
	state_str='[' + ",".join(['{:.3f}'.format(float(x)) for x in state]) + "]"
	a.set_title(seq_name + "/" + base_name + ": \n" + state_str)

	original_coord = true_states[img_name]

	for i in range(0,nbr_neighbors):
		a=fig.add_subplot(1,nbr_neighbors+1,i+2)
		img_name=images[id[i+1]]
		img = mpimg.imread(img_name)
		imgplot = plt.imshow(img)

		base_name_n= os.path.splitext(os.path.basename(img_name))[0]
		seq_name_n= img_name.split("/")[1]

		dist_str = ' d=' + '{:.4f}'.format(dist[i+1])

		state_str='[' + ",".join(['{:.3f}'.format(float(x)) for x in representations[id[i+1]]]) + "]"
		a.set_title(seq_name_n + "/" + base_name_n + ": \n" + state_str +dist_str)
		a.axis('off')

        neighbour_coord = true_states[img_name]
        total_error += np.linalg.norm(neighbour_coord-original_coord)
        nb_tot_img += 1


	plt.tight_layout()
	output_file = path_to_neighbour + seq_name + "_" + base_name
        
	plt.savefig(output_file, bbox_inches='tight')
	plt.close() # efficiency: avoids keeping all images into RAM

mean_error = total_error/nb_tot_img  #print "MEAN MSE ERROR : ", str(mean_error)[:5]

score_file = path_to_model+'/scoreNN.txt'
f = open(score_file,'w')
f.write(str(mean_error)[:5])
f.close()

# writing scores to global log for plotting and reporting
header = ['Model', 'KNN_MSE']
d = {'Model':[last_model_name], 'KNN_MSE': [mean_error]}	
global_scores_df = pd.DataFrame(data=d, columns = header) #global_scores_df.reset_index() 

if not os.path.isfile(GLOBAL_SCORE_LOG_FILE):
    global_scores_df.to_csv(GLOBAL_SCORE_LOG_FILE, header=True) 
else: # it exists so append without writing the header
    global_scores_df.to_csv(GLOBAL_SCORE_LOG_FILE, mode ='a', header=False) 
    
print 'Saved mean KNN MSE score entry from model \n++++++++ ', last_model_name, ' ++++++++\n to ', GLOBAL_SCORE_LOG_FILE, '. Last score is in: ',score_file 
print global_scores_df.head()