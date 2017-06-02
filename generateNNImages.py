import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
import shutil
import random
import sys

nbr_images = -1

if len(sys.argv) <= 1:
        sys.exit("Give number of neighbors (and number of images and model dir if you don't want to use the last model created")

# Some parameters
nbr_neighbors= int(sys.argv[1])

if len(sys.argv) >= 3:
        nbr_images=int(sys.argv[2])

if len(sys.argv) == 4:
        path_to_model = sys.argv[3]
else:
        lastModelFile = open('lastModel.txt')
        path_to_model = lastModelFile.readline()[:-1]

data_file=path_to_model+"/saveImagesAndRepr.txt"

#reading data
file  = open(data_file, "r")

images=[]
states=[]
for line in file:
        if line[0]!='#':
                words = line.split()
                images.append(words[0])
                states.append(words[1:])

dim_state= len(states[0])

#Compute nearest neighbors
nbrs = NearestNeighbors(n_neighbors=(nbr_neighbors+1), algorithm='ball_tree').fit(states)
distances, indexes = nbrs.kneighbors(states)

#Generate mosaics
path_to_neighbour = path_to_model + '/NearestNeighbors/'
print "path_to_neighbour",path_to_neighbour
#shutil.rmtree('NearestNeighbors', 1)
if not os.path.exists(path_to_neighbour):
	os.mkdir(path_to_neighbour)

if nbr_images == -1:
	data= zip(images,indexes,distances,states)
else:
	data= random.sample(zip(images,indexes,distances,states),nbr_images)


for img_name,id,dist,state in data:
	base_name= os.path.splitext(os.path.basename(img_name))[0]
	seq_name= img_name.split("/")[1]
	print('Processing ' + seq_name + "/" + base_name + '...')
	fig = plt.figure()
	fig.set_size_inches(6*(nbr_neighbors+1), 6)
	a=fig.add_subplot(1,nbr_neighbors+1,1)
	a.axis('off')
	img = mpimg.imread(img_name)
	imgplot = plt.imshow(img)
	state_str='[' + ",".join(['{:.3f}'.format(float(x)) for x in state]) + "]"
	a.set_title(seq_name + "/" + base_name + ": \n" + state_str)

	for i in range(0,nbr_neighbors):
		a=fig.add_subplot(1,nbr_neighbors+1,i+2)
		img_name=images[id[i+1]]
		img = mpimg.imread(img_name)
		imgplot = plt.imshow(img)

		base_name_n= os.path.splitext(os.path.basename(img_name))[0]
		seq_name_n= img_name.split("/")[1]

		dist_str = ' d=' + '{:.4f}'.format(dist[i+1])

		state_str='[' + ",".join(['{:.3f}'.format(float(x)) for x in states[id[i+1]]]) + "]"
		a.set_title(seq_name_n + "/" + base_name_n + ": \n" + state_str +dist_str)
		a.axis('off')

	plt.tight_layout()
	output_file = path_to_neighbour + seq_name + "_" + base_name + "_" + 'Neigbors.png'
	#print('Saved nearest neighbor image to '+output_file)
	plt.savefig(output_file,bbox_inches='tight')
	plt.close() # efficiency: avoids keeping all images into RAM
