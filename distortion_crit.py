#!/usr/bin/python
#coding: utf-8

from __future__ import division
import numpy as np
import random

import sys

from Utils import parse_true_state_file, parse_repr_file, LAST_MODEL_FILE,get_data_folder_from_model_name,LEARNED_REPRESENTATIONS_FILE

from sklearn.metrics.pairwise import euclidean_distances


#File should be called after generateNNImages

if len(sys.argv) >= 2:
    print "file used", sys.argv[1]
    path_to_model = sys.argv[1]
else:
    lastModelFile = open(LAST_MODEL_FILE)
    path_to_model = lastModelFile.readline()[:-1]

data_folder = get_data_folder_from_model_name(path_to_model)
file_representation_string=path_to_model+"/"+LEARNED_REPRESENTATIONS_FILE
images, representations = parse_repr_file(file_representation_string)

true_states = parse_true_state_file()


# convert learned states to numpy
learned_states = np.asarray([map(float,x) for x in representations])

# compute distance matrix for learned states
learned_distances = euclidean_distances(learned_states)
learned_non_zero =learned_distances[learned_distances.nonzero()]
print 'Learned states : min dist ',learned_non_zero.min(), ' max dist : ', learned_non_zero.max(), ' mean dist : ', learned_non_zero.mean()


# convert true states to numpy
ref_states = np.asarray([true_states[x] for x in images])

# compute distance matrix for ref states
ref_distances = euclidean_distances(ref_states)
ref_non_zero =ref_distances[ref_distances.nonzero()]
print 'True states : min dist ',ref_non_zero.min(), ' max dist : ', ref_non_zero.max(), ' mean dist : ', ref_non_zero.mean()


# compute relative distance
ref_distances[ref_distances==0]=1 # in order avoid division by zero
coefs = np.triu(np.divide(learned_distances,ref_distances)) # keep only upper triangle


# find min/max
coefs[coefs==0]=np.Inf # remove diagonal
global_min_coef = coefs.min()  # global minimum
local_min_coef = coefs[ref_distances <0.1].min() # min for points with true state dis < 0.1
far_min_coef = coefs[ref_distances >0.1].min() # min for points with true state dis > 0.1

coefs[coefs==np.Inf]=-np.Inf
global_max_coef = coefs.max()
local_max_coef = coefs[ref_distances <0.1].max()
far_max_coef = coefs[ref_distances >0.1].max()

score_file = open(path_to_model+'/scoreNN.txt','a')


global_distortion =  global_max_coef/global_min_coef
global_str = '\nGlobal Distortion : '+str(global_distortion)
print global_str
score_file.write(global_str+'\n')


local_distortion = local_max_coef/local_min_coef
local_str = "local distortion : "+str(local_min_coef)
print local_str
score_file.write(local_str+'\n')

far_distortion = far_max_coef/far_min_coef
far_str = 'Far Distortion : '+str(far_distortion)
print far_str
score_file.write(far_str+'\n')

score_file.close()


