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

nbr_images = 50

data_folder = get_data_folder_from_model_name(path_to_model)
file_representation_string=path_to_model+"/"+LEARNED_REPRESENTATIONS_FILE
images, representations = parse_repr_file(file_representation_string)



true_states = parse_true_state_file()

# convert learned states to numpy
learned_states = np.asarray([map(float,x) for x in representations])

# compute distance matrix for learned states
learned_distances = euclidean_distances(learned_states)


# convert true states to numpy
ref_states = np.asarray([true_states[x] for x in images])


# compute distance matrix for ref states
ref_distances = euclidean_distances(learned_states)


# compute relative distance
coefs = np.triu(np.divide(learned_distances,ref_distances))

print coefs

# TODO , ignore nans...
min_coef = coefs.nanmin()
max_coef = coefs.nanmax()

print min_coef
print max_coef

print 'Distorsion :', max_coef/min_coef

#print 'Mean Distorsion :' , (sum_dist/nb_dist)/min_dist