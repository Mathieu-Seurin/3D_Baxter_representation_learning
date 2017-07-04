#!/usr/bin/python
#coding: utf-8

from __future__ import division
import numpy as np
import random

import sys

from Utils import parse_true_state_file, parse_repr_file, LAST_MODEL_FILE,get_data_folder_from_model_name,LEARNED_REPRESENTATIONS_FILE


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

data=zip(images, representations)
data1 = random.sample(data,len(images))
data2 = random.sample(data,len(images))

for a in range(1,500):
    data1 = data1 + random.sample(data, len(images))
    data2 = data2 + random.sample(data, len(images))

print len(data1)


min_dist= 1e8
max_dist=0
sum_dist=0
nb_dist=0

for a in zip(data1,data2):
    i1=a[0][0]
    i2=a[1][0]
    r1=np.asarray(map(float,a[0][1]))
    r2=np.asarray(map(float,a[1][1]))

    s1=true_states[i1]
    s2=true_states[i2]

    dist_true=np.linalg.norm(s1-s2)
    dist_repr=np.linalg.norm(r1-r2)

    if (dist_true>0.1) & (dist_repr>0.1):
        dist = dist_true / dist_repr

        sum_dist = sum_dist + dist
        nb_dist=nb_dist+1
        if (dist<min_dist):
            min_dist=dist
        if (dist>max_dist):
            max_dist=dist


print 'Distorsion :', max_dist/min_dist

print 'Mean Distorsion :' , (sum_dist/nb_dist)/min_dist