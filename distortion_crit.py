#!/usr/bin/python
#coding: utf-8

from __future__ import division
import numpy as np

import sys

from Utils import parse_true_state_file, parse_repr_file, LAST_MODEL_FILE


#File should be called after generateNNImages

if len(sys.argv) >= 2:
    print "file used", sys.argv[1]
    path_to_model = sys.arg[1]
else:
    lastModelFile = open(LAST_MODEL_FILE)
    path_to_model = lastModelFile.readline()[:-1]

data_folder = get_data_folder_from_model_name(path_to_model)
file_representation_string=path_to_model+"/"+LEARNED_REPRESENTATIONS_FILE
images, representations = parse_repr_file(file_representation_string)

true_states = parse_true_state_file()

