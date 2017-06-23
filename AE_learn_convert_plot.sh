#!/bin/bash

th learn_autoencoder.lua -use_cuda
th imagesAndReprToTxt.lua -use_cuda
python generateNNImages.py 10
python plotStates.py
python report_results.py
path=`cat lastModel.txt | grep Log`
nautilus $path
