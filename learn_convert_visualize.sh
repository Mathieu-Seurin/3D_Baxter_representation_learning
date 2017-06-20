#!/bin/bash

th script.lua -use_cuda 
th imagesAndReprToTxt.lua
python generateNNImages.py 10 25
path=`cat lastModel.txt | grep Log`
nautilus $path
