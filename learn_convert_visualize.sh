#!/bin/bash

th script.lua -use_cuda
th imagesAndReprToTxt.lua
python generateNNImages.py 10 25
#python report_results.py
path=`cat lastModel.txt | grep Log`
nautilus $path
