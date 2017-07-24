#!/bin/bash

th script.lua
th imagesAndReprToTxt.lua
python generateNNImages.py 10 25
python plotStates.py
path=`cat lastModel.txt | grep Log`
nautilus $path
