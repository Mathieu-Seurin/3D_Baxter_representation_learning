#!/bin/bash

th script.lua
th imagesAndReprToTxt.lua
python generateNNImages.py 40
path=`cat lastModel.txt | grep Log`
nautilus $path
