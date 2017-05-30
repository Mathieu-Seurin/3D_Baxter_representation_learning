#!/bin/bash

th script.lua
th imagesAndReprToTxt.lua
python generateNNImages.py 10 40
path=`cat lastModel.txt | grep Log`
nautilus $path

