#!/bin/bash

th script.lua
th imagesAndReprToTxt.lua
python generateNNImages.py 10 25
path=`cat lastModel.txt | grep Log`
nautilus $path

