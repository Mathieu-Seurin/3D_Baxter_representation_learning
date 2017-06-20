#!/bin/bash

echo 'WARNING, check that model=BASE_TIMNET and NORMALIZE=True'

echo 'Log/save/supervised_res' > lastModel.txt
echo 'mobile_robot_supervised.t7' >> lastModel.txt
th imagesAndReprToTxt.lua
python generateNNImages.py 10 25
python plotStates.py
path=`cat lastModel.txt | grep Log`
nautilus $path

