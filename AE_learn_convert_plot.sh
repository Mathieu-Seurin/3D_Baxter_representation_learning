#!/bin/bash

function has_command_finish_correctly {
    if [ "$?" -ne "0" ]
    then
        exit
    else
        return 0
    fi
}

th learn_autoencoder.lua -use_cuda
has_command_finish_correctly
th imagesAndReprToTxt.lua -use_cuda
has_command_finish_correctly
python generateNNImages.py 10 -use_cuda
has_command_finish_correctly
python plotStates.py
has_command_finish_correctly
python report_results.py
has_command_finish_correctly
path=`cat lastModel.txt | grep Log`
nautilus $path
