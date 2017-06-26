#!/bin/bash


function has_command_finish_correctly {
    if [ "$?" -ne "0" ]
    then
        exit
    else
        return 0
    fi
}

th rmLast.lua
has_command_finish_correctly

th learn_autoencoder.lua -use_cuda
has_command_finish_correctly

th imagesAndReprToTxt.lua -use_cuda
has_command_finish_correctly

python generateNNImages.py 10
has_command_finish_correctly

# TODO: add flag -skip_rendering for the plotStates and generateNNImages to work on grid search remotely via ssh
python plotStates.py
has_command_finish_correctly

python report_results.py
has_command_finish_correctly

path=`cat lastModel.txt | grep Log`
nautilus $path

