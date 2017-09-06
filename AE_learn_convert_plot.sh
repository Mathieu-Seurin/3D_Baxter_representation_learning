#!/bin/bash

if [ "$1" != "" ]; then
    DATA_FOLDER=$1
else
    echo No data folder given
    exit
fi

function has_command_finish_correctly {
    if [ "$?" -ne "0" ]
    then
        exit
    else
        return 0
    fi
}

DATA_FOLDER='colorful75'

th learn_autoencoder.lua -use_cuda -data_folder $DATA_FOLDER
has_command_finish_correctly

th imagesAndReprToTxt.lua -use_cuda -data_folder $DATA_FOLDER
has_command_finish_correctly

python generateNNImages.py 10
has_command_finish_correctly

# TODO: add flag -skip_rendering for the plotStates and generateNNImages to work on grid search remotely via ssh solving Tkinker problem on this script

python plotStates.py
has_command_finish_correctly

python report_results.py
has_command_finish_correctly

path=`cat lastModel.txt | grep Log`
nautilus $path
