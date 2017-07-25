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


th script.lua -use_cuda -data_folder $DATA_FOLDER
has_command_finish_correctly
th imagesAndReprToTxt.lua -use_cuda -data_folder $DATA_FOLDER
has_command_finish_correctly
python plotStates.py
has_command_finish_correctly
python generateNNImages.py 10
has_command_finish_correctly
python distortion_crit.py
path=`cat lastModel.txt | grep Log`
nautilus $path
