#!/bin/bash

function has_command_finish_correctly {
    if [ "$?" -ne "0" ]
    then
        exit
    else
        return 0
    fi
}

th script.lua -use_cuda
has_command_finish_correctly
th imagesAndReprToTxt.lua -use_cuda
has_command_finish_correctly
python generateNNImages.py 10 25
python plotStates.py
path=`cat lastModel.txt | grep Log`
nautilus $path

