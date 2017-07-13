#!/bin/bash
function has_command_finish_correctly {
    if [ "$?" -ne "0" ]
    then
        exit
    else
        return 0
    fi
}


if [ "$1" != "" ]; then
    DATA_FOLDER=$1
else
    echo No data folder given
    exit
fi

echo 'WARNING, check that model=BASE_TIMNET and NORMALIZE=True'

th supervised.lua -use_cuda -data_folder $DATA_FOLDER

th imagesAndReprToTxt.lua -use_cuda -data_folder $DATA_FOLDER
has_command_finish_correctly

python generateNNImages.py 10
has_command_finish_correctly

python plotStates.py
has_command_finish_correctly

path=`cat lastModel.txt | grep Log`
nautilus $path
