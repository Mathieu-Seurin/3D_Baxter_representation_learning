#!/bin/bash

function has_command_finish_correctly {
    if [ "$?" -ne "0" ]
    then
        exit
    else
        return 0
    fi
}


echo 'WARNING, check that model=BASE_TIMNET and NORMALIZE=True'

echo 'Log/save/supervised_res' > lastModel.txt
echo 'mobile_robot_supervised.t7' >> lastModel.txt
th imagesAndReprToTxt.lua
has_command_finish_correctly

python generateNNImages.py 10
has_command_finish_correctly

python plotStates.py
has_command_finish_correctly

path=`cat lastModel.txt | grep Log`
nautilus $path
