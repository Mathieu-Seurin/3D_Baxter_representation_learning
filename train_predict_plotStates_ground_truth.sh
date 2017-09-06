#!/bin/bash

# CONFIG OPTIONS:
# -use_cuda
# -use_continuous
# -params.sigma  is CONTINUOUS_ACTION_SIGMA
# -params.mcd is MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD
# -data_folder options: DATA_FOLDER (Dataset to use):
#          staticButtonSimplest, mobileRobot, simpleData3D, pushingButton3DAugmented, babbling')
#data= staticButtonSimplest, mobileRobot, complexData colorful, colorful75  #staticButtonSimplest https://stackoverflow.com/questions/2459286/unable-to-set-variables-in-bash-script  #"$data"='staticButtonSimplest'

function has_command_finished_correctly {
    if [ "$?" -ne "0" ]
    then
        exit
    else
        return 0
    fi
}

echo " ********** Running Denoising Conv. Autoencoder script: *************"
#qlua script.lua  -use_cuda -use_continuous -mcd $max_cos_dis -sigma $s -data_folder mobileRobot #complexData #colorful  #stati$
th supervised.lua -use_cuda -data_folder colorful75
has_command_finished_correctly

#th imagesAndReprToTxt.lua  -use_cuda -use_continuous -data_folder mobileRobot # complexData #colorful  #staticButtonSimplest
th imagesAndReprToTxt.lua  -use_cuda -data_folder colorful75 #mobileRobot
has_command_finished_correctly

python generateNNImages.py 10 # -1 uses a characteristic set of image for creating the neigbors for a GIF animation, REQUIRES SETTING DEFAULT_DATASET in Utils.py
#   ----- Note: includes the call to:
#                th create_all_reward.lua
#                th create_plotStates_file_for_all_seq.lua
has_command_finished_correctly

python plotStates.py # TODO SET HERE AUTOMATICALLY AS PARAM -use_ground_truth and in gnerateNNImages.py
has_command_finished_correctly

#python distortion_crit.py  # short to compute, it's just that it doesn't seem to be very useful
#has_command_finished_correctly

