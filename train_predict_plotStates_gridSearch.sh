#!/bin/bash

# CONFIG OPTIONS:
# -use_cuda
# -use_continuous
# -params.sigma  is CONTINUOUS_ACTION_SIGMA
# -params.mcd is MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD
# -data_folder options: DATA_FOLDER (Dataset to use):
#          staticButtonSimplest, mobileRobot, simpleData3D, pushingButton3DAugmented, babbling')

function has_command_finish_correctly {
    if [ "$?" -ne "0" ]
    then
        exit
    else
        return 0
    fi
}

for max_cos_dis in 0.1 0.2 0.4 0.5 0.8
do
    for s in 0.1 0.2 0.4 0.5 0.8
    do
        echo "\n ********** Running pipeline for finetuning mcd: $max_cos_dis and sigma: $s *************"
        th script.lua -use_cuda -use_continuous -mcd $max_cos_dis -sigma $s -data_folder staticButtonSimplest
        has_command_finish_correctly
        #  -data_folder staticButtonSimplest
        th imagesAndReprToTxt.lua -use_cuda -use_continuous -data_folder staticButtonSimplest
        has_command_finish_correctly

        python generateNNImages.py 10
        has_command_finish_correctly

        #   ----- includes the call to:
        #                th create_all_reward.lua
        #                th create_plotStates_file_for_all_seq.lua
        python plotStates.py
        has_command_finish_correctly

        python report_results.py
        has_command_finish_correctly

    done
done

# best so far in a 49 images dataset: modelY2017_D24_M06_H02M02S49_staticButtonSimplest_resnet_cont_MCD0_5_S0_1,0.222667244673
