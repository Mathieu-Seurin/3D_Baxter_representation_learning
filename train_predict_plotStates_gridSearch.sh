#!/bin/bash

# CONFIG OPTIONS:
# -use_cuda
# -use_continuous
# -params.sigma  is CONTINUOUS_ACTION_SIGMA
# -params.mcd is MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD
# -data_folder options: DATA_FOLDER (Dataset to use):
#          staticButtonSimplest, mobileRobot, simpleData3D, pushingButton3DAugmented, babbling')
#data= staticButtonSimplest, mobileRobot, complexData colorful  #staticButtonSimplest https://stackoverflow.com/questions/2459286/unable-to-set-variables-in-bash-script  #"$data"='staticButtonSimplest'

function has_command_finished_correctly {
    if [ "$?" -ne "0" ]
    then
        exit
    else
        return 0
    fi
}

# losses result in being nan for MCD 0.9 and sigma 0.01
#for max_cos_dis in 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9  #for max_cos_dis in 0.9
for max_cos_dis in 0.1 0.2 0.4 0.5 0.6 0.8 0.9 0.95 #0.4 0.5 0.8
do
    #for s in  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    for s in 0.1 0.2 0.4 0.5 0.6 0.8 0.9 #0.2 0.4 0.5
    do
        echo " ********** Running pipeline for finetuning mcd: $max_cos_dis and sigma: $s *************"
        qlua script.lua  -use_cuda -use_continuous -mcd $max_cos_dis -sigma $s -data_folder mobileRobot #complexData #colorful  #stati$
        has_command_finished_correctly

        th imagesAndReprToTxt.lua  -use_cuda -use_continuous -data_folder mobileRobot # complexData #colorful  #staticButtonSimplest
        has_command_finished_correctly

        python generateNNImages.py 10
        #   ----- Note: includes the call to:
        #                th create_all_reward.lua
        #                th create_plotStates_file_for_all_seq.lua
        has_command_finished_correctly

        python plotStates.py
        has_command_finished_correctly

        python report_results.py
        has_command_finished_correctly

        python distortion_crit.py
        has_command_finished_correctly
    done
done

# best so far in a 49 images dataset: modelY2017_D24_M06_H02M02S49_staticButtonSimplest_resnet_cont_MCD0_5_S0_1,0.222667244673
