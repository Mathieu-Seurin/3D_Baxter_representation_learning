#!/bin/bash

# CONFIG OPTIONS:
# -use_cuda
# -use_continuous
# -params.sigma  is CONTINUOUS_ACTION_SIGMA
# -params.mcd is MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD
# -data_folder options: DATA_FOLDER (Dataset to use):
#          staticButtonSimplest, mobileRobot, simpleData3D, pushingButton3DAugmented, babbling')

for max_cos_dis in 0.1 0.2 0.4 0.5 0.8
do
    for s in 0.1 0.2 0.4 0.5 0.8
    do
        echo "\n ********** Running pipeline for finetuning mcd: $max_cos_dis and sigma: $s *************"
        th script.lua -use_continuous -use_cuda -mcd $max_cos_dis -sigma $s
        #  -data_folder staticButtonSimplest
        th imagesAndReprToTxt.lua -use_continuous -use_cuda
        python generateNNImages.py 10
        #   ----- includes the call to:
        #                th create_all_reward.lua
        #                th create_plotStates_file_for_all_seq.lua
        python plotStates.py
        python report_results.py

    done
done
