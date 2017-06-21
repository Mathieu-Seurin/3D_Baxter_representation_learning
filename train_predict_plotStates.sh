#!/bin/bash

# CONFIG OPTIONS:
# -use_cuda
# -use_continuous
# -params.sigma  is CONTINUOUS_ACTION_SIGMA
# -params.mcd is MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD
# -data_folder options: DATA_FOLDER (Dataset to use):
#          staticButtonSimplest, mobileRobot, simpleData3D, pushingButton3DAugmented, babbling')
th script.lua -use_continuous -use_cuda
#-mcd 0.8 -sigma 0.8
# -data_folder staticButtonSimplest
# -data_folder mobileRobot
th imagesAndReprToTxt.lua -use_continuous -use_cuda
python generateNNImages.py 10 25
#   ----- includes the call to:
#                th create_all_reward.lua
#                th create_pl8otStates_file_for_all_seq.lua
python plotStates.py
python report_results.py
