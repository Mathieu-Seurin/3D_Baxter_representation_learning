#!/bin/bash

# CONFIG OPTIONS:
# -data_folder options: Dataset to use:
#          staticButtonSimplest (default), mobileRobot, simpleData3D, pushingButton3DAugmented, babbling')
th script.lua -use_continuous -use_cuda
# -data_folder mobileRobot
th imagesAndReprToTxt.lua
python generateNNImages.py 10 25
#   ----- includes the call to:
#                th create_all_reward.lua
#                th create_pl8otStates_file_for_all_seq.lua
python plotStates.py
python report_results.py
