#!/bin/bash
th script.lua -use_continuous
th imagesAndReprToTxt.lua
python generateNNImages.py 10 25
#   ----- includes the call to:
#                th create_all_reward.lua
#                th create_plotStates_file_for_all_seq.lua
python plotStates.py
python report_results.py
