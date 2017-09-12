#!/bin/bash

# NOTE This script assumes the model folders have already the output files after running the following two scripts for each model
#qlua script.lua  -use_cuda  -mcd $max_cos_dis -sigma $s -data_folder complexData # mobileRobot #complexData #colorful  #stati$
#th imagesAndReprToTxt.lua  -use_cuda -data_folder complexData #mobileRobot # complexData #colorful  #staticButtonSimplest
# where: 
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

# REQUIREMENT TO RUN THIS SCRIPT: THE MODEL FOLDER MUST CONTAIN THE EXACT NAME OF THE DATASET IT USES FOR RUNNING WHOLE PIPELINE SMOOTHLY
# AFTER RUNNING THIS SCRIPT, COPY THE MODEL FOLDERS TO ./ TO BE ABLE TO RUN makeMovieFromComparingKNNAcrossModels.py
# AND AFTER, EITHER RUN http://gifmaker.me/ with ./Mosaics or run Utils.create_1_row_mosaic.py

#### OPTIONS TO RUN THIS SCRIPT DEPENDING ON YOUR FAVOURITE DATASET:
# A) COLORFUL75 datasets 
for path_to_model in './Log/colorful75_0.196_button_ref' './Log/colorful75_supervised' './Log/colorful75_3AEmodelY2017_D06_M09_H18M14S22_colorful75_autoencoder_conv_cont_MCD0_5_S0_1_ProTemCauRepFix' #'./Log/colorful75_0.013_ground_truth' #   

# B) 3D (STATIC_BUTTON_SIMPLEST) dataset     './Log/3D_0.097_AE_staticButtonSimplest' gives core dump? / 3D_0.053_fix_butt_15ep_staticButtonSimplest gives  KNN_MSE score for given neighbors:  0.070309485136
# ./generate_neighbors_for_all_models_movie.sh: line 61: unexpected EOF while looking for matching `"'
# ./generate_neighbors_for_all_models_movie.sh: line 66: syntax error: unexpected end of file
#for path_to_model in  './Log/3D_0.053_fix_butt_15ep_staticButtonSimplest' './Log/3D_0.097_AE_staticButtonSimplest' './Log/3D_0.03_supervised_staticButtonSimplest'  

# C) COMPLEX_DATA  complexDataset
#for path_to_model in './Log/complexData_0.145_AE_res' './Log/complexData_0.071_supervised_bit_bad' './Log/complexData_0.078_fix_above_more_tol'  #'./Log/complexData_0.035_GT' 

# D) MOBILE_ROBOT 'mobileRobot'
#for path_to_model in  './Log/mobileRobot_1.7_AE' './Log/mobileRobot_0.185_supervised' './Log/mobileRobot_0.183_frozen0_dim20' #'./Log/mobileRobot_0.172_ground_truth' 

# E) ALL TOGETHER: ALL MODELS FROM ALL DATASETS
#for path_to_model in './Log/colorful75_0.196_button_ref' './Log/colorful75_supervised' './Log/colorful75_3AEmodelY2017_D06_M09_H18M14S22_colorful75_autoencoder_conv_cont_MCD0_5_S0_1_ProTemCauRepFix' './Log/3D_0.053_fix_butt_15ep_staticButtonSimplest' './Log/3D_0.097_AE_staticButtonSimplest' './Log/3D_0.03_supervised_staticButtonSimplest' './Log/complexData_0.145_AE_res' './Log/complexData_0.071_supervised_bit_bad' './Log/complexData_0.078_fix_above_more_tol'  './Log/mobileRobot_1.7_AE' './Log/mobileRobot_0.185_supervised' './Log/mobileRobot_0.183_frozen0_dim20' 
#'./Log/mobileRobot_0.172_ground_truth' './Log/colorful75_0.013_ground_truth' './Log/complexData_0.035_GT'
# GT left out for now:
do
    echo "***** generate_neighbors_for_all_models_movie.sh: Running neighbour generation for all models. Model: $path_to_model *****"
    python generateNNImages.py -1 $path_to_model 
    has_command_finished_correctly
done




# other best model names @ Mathieu's: NOTICE THEY NEED TO CONTAIN THE EXACT NAME OF THE DATASET FOLDER AND THE FILES BELOW DO NOT!
# /home/natalia/dream/baxter_representation_learning_3D/Log/3D_0.03_supervised
# /home/natalia/dream/baxter_representation_learning_3D/Log/3D_0.053_fix_butt_15ep
# /home/natalia/dream/baxter_representation_learning_3D/Log/3D_0.097_AE
# /home/natalia/dream/baxter_representation_learning_3D/Log/colorful75_0.013_ground_truth
# /home/natalia/dream/baxter_representation_learning_3D/Log/colorful75_0.196_button_ref
# /home/natalia/dream/baxter_representation_learning_3D/Log/complex_0.13_fix_rand_round0.06
# /home/natalia/dream/baxter_representation_learning_3D/Log/complex_0.035_groundTruth
# /home/natalia/dream/baxter_representation_learning_3D/Log/complex_0.071_supervised_bit_bad
# /home/natalia/dream/baxter_representation_learning_3D/Log/complex_0.078_fix_above_more_tol
# /home/natalia/dream/baxter_representation_learning_3D/Log/complex_0.145_AE_res
# /home/natalia/dream/baxter_representation_learning_3D/Log/complex bst078
# /home/natalia/dream/baxter_representation_learning_3D/Log/modelY2017_D26_M08_H20M07S28_colorful75_resnet_cont_MCD0_4_S0_3_ProTemCauRepFix
