#!/bin/bash


function has_command_finished_correctly {
    if [ "$?" -ne "0" ]
    then
        exit
    else
        return 0
    fi
}


path_to_model='natalia@ueiaar2:~/baxter_representation_learning_3D/Log'
#path_to_model='gpu_center@uei18:~/baxter_representation_learning_3D/Log'

# continuous models:
for model in 'modelY2017_D03_M08_H09M40S59_colorful75_resnet_cont_MCD0_3_S0_3_ProTemCauRep' 'modelY2017_D02_M09_H21M48S12_complexData_resnet_cont_MCD0_01_S0_9_ProTemCauRepFix'  'modelY2017_D02_M08_H01M24S57_complexData_resnet_cont_MCD0_01_S0_9_ProTemCauRep' #'modelY2017_D26_M08_H20M07S28_colorful75_resnet_cont_MCD0_4_S0_3_ProTemCauRepFix' 
# discrete models:
for model in 'modelY2017_D22_M08_H18M41S20_colorful75_resnet_ProTemCauRepFix' 'modelY2017_D21_M08_H15M41S48_colorful75_resnet_ProTemCauRep' 'modelY2017_D01_M09_H18M20S22_complexData_resnet_ProTemCauRepFix' 'modelY2017_D21_M08_H12M26S01_complexData_resnet_ProTemCauRep' 
do
    echo "***** Copying Model: $path_to_model *****"
    mkdir -p ./continuous_actions/$model/NearestNeighbors/ #./continuous_actions/$path_to_model/
    scp $path_to_model/$model/NearestNeighbors/*.* ./continuous_actions/$model/NearestNeighbors/
    scp $path_to_model/$model/*.png ./continuous_actions/$model
    has_command_finished_correctly
done
