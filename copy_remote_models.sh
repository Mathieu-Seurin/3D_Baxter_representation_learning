#!/bin/bash


function has_command_finished_correctly {
    if [ "$?" -ne "0" ]
    then
        exit
    else
        return 0
    fi
}


# ALL GOOD MODELS USED: missing last gridsearch of mobile robot continuous
# /home/natalia/dream/baxter_representation_learning_3D/continuous_actions/modelY2017_D01_M09_H18M20S22_complexData_resnet_ProTemCauRepFix
# /home/natalia/dream/baxter_representation_learning_3D/continuous_actions/modelY2017_D02_M08_H01M24S57_complexData_resnet_cont_MCD0_01_S0_9_ProTemCauRep
# /home/natalia/dream/baxter_representation_learning_3D/continuous_actions/modelY2017_D02_M09_H21M48S12_complexData_resnet_cont_MCD0_01_S0_9_ProTemCauRepFix
# /home/natalia/dream/baxter_representation_learning_3D/continuous_actions/modelY2017_D03_M08_H09M40S59_colorful75_resnet_cont_MCD0_3_S0_3_ProTemCauRep
# /home/natalia/dream/baxter_representation_learning_3D/continuous_actions/modelY2017_D06_M09_H03M49S53_staticButtonSimplest_resnet_cont_MCD0_95_S0_8_ProTemCauRepFix
# /home/natalia/dream/baxter_representation_learning_3D/continuous_actions/modelY2017_D06_M09_H11M04S57_staticButtonSimplest_resnet_ProTemCauRepFix
# /home/natalia/dream/baxter_representation_learning_3D/continuous_actions/modelY2017_D20_M09_H10M18S27_mobileRobot_resnet_ProTemCauRepFix
# /home/natalia/dream/baxter_representation_learning_3D/continuous_actions/modelY2017_D20_M09_H11M25S33_colorful75_resnet_cont_MCD0_3_S0_3_ProTemCauRep
# /home/natalia/dream/baxter_representation_learning_3D/continuous_actions/modelY2017_D21_M08_H12M26S01_complexData_resnet_ProTemCauRep
# /home/natalia/dream/baxter_representation_learning_3D/continuous_actions/modelY2017_D21_M08_H15M41S48_colorful75_resnet_ProTemCauRep
# /home/natalia/dream/baxter_representation_learning_3D/continuous_actions/modelY2017_D21_M08_H16M35S27_mobileRobot_resnet_ProTemCauRep
# /home/natalia/dream/baxter_representation_learning_3D/continuous_actions/modelY2017_D21_M08_H17M09S55_staticButtonSimplest_resnet_ProTemCauRep
# /home/natalia/dream/baxter_representation_learning_3D/continuous_actions/modelY2017_D22_M08_H18M41S20_colorful75_resnet_ProTemCauRepFix
# /home/natalia/dream/baxter_representation_learning_3D/continuous_actions/modelY2017_D24_M08_H12M41S54_staticButtonSimplest_resnet_cont_MCD0_9_S0_6_ProTemCauRep
# /home/natalia/dream/baxter_representation_learning_3D/continuous_actions/modelY2017_D26_M08_H07M54S09_mobileRobot_resnet_cont_MCD0_4_S0_3_ProTemCauRep
# /home/natalia/dream/baxter_representation_learning_3D/continuous_actions/modelY2017_D26_M08_H20M07S28_colorful75_resnet_cont_MCD0_4_S0_3_ProTemCauRepFix

#

#path_to_project='natalia@ueiaar2:~/baxter_representation_learning_3D' # Tim's PC
path_to_project='gpu_center@uei18:~/baxter_representation_learning_3D'  # Mathieu's pc


# SAVE ALL STATS AND MODELS BENCHMARKS:
scp $path_to_project/allStats.csv ./continuous_actions/ # MOST IMPORTANT, SUMMARY OF OTHER 2
scp $path_to_project/globalScoreLog.csv ./continuous_actions/
scp $path_to_project/modelsConfigLog.csv ./continuous_actions/


path_to_model='natalia@ueiaar2:~/baxter_representation_learning_3D/Log' # Tim's PC
#path_to_model='gpu_center@uei18:~/baxter_representation_learning_3D/Log'  # Mathieu's pc


# continuous models:
#for model in 'modelY2017_D03_M08_H09M40S59_colorful75_resnet_cont_MCD0_3_S0_3_ProTemCauRep' 'modelY2017_D02_M09_H21M48S12_complexData_resnet_cont_MCD0_01_S0_9_ProTemCauRepFix'  'modelY2017_D02_M08_H01M24S57_complexData_resnet_cont_MCD0_01_S0_9_ProTemCauRep' #'modelY2017_D26_M08_H20M07S28_colorful75_resnet_cont_MCD0_4_S0_3_ProTemCauRepFix'
# discrete models:
#for model in 'modelY2017_D21_M08_H15M41S48_colorful75_resnet_ProTemCauRep' 'modelY2017_D01_M09_H18M20S22_complexData_resnet_ProTemCauRepFix' 'modelY2017_D21_M08_H12M26S01_complexData_resnet_ProTemCauRep' #'modelY2017_D22_M08_H18M41S20_colorful75_resnet_ProTemCauRepFix'
# ALL:
# for model in  'modelY2017_D21_M08_H15M41S48_colorful75_resnet_ProTemCauRep' 'modelY2017_D01_M09_H18M20S22_complexData_resnet_ProTemCauRepFix' 'modelY2017_D21_M08_H12M26S01_complexData_resnet_ProTemCauRep' 'modelY2017_D03_M08_H09M40S59_colorful75_resnet_cont_MCD0_3_S0_3_ProTemCauRep' 'modelY2017_D02_M09_H21M48S12_complexData_resnet_cont_MCD0_01_S0_9_ProTemCauRepFix'  'modelY2017_D02_M08_H01M24S57_complexData_resnet_cont_MCD0_01_S0_9_ProTemCauRep'
# do
#     echo "***** Copying Model: $path_to_model $model *****"
#     mkdir -p ./continuous_actions/$model/NearestNeighbors/ #./continuous_actions/$path_to_model/
#     scp $path_to_model/$model/NearestNeighbors/*.* ./continuous_actions/$model/NearestNeighbors/
#     scp $path_to_model/$model/*.png ./continuous_actions/$model
#     #has_command_finished_correctly
# done





# # Models at Tim's PC:
# /home/natalia/dream/baxter_representation_learning_3D/continuous_actions/modelY2017_D01_M09_H18M20S22_complexData_resnet_ProTemCauRepFix
# /home/natalia/dream/baxter_representation_learning_3D/continuous_actions/modelY2017_D02_M09_H21M48S12_complexData_resnet_cont_MCD0_01_S0_9_ProTemCauRepFix
# /home/natalia/dream/baxter_representation_learning_3D/continuous_actions/modelY2017_D03_M08_H09M40S59_colorful75_resnet_cont_MCD0_3_S0_3_ProTemCauRep
# /home/natalia/dream/baxter_representation_learning_3D/continuous_actions/modelY2017_D22_M08_H18M41S20_colorful75_resnet_ProTemCauRepFix
# /home/natalia/dream/baxter_representation_learning_3D/continuous_actions/modelY2017_D26_M08_H20M07S28_colorful75_resnet_cont_MCD0_4_S0_3_ProTemCauRepFix

# Models at Mathieu's:
# path_to_model='gpu_center@uei18:~/baxter_representation_learning_3D/Log'
# for model in 'modelY2017_D21_M08_H15M41S48_colorful75_resnet_ProTemCauRep' 'modelY2017_D21_M08_H12M26S01_complexData_resnet_ProTemCauRep' 'modelY2017_D03_M08_H09M40S59_colorful75_resnet_cont_MCD0_3_S0_3_ProTemCauRep' 'modelY2017_D02_M08_H01M24S57_complexData_resnet_cont_MCD0_01_S0_9_ProTemCauRep'
# do
#     echo "***** Copying Model: $path_to_model/$model *****"
#     mkdir -p ./continuous_actions/$model/NearestNeighbors/ #./continuous_actions/$path_to_model/
#     scp $path_to_model/$model/NearestNeighbors/*.* ./continuous_actions/$model/NearestNeighbors/
#     scp $path_to_model/$model/*.png ./continuous_actions/$model
# done


#for model in  'modelY2017_D28_M07_H14M34S42_mobileRobot_resnet_ProTemCauRepfix' 'modelY2017_D21_M08_H16M35S27_mobileRobot_resnet_ProTemCauRep' 'modelY2017_D26_M08_H07M54S09_mobileRobot_resnet_cont_MCD0_4_S0_3_ProTemCauRep' 'modelY2017_D24_M08_H12M41S54_staticButtonSimplest_resnet_cont_MCD0_9_S0_6_ProTemCauRep' 'modelY2017_D21_M08_H17M09S55_staticButtonSimplest_resnet_ProTemCauRep' 'modelY2017_D06_M09_H03M49S53_staticButtonSimplest_resnet_cont_MCD0_95_S0_8_ProTemCauRepFix' 'modelY2017_D06_M09_H11M04S57_staticButtonSimplest_resnet_ProTemCauRepFix'
#for model in 'modelY2017_D27_M07_H12M08S59_mobileRobot_resnet_cont_MCD0_2_S0_4_ProTemCauRep' 'modelY2017_D28_M07_H14M34S42_mobileRobot_resnet_ProTemCauRepfix'
for model in 'modelY2017_D20_M09_H15M24S19_mobileRobot_resnet_cont_MCD0_4_S0_3_ProTemCauRepFix'   # useful? 'modelY2017_D20_M09_H11M25S33_colorful75_resnet_cont_MCD0_3_S0_3_ProTemCauRep'
do
    echo "***** Copying Model: $path_to_model/$model *****"
    mkdir -p ./continuous_actions/$model/NearestNeighbors/ #./continuous_actions/$path_to_model/
    scp $path_to_model/$model/NearestNeighbors/*.* ./continuous_actions/$model/NearestNeighbors/
    scp $path_to_model/$model/*.png ./continuous_actions/$model

    # To save representations also:
    #scp $path_to_model/$model/saveImagesAndRepr.txt ./continuous_actions/$model

done
